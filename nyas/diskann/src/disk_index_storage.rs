use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;

use rkyv::api::high::to_bytes_with_alloc;
use rkyv::ser::allocator::Arena;
use rkyv::{Archive, Deserialize, Serialize, from_bytes, rancor};
use system::metric::{Distance, MetricType};
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;
use tokio::fs::{File, OpenOptions};
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Mutex;

// use tracing::debug;
use crate::in_mem_index::InMemIndex;
use crate::pq_loader;
use crate::pq_loader::{MmappedCodes, MmappedPQ};
use crate::product_quantizer::ProductQuantizer;
use crate::utils::SearchCandidate;
const BLOCK_SIZE: usize = 4096;
const MAX_NEIGHBORS: usize = 128;

#[derive(Serialize, Deserialize, Debug, Archive, Clone)]
pub struct DiskMeta {
    num_pts: u64,
    dimension: u64,
    max_node_size: u64,
    nodes_per_block: u64,
    total_nodes: u64,
    point_size: u64,
    vector_per_block: u64,
    metric_type: MetricType,
    start_node: u32,
}

#[derive(Serialize, Deserialize, Debug, Archive, Clone)]
pub struct DiskNode {
    pub id: u32,
    pub neighbors: Vec<u32>,
}

pub struct IndexFiles {
    data_file: Mutex<File>,
    graph_file: Mutex<File>,
}

// The Disk-Based Index Manager
pub struct DiskIndexStorage {
    pub index_name: String,
    pub index_file: Option<IndexFiles>,
    pub pq: Option<MmappedPQ>,
    pub codes: Option<MmappedCodes>,
    pub meta: Option<DiskMeta>,
}
const ALIGNMENT: usize = 8;
const INDEX_DATA_FILE: &str = "disk_view_data.bin";
const INDEX_GRAPH_FILE: &str = "disk_view_index.bin";
const INDEX_PQ_FILE: &str = "disk_view_pq.bin";
const INDEX_CODES_FILE: &str = "disk_view_codes.bin";

#[inline]
fn align_up(value: usize) -> usize {
    (value + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

fn index_file_paths(index_name: &str) -> (String, String, String, String) {
    let data_file_path = format!("{}/{}", index_name, INDEX_DATA_FILE);
    let graph_file_path = format!("{}/{}", index_name, INDEX_GRAPH_FILE);
    let pq_file_path = format!("{}/{}", index_name, INDEX_PQ_FILE);
    let codes_file_path = format!("{}/{}", index_name, INDEX_CODES_FILE);
    (data_file_path, graph_file_path, pq_file_path, codes_file_path)
}

macro_rules! read_sized_data {
    ($file:expr, $offset:expr, $max_size:expr, $count:expr, $type:ty) => {
        async move {
            let mut block_buffer = vec![0u8; BLOCK_SIZE];

            let mut file_guard = $file.lock().await;
            file_guard.seek(SeekFrom::Start($offset)).await?;
            file_guard.read_exact(&mut block_buffer).await?;
            drop(file_guard);

            let mut data: Vec<$type> = Vec::new();
            for i in 0..$count {
                let start_idx = (i * $max_size) as usize;
                let buffer = &block_buffer[start_idx..(start_idx + $max_size as usize)];
                let size_bytes: [u8; 8] = buffer[0..8].try_into().unwrap();
                let size = u64::from_le_bytes(size_bytes);

                if size == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Empty data slot",
                    ));
                } else {
                    let data_bytes = &buffer[8..8 + size as usize];
                    let item = from_bytes::<$type, rancor::Error>(data_bytes)
                        .map_err(|e| std::io::Error::other(e.to_string()))?;
                    data.push(item);
                }
            }
            Ok(data)
        }
    };
}

impl DiskIndexStorage {
    pub async fn new(index_name: &str) -> Result<Self, std::io::Error> {
        if let Ok(storage) = Self::open(index_name).await {
            Ok(storage)
        } else {
            Ok(DiskIndexStorage {
                index_name: index_name.to_string(),
                index_file: None,
                pq: None,
                codes: None,
                meta: None,
            })
        }
    }

    pub async fn open(index_name: &str) -> Result<Self, std::io::Error> {
        let (data_file_path, graph_file_path, pq_file_path, codes_file_path) =
            index_file_paths(index_name);

        let data_file = File::open(data_file_path).await?;
        let graph_file = File::open(graph_file_path).await?;

        let index_files =
            IndexFiles { data_file: Mutex::new(data_file), graph_file: Mutex::new(graph_file) };

        let mut storage = DiskIndexStorage {
            index_name: index_name.to_string(),
            index_file: Some(index_files),
            pq: None,
            codes: None,
            meta: None,
        };

        let meta = storage.get_disk_meta(&storage.index_file.as_ref().unwrap().graph_file).await?;
        storage.meta = Some(meta);

        storage.load_pq(&pq_file_path, &codes_file_path).await?;

        Ok(storage)
    }

    pub async fn insert(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        let (data_file_path, graph_file_path, _, _) = index_file_paths(&self.index_name);

        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&data_file_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(data_file_path)
            .await?;
        let graph_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(graph_file_path)
            .await?;

        self.index_file = Some(IndexFiles {
            data_file: Mutex::new(data_file),
            graph_file: Mutex::new(graph_file),
        });

        self.write_all(mem_index).await?;
        self.create_pq(mem_index).await?;
        Ok(())
    }

    async fn write_all(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        self.write_graph(mem_index).await?;
        self.write_data(mem_index).await?;
        Ok(())
    }

    async fn write_graph(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        let index_file = self.index_file.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "No index file")
        })?;

        let mut file_guard = index_file.graph_file.lock().await;
        let file: &mut File = &mut file_guard;

        let binding =
            mem_index.points.iter().next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "Empty index")
            })?;
        let first_element = binding.value();

        // Calculate a fixed node size by using a dummy node with MAX_NEIGHBORS (neighbors only)
        let dummy_neighbors = vec![0u32; MAX_NEIGHBORS];
        let dummy_node = DiskNode { id: 0, neighbors: dummy_neighbors };
        let mut arena = Arena::new();
        let dummy_bytes = to_bytes_with_alloc::<_, rancor::Error>(&dummy_node, arena.acquire())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let node_size = dummy_bytes.len();
        let aligned_node_size = align_up(node_size + 8); // +8 for the length prefix

        let nodes_per_block = BLOCK_SIZE / aligned_node_size;

        let mut arena = Arena::new();
        let vector_bytes =
            to_bytes_with_alloc::<_, rancor::Error>(&first_element.vector, arena.acquire())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        let vector_serialized_size = vector_bytes.len();
        let aligned_vector_size = align_up(vector_serialized_size + 8); // +8 for length prefix
        let vector_per_block = BLOCK_SIZE / aligned_vector_size;

        let disk_meta = DiskMeta {
            num_pts: mem_index.points.len() as u64,
            dimension: first_element.vector.dim() as u64,
            max_node_size: aligned_node_size as u64,
            nodes_per_block: nodes_per_block as u64,
            total_nodes: mem_index.points.len() as u64,
            metric_type: mem_index.metric,
            point_size: aligned_vector_size as u64,
            vector_per_block: vector_per_block as u64,
            start_node: mem_index.start_node.unwrap_or(0),
        };
        self.meta = Some(disk_meta.clone());

        println!("Index metadata: {:?}", disk_meta);
        let mut arena = Arena::new();
        let meta_bytes = to_bytes_with_alloc::<_, rancor::Error>(&disk_meta, arena.acquire())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        file.write_all(&meta_bytes).await?;

        let padding_len = BLOCK_SIZE.saturating_sub(meta_bytes.len());
        if padding_len > 0 {
            file.write_all(&vec![0u8; padding_len]).await?;
        }

        file.seek(SeekFrom::Start(BLOCK_SIZE as u64)).await?;

        let mut points_vec: Vec<_> = mem_index.points.iter().collect();
        points_vec.sort_by_key(|p| *p.key());
        let mut graph_block = Vec::with_capacity(BLOCK_SIZE);
        let mut count = 0;
        let mut block_idx = 1;
        let mut node_buffer = Vec::with_capacity(aligned_node_size);
        for r in points_vec {
            node_buffer.clear();
            let point = r.value();
            let mut neighbors =
                mem_index.graph.get(&point.id).map(|v| v.clone()).unwrap_or_default();

            if neighbors.len() < MAX_NEIGHBORS {
                neighbors.resize(MAX_NEIGHBORS, 0);
            } else if neighbors.len() > MAX_NEIGHBORS {
                neighbors.truncate(MAX_NEIGHBORS);
            }

            let disk_node = DiskNode { id: point.id, neighbors };

            let mut arena = Arena::new();
            let bytes = to_bytes_with_alloc::<_, rancor::Error>(&disk_node, arena.acquire())
                .map_err(|e| std::io::Error::other(e.to_string()))?;

            node_buffer.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            node_buffer.extend_from_slice(&bytes);

            if node_buffer.len() < aligned_node_size {
                node_buffer.resize(aligned_node_size, 0);
            }
            graph_block.extend_from_slice(&node_buffer);
            count += 1;

            if count == nodes_per_block {
                graph_block.resize(BLOCK_SIZE, 0);
                let target_offset = BLOCK_SIZE * block_idx;
                file.seek(SeekFrom::Start(target_offset as u64)).await?;
                file.write_all(&graph_block).await?;
                count = 0;
                block_idx += 1;
                graph_block.clear();
            }
        }

        if !graph_block.is_empty() {
            graph_block.resize(BLOCK_SIZE, 0);
            let target_offset = BLOCK_SIZE * block_idx;
            file.seek(SeekFrom::Start(target_offset as u64)).await?;
            file.write_all(&graph_block).await?;
        }

        file.flush().await?;
        Ok(())
    }

    async fn write_data(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        let index_file = self.index_file.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "No index file")
        })?;
        let meta = self
            .meta
            .as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "No metadata"))?;

        let mut file_guard = index_file.data_file.lock().await;
        let file: &mut File = &mut file_guard;

        let aligned_vector_size = meta.point_size as usize;
        let vectors_per_block = meta.vector_per_block;
        let mut points_vec: Vec<_> = mem_index.points.iter().collect();
        points_vec.sort_by_key(|p| *p.key());
        let mut vectors_block = Vec::with_capacity(BLOCK_SIZE);
        let mut count = 0;
        let mut block_idx = 0;

        for r in points_vec {
            let point = r.value();

            let mut arena = Arena::new();
            let bytes = to_bytes_with_alloc::<_, rancor::Error>(&point.vector, arena.acquire())
                .map_err(|e| std::io::Error::other(e.to_string()))?;

            let mut buffer = Vec::with_capacity(aligned_vector_size);
            buffer.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            buffer.extend_from_slice(&bytes);

            if buffer.len() < aligned_vector_size {
                buffer.resize(aligned_vector_size, 0);
            }

            vectors_block.extend_from_slice(&buffer);
            count += 1;

            if count == vectors_per_block as usize {
                vectors_block.resize(BLOCK_SIZE, 0);
                let target_offset = BLOCK_SIZE * block_idx;
                file.seek(SeekFrom::Start(target_offset as u64)).await?;
                file.write_all(&vectors_block).await?;
                count = 0;
                block_idx += 1;
                vectors_block.clear();
            }
        }

        if !vectors_block.is_empty() {
            vectors_block.resize(BLOCK_SIZE, 0);
            let target_offset = BLOCK_SIZE * block_idx;
            file.seek(SeekFrom::Start(target_offset as u64)).await?;
            file.write_all(&vectors_block).await?;
        }

        file.flush().await?;
        Ok(())
    }

    pub async fn read_node(
        &self, id: u32,
    ) -> Result<(Vec<DiskNode>, Vec<VectorPoint>), std::io::Error> {
        let index_file = self.index_file.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "No index file")
        })?;
        let meta = self
            .meta
            .as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "No metadata"))?;

        let graph_block_id = id as u64 / meta.nodes_per_block;
        let vector_block_id = id as u64 / meta.vector_per_block;

        let graph_offset = BLOCK_SIZE as u64 * (graph_block_id + 1);
        let vector_offset = BLOCK_SIZE as u64 * vector_block_id;
        let possible_graph_count =
            (meta.total_nodes - meta.nodes_per_block * graph_block_id).min(meta.nodes_per_block);
        let possible_vector_count =
            (meta.num_pts - meta.vector_per_block * vector_block_id).min(meta.vector_per_block);

        let (nodes, vectors) = tokio::try_join!(
            read_sized_data!(
                &index_file.graph_file,
                graph_offset,
                meta.max_node_size,
                possible_graph_count,
                DiskNode
            ),
            read_sized_data!(
                &index_file.data_file,
                vector_offset,
                meta.point_size,
                possible_vector_count,
                VectorData
            )
        )?;

        //TODO: Fetch additional vector for each fetched node
        let vector_start_id = (vector_block_id * meta.vector_per_block) as u32;
        let vectors = vectors
            .into_iter()
            .enumerate()
            .map(|(i, v)| VectorPoint::new(vector_start_id + i as u32, v))
            .collect();

        Ok((nodes, vectors))
    }

    #[inline(always)]
    async fn get_disk_meta(&self, file: &Mutex<File>) -> Result<DiskMeta, std::io::Error> {
        let mut meta_bytes = [0u8; BLOCK_SIZE];

        let mut f = file.lock().await;
        f.seek(SeekFrom::Start(0)).await?;
        f.read_exact(&mut meta_bytes).await?;
        drop(f);

        let size = std::mem::size_of::<DiskMeta>();
        let meta = from_bytes::<DiskMeta, rancor::Error>(&meta_bytes[0..size])
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(meta)
    }

    pub async fn search(
        &self, query: &VectorData, l: usize, visited_map: &mut HashMap<u32, VectorPoint>,
    ) {
        if self.index_file.is_none() || self.meta.is_none() {
            return;
        };
        let meta = self.meta.as_ref().unwrap();
        let start_node_id = meta.start_node;

        self.greedy_search_ssd(visited_map, start_node_id, query, l, meta.metric_type).await;
    }

    pub async fn greedy_search_ssd(
        &self, visited_map: &mut HashMap<u32, VectorPoint>, start_node_id: u32, query: &VectorData,
        l: usize, metric_type: MetricType,
    ) {
        let mut candidates = BinaryHeap::new();
        // Cache for loaded nodes in this search to avoid re-reading same node
        let mut node_cache: HashMap<u32, (DiskNode, VectorData)> = HashMap::new();

        let Ok(start_data) = self.read_node(start_node_id).await else {
            return;
        };
        add_to_cache(&mut node_cache, start_data);

        let start_data = node_cache.get(&start_node_id).unwrap();
        candidates.push(SearchCandidate {
            point_id: start_node_id,
            distance: start_data.1.distance(query, metric_type),
        });

        let lut = self.pq.as_ref().and_then(|pq| pq.compute_lut(query));

        while let Some(current) = candidates.pop() {
            if visited_map.contains_key(&current.point_id) {
                continue;
            }

            // If we don't have the vector in cache, we need to read it now (lazy loading)
            let (node, vector) = if let Some(cached) = node_cache.get(&current.point_id) {
                cached
            } else if let Ok(disk_data) = self.read_node(current.point_id).await {
                add_to_cache(&mut node_cache, disk_data);

                node_cache.get(&current.point_id).unwrap()
            } else {
                continue;
            };

            let point = VectorPoint::new(node.id, vector.clone());
            visited_map.insert(current.point_id, point);

            if visited_map.len() >= l {
                break;
            }

            let neighbors = node.neighbors.clone();
            for neighbor_id in neighbors {
                if visited_map.contains_key(&neighbor_id) {
                    continue;
                }

                let mut distance_estimated = None;
                if let Some(codes) = &self.codes
                    && let Some(lut) = &lut
                    && let Some(codes) = codes.get(neighbor_id)
                {
                    distance_estimated = Some(pq_loader::estimate_distance(lut, codes));
                }

                let distance = if let Some(dist) = distance_estimated {
                    dist
                } else if let Some((_, vector)) = node_cache.get(&neighbor_id) {
                    vector.distance(query, metric_type)
                } else if let Ok(neighbor_data) = self.read_node(neighbor_id).await {
                    add_to_cache(&mut node_cache, neighbor_data);
                    let vector = &node_cache.get(&neighbor_id).unwrap().1;
                    vector.distance(query, metric_type)
                } else {
                    continue;
                };

                if candidates.len() < l {
                    candidates.push(SearchCandidate { point_id: neighbor_id, distance });
                } else if let Some(mut worst) = candidates.peek_mut()
                    && distance < worst.distance
                {
                    *worst = SearchCandidate { point_id: neighbor_id, distance };
                }
            }
        }
    }

    async fn create_pq(&mut self, mem_index: &InMemIndex) -> Result<(), io::Error> {
        if mem_index.points.is_empty() {
            return Ok(());
        }
        let points_vec: Vec<_> = mem_index.points.iter().collect();
        let training_data = points_vec.iter().map(|p| p.vector.clone()).collect::<Vec<_>>();
        let dim = training_data[0].dim();
        let num_subspaces = if dim >= 32 { 32 } else { dim };
        let pq = ProductQuantizer::train(&training_data, num_subspaces, 256, 10);
        let mut pq_codes = HashMap::new();
        for r in points_vec {
            let point = r.value();
            let codes = pq.encode_batch(std::slice::from_ref(&point.vector));
            pq_codes.insert(point.id, codes);
        }
        self.write_pq(pq, pq_codes).await
    }

    async fn write_pq(
        &self, pq: ProductQuantizer, pq_codes: HashMap<u32, Vec<u8>>,
    ) -> io::Result<()> {
        let (_, _, pq_file_path, codes_file_path) = index_file_paths(&self.index_name);

        let mut arena = Arena::new();
        let pq_bytes = to_bytes_with_alloc::<_, rancor::Error>(&pq, arena.acquire())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        tokio::fs::write(pq_file_path, &pq_bytes).await?;

        if !pq_codes.is_empty() {
            let mut arena = Arena::new();
            let codes_bytes = to_bytes_with_alloc::<_, rancor::Error>(&pq_codes, arena.acquire())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            tokio::fs::write(codes_file_path, &codes_bytes).await?;
        }

        Ok(())
    }

    async fn load_pq(&mut self, pq_file_path: &str, codes_file_path: &str) -> io::Result<()> {
        let (pq, codes) = tokio::try_join!(
            pq_loader::load_pq_data(pq_file_path),
            pq_loader::load_codes(codes_file_path)
        )?;
        self.pq = pq;
        self.codes = codes;
        Ok(())
    }
}

fn add_to_cache(
    node_cache: &mut HashMap<u32, (DiskNode, VectorData)>,
    disk_data: (Vec<DiskNode>, Vec<VectorPoint>),
) {
    let mut vector_map: HashMap<u32, VectorData> =
        disk_data.1.into_iter().map(|p| (p.id, p.vector)).collect();
    for node in disk_data.0.into_iter() {
        let id = node.id;
        if let Some(vector) = vector_map.remove(&id) {
            node_cache.insert(id, (node, vector));
        }
    }
}

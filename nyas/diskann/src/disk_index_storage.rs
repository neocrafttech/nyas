use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;

use rkyv::api::high::to_bytes_with_alloc;
use rkyv::ser::allocator::Arena;
use rkyv::{Archive, Deserialize, Serialize, from_bytes, rancor};
use system::metric::MetricType;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;
use tokio::fs::{File, OpenOptions};
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Mutex;
use tracing::debug;

use crate::in_mem_index::InMemIndex;
use crate::utils::SearchCandidate;
const BLOCK_SIZE: usize = 4096;
const MAX_NEIGHBORS: usize = 64;

#[derive(Serialize, Deserialize, Debug, Archive)]
pub struct DiskMeta {
    num_pts: u64,
    dimension: u64,
    max_node_size: u64,
    nodes_per_block: u64,
    total_nodes: u64,
    point_size: u64,
    metric_type: MetricType,
}

#[derive(Serialize, Deserialize, Debug, Archive)]
pub struct DiskNode {
    // The vector data
    id: u64,
    // The IDs of neighbors
    neighbors: Vec<u64>,
}

pub struct IndexFiles {
    data_file: Mutex<File>,
    graph_file: Mutex<File>,
}

// The Disk-Based Index Manager
pub struct DiskIndexStorage {
    files: Vec<IndexFiles>,
}
const ALIGNMENT: usize = 8;
#[inline]
fn align_up(value: usize) -> usize {
    (value + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

impl DiskIndexStorage {
    pub fn new() -> Result<Self, std::io::Error> {
        Ok(DiskIndexStorage { files: vec![] })
    }

    pub async fn insert(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        let file_count = self.files.len();
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!("disk_view_data_{}.bin", file_count))
            .await?;
        let graph_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!("disk_view_index_{}.bin", file_count))
            .await?;
        self.files.push(IndexFiles {
            data_file: Mutex::new(data_file),
            graph_file: Mutex::new(graph_file),
        });

        self.write_graph(mem_index).await?;
        self.write_data(mem_index).await?;
        Ok(())
    }

    async fn write_graph(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        let last_file =
            self.files.last().ok_or_else(|| std::io::Error::other("No files in the list"))?;

        let mut file_guard = last_file.graph_file.lock().await;
        let file: &mut File = &mut file_guard;

        let binding =
            mem_index.points.iter().next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "Empty index")
            })?;
        let first_element = binding.value();
        let max_node_size = std::mem::size_of::<u64>() + MAX_NEIGHBORS * std::mem::size_of::<u64>();
        let aligned_max_node_size = align_up(max_node_size + 8);
        let nodes_per_block = BLOCK_SIZE / max_node_size;
        println!("Number of nodes per block: {} {}", nodes_per_block, first_element.size_bytes());
        println!("Number of points: {}", mem_index.points.len());
        let disk_meta = DiskMeta {
            num_pts: mem_index.points.len() as u64,
            dimension: first_element.vector.dim() as u64,
            max_node_size: aligned_max_node_size as u64,
            nodes_per_block: nodes_per_block as u64,
            total_nodes: mem_index.points.len() as u64,
            metric_type: mem_index.metric,
            point_size: first_element.size_bytes() as u64,
        };
        println!("Index metadata: {:?}", disk_meta);
        let mut arena = Arena::new();
        let meta_bytes = to_bytes_with_alloc::<_, rancor::Error>(&disk_meta, arena.acquire())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        file.write_all(&meta_bytes).await?;

        let padding_len = BLOCK_SIZE.saturating_sub(meta_bytes.len());
        if padding_len > 0 {
            file.write_all(&vec![0u8; padding_len]).await?;
        }
        let mut block_buffer = Vec::with_capacity(BLOCK_SIZE);
        let mut nodes_in_current_block = 0;
        for r in mem_index.points.iter() {
            let point = r.value();

            let neighbors = mem_index.graph.get(&point.id).map(|v| v.clone()).unwrap_or_default();
            println!("{}:{:?}", point.id, neighbors);
            let disk_node = DiskNode { id: point.id, neighbors };

            let mut arena = Arena::new();
            let bytes =
                to_bytes_with_alloc::<_, rancor::Error>(&disk_node, arena.acquire()).unwrap();

            block_buffer.extend_from_slice(&bytes.len().to_le_bytes());
            block_buffer.extend_from_slice(&bytes);
            let current_size = std::mem::size_of::<u64>() + bytes.len();
            let aligned_size = align_up(current_size);
            let padding = aligned_size - current_size;
            if padding > 0 {
                block_buffer.extend_from_slice(&vec![0u8; padding]);
            }

            nodes_in_current_block += 1;

            if nodes_in_current_block == nodes_per_block
                || block_buffer.len() + max_node_size > BLOCK_SIZE
            {
                if block_buffer.len() < BLOCK_SIZE {
                    block_buffer.resize(BLOCK_SIZE, 0);
                }

                file.write_all(&block_buffer).await?;

                block_buffer.clear();
                nodes_in_current_block = 0;
            }
        }
        if !block_buffer.is_empty() {
            if block_buffer.len() < BLOCK_SIZE {
                block_buffer.resize(BLOCK_SIZE, 0);
            }
            file.write_all(&block_buffer).await?;
        }
        Ok(())
    }

    async fn write_data(&mut self, mem_index: &InMemIndex) -> io::Result<()> {
        let mut file_guard = self.files.last().unwrap().data_file.lock().await;
        let file: &mut File = &mut file_guard;
        if mem_index.points.is_empty() {
            return Err(std::io::Error::other("Empty index"));
        }
        let binding = mem_index.points.iter().next().unwrap();
        let first_element = binding.value();
        let max_node_size = first_element.size_bytes();
        let nodes_per_block = BLOCK_SIZE / max_node_size;
        println!("Number of nodes per block: {}", nodes_per_block);

        let mut block_buffer = Vec::with_capacity(BLOCK_SIZE);
        let mut nodes_in_current_block = 0;
        for r in mem_index.points.iter() {
            let point = r.value();

            let mut arena = Arena::new();
            let bytes = to_bytes_with_alloc::<_, rancor::Error>(point, arena.acquire()).unwrap();

            block_buffer.extend_from_slice(&bytes);
            let current_size = bytes.len();
            let aligned_size = align_up(current_size);
            let padding = aligned_size - current_size;
            if padding > 0 {
                block_buffer.extend_from_slice(&vec![0u8; padding]);
            }

            nodes_in_current_block += 1;

            if nodes_in_current_block == nodes_per_block
                || block_buffer.len() + max_node_size > BLOCK_SIZE
            {
                if block_buffer.len() < BLOCK_SIZE {
                    block_buffer.resize(BLOCK_SIZE, 0);
                }

                file.write_all(&block_buffer).await?;

                block_buffer.clear();
                nodes_in_current_block = 0;
            }
        }
        if !block_buffer.is_empty() {
            if block_buffer.len() < BLOCK_SIZE {
                block_buffer.resize(BLOCK_SIZE, 0);
            }
            file.write_all(&block_buffer).await?;
        }
        Ok(())
    }

    pub async fn get_nodes(
        &self, file: &Mutex<File>, block_offset: u64, total_nodes: u64,
    ) -> Result<Vec<DiskNode>, std::io::Error> {
        let mut nodes = Vec::with_capacity(total_nodes as usize);
        let mut block_offset = block_offset;
        while nodes.len() < total_nodes as usize {
            let disk_bytes = Self::read_bytes(file, block_offset).await?;
            let mut offset = 0;
            while offset + 8 < BLOCK_SIZE && nodes.len() < total_nodes as usize {
                if offset % ALIGNMENT != 0 {
                    offset = align_up(offset);
                }
                let size_bytes: [u8; 8] = disk_bytes[offset..offset + 8].try_into().unwrap();
                let size = u64::from_le_bytes(size_bytes);
                if size == 0 {
                    break;
                }
                if offset + 8 + size as usize > BLOCK_SIZE {
                    break;
                }

                let bytes = &disk_bytes[offset + 8..offset + 8 + size as usize];
                let node: DiskNode = from_bytes::<DiskNode, rancor::Error>(bytes)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;

                nodes.push(node);
                offset += 8 + size as usize;
            }
            block_offset += 1;
        }
        Ok(nodes)
    }

    pub async fn get_points(
        &self, file: &Mutex<File>, block_offset: u64, nodes_per_block: u64, point_size: u64,
        total_nodes: u64,
    ) -> io::Result<Vec<VectorPoint>> {
        let mut block_offset = block_offset;
        let mut points = Vec::with_capacity(nodes_per_block as usize);
        while points.len() < total_nodes as usize {
            let disk_bytes = Self::read_bytes(file, block_offset).await?;
            let mut offset = 0;
            while offset < BLOCK_SIZE && points.len() < total_nodes as usize {
                if offset % ALIGNMENT != 0 {
                    offset = align_up(offset);
                }
                let bytes = &disk_bytes[offset..offset + point_size as usize];
                let point: VectorPoint = from_bytes::<VectorPoint, rancor::Error>(bytes)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;

                points.push(point);
                offset += point_size as usize;
            }
            block_offset += 1;
        }
        Ok(points)
    }

    #[inline(always)]
    async fn get_disk_meta(&self, file: &Mutex<File>) -> Result<DiskMeta, std::io::Error> {
        let meta_bytes = Self::read_bytes(file, 0).await?;

        let size = std::mem::size_of::<DiskMeta>();
        let meta = from_bytes::<DiskMeta, rancor::Error>(&meta_bytes[0..size])
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(meta)
    }

    pub async fn search(
        &self, query: &VectorData, l: usize, visited_map: &mut HashMap<u64, VectorPoint>,
    ) {
        for file in self.files.iter() {
            let disk_meta = self.get_disk_meta(&file.graph_file).await;
            let Ok(disk_meta) = disk_meta else {
                debug!("Disk metadata not found");
                continue;
            };
            debug!("Disk Meta found: {:?}", disk_meta);

            let Ok(nodes) = self.get_nodes(&file.graph_file, 1, disk_meta.total_nodes).await else {
                debug!("Nodes not found");
                continue;
            };
            debug!("Nodes {:?}", nodes.iter().map(|n| n.id).collect::<Vec<_>>());

            let Ok(points) = self
                .get_points(
                    &file.data_file,
                    0,
                    disk_meta.nodes_per_block,
                    disk_meta.point_size * 2,
                    disk_meta.total_nodes,
                )
                .await
            else {
                debug!("Points not found in file");
                continue;
            };
            debug!("Points {:?}", points.iter().map(|p| p.id).collect::<Vec<_>>());
            Self::greedy_search(visited_map, &nodes, &points, query, l, disk_meta.metric_type);
        }
    }

    pub fn greedy_search(
        visited_map: &mut HashMap<u64, VectorPoint>, nodes: &[DiskNode], points: &[VectorPoint],
        query: &VectorData, l: usize, metric_type: MetricType,
    ) {
        let mut candidates = BinaryHeap::new();

        let Some(start_point) = points.first() else {
            return;
        };
        candidates.push(SearchCandidate {
            point_id: start_point.id,
            distance: start_point.distance_to_vector(query, metric_type),
        });

        let graph = nodes
            .iter()
            .map(|node| (node.id, node.neighbors.clone()))
            .collect::<HashMap<u64, Vec<u64>>>();

        let points_map = points.iter().map(|p| (p.id, p)).collect::<HashMap<u64, &VectorPoint>>();

        while let Some(current) = candidates.pop()
            && !visited_map.contains_key(&current.point_id)
        {
            visited_map.insert(current.point_id, points_map[&current.point_id].clone());

            if let Some(neighbors) = graph.get(&current.point_id) {
                for neighbor_id in neighbors {
                    if !visited_map.contains_key(neighbor_id) {
                        Self::push_candidate(
                            &points_map,
                            *neighbor_id,
                            query,
                            l,
                            &mut candidates,
                            metric_type,
                        );
                    }
                }
            }

            if visited_map.len() >= l {
                break;
            }
        }
    }

    fn push_candidate(
        points: &HashMap<u64, &VectorPoint>, neighbor_id: u64, query: &VectorData, l: usize,
        candidates: &mut BinaryHeap<SearchCandidate<u64>>, metric: MetricType,
    ) {
        if let Some(neighbor_point) = points.get(&neighbor_id) {
            if candidates.len() < l {
                candidates.push(SearchCandidate {
                    point_id: neighbor_id,
                    distance: neighbor_point.distance_to_vector(query, metric),
                });
            } else if let Some(mut worst) = candidates.peek_mut() {
                let distance = neighbor_point.distance_to_vector(query, metric);
                if distance < worst.distance {
                    *worst = SearchCandidate { point_id: neighbor_id, distance };
                }
            }
        }
    }

    async fn read_bytes(
        file: &Mutex<File>, block_offset: u64,
    ) -> std::io::Result<[u8; BLOCK_SIZE]> {
        let mut bytes = [0u8; BLOCK_SIZE];
        let offset = block_offset * BLOCK_SIZE as u64;

        let mut f = file.lock().await;
        f.seek(SeekFrom::Start(offset)).await?;
        f.read_exact(&mut bytes).await?;

        Ok(bytes)
    }
}

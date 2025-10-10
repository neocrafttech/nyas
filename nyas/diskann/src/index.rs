use lru::LruCache;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use std::num::NonZero;
use tokio::fs::File;
use tokio::sync::Mutex;
use tokio::try_join;

type NodeId = u32;

#[derive(Debug)]
pub struct DiskANNIndex {
    /// Memory-mapped vector storage
    vec_file: Mmap,

    /// Memory-mapped adjacency lists
    adj_file: Mmap,

    /// Cache for hot nodes: NodeId -> Vector + neighbors
    hot_cache: Mutex<LruCache<NodeId, CachedNode>>,

    /// Entry points for search
    entry_points: Vec<NodeId>,

    /// Number of neighbors per node
    max_neighbors: usize,

    /// Dimension of vectors
    dim: usize,
}

#[derive(Debug, Clone)]
pub struct CachedNode {
    pub id: NodeId,
    pub vector: Vec<f32>,
    pub neighbors: Vec<NodeId>,
}

impl DiskANNIndex {
    pub async fn load(
        vec_path: &str,
        adj_path: &str,
        dim: usize,
        max_neighbors: usize,
    ) -> Result<Self, std::io::Error> {
        let (vec_file, adj_file) = try_join!(File::open(vec_path), File::open(adj_path))?;
        let vec_mmap = unsafe { Mmap::map(&vec_file)? };
        let adj_mmap = unsafe { Mmap::map(&adj_file)? };

        Ok(Self {
            vec_file: vec_mmap,
            adj_file: adj_mmap,
            hot_cache: Mutex::new(LruCache::new(NonZero::new(1024).unwrap())),
            entry_points: vec![0],
            max_neighbors,
            dim,
        })
    }

    async fn load_node(&self, node_id: NodeId) -> CachedNode {
        if let Some(node) = self.hot_cache.lock().await.get(&node_id) {
            return node.clone();
        }

        let start = (node_id as usize) * self.dim * 4;
        let vec_bytes = &self.vec_file[start..start + self.dim * 4];
        let vector = vec_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let neighbors_start = (node_id as usize) * self.max_neighbors * 4;
        let neighbors_bytes =
            &self.adj_file[neighbors_start..neighbors_start + self.max_neighbors * 4];
        let neighbors: Vec<NodeId> = neighbors_bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let cached = CachedNode {
            id: node_id,
            vector,
            neighbors,
        };
        self.hot_cache.lock().await.put(node_id, cached.clone());
        cached
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec() // avoid division by zero
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    pub fn distance(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        let a_norm = Self::normalize(a);
        let b_norm = Self::normalize(b);

        a_norm
            .iter()
            .zip(b_norm.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
    ) -> Vec<(NodeId, OrderedFloat<f32>)> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashSet};

        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, NodeId)>> = BinaryHeap::new();
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut top_k: BinaryHeap<Reverse<(OrderedFloat<f32>, NodeId)>> = BinaryHeap::new();

        for &ep in &self.entry_points {
            let node = self.load_node(ep).await;
            let dist = Self::distance(query, &node.vector);
            candidates.push(Reverse((OrderedFloat(dist), ep)));
            visited.insert(ep);
        }

        while let Some(Reverse((dist, node_id))) = candidates.pop() {
            let node = self.load_node(node_id).await;

            if top_k.len() < k {
                top_k.push(Reverse((dist, node_id)));
            } else if dist < top_k.peek().unwrap().0.0 {
                top_k.pop();
                top_k.push(Reverse((dist, node_id)));
            }

            for &nbr in &node.neighbors {
                if !visited.contains(&nbr) {
                    visited.insert(nbr);
                    let nbr_node = self.load_node(nbr).await;
                    let nbr_dist = Self::distance(query, &nbr_node.vector);
                    candidates.push(Reverse((OrderedFloat(nbr_dist), nbr)));
                    if candidates.len() > beam_width {
                        candidates.pop();
                    }
                }
            }
        }

        let mut result: Vec<_> = top_k.into_iter().map(|Reverse((d, id))| (id, d)).collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result
    }
}

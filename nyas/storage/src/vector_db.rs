use diskann::index_builder::build_index;
use hnsw_rs::prelude::*;
use std::collections::HashMap;

pub struct VectorDB {
    pub storage: HashMap<String, Vec<f32>>,
    pub hnsw: Hnsw<'static, f32, DistL2>,
    id_to_idx: HashMap<String, usize>,
    idx_to_id: HashMap<usize, String>,
    next_idx: usize,
}

impl VectorDB {
    pub fn new(dim: usize) -> Self {
        Self {
            storage: HashMap::new(),
            hnsw: Hnsw::<f32, DistL2>::new(
                16,     // max_nb_connection
                200,    // max_layer
                16,     // ef_construction
                dim,    // dimension
                DistL2, // distance function
            ),
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            next_idx: 0,
        }
    }

    pub fn insert(&mut self, id: String, vector: Vec<f32>) {
        let idx = self.next_idx;
        self.next_idx += 1;

        self.id_to_idx.insert(id.clone(), idx);
        self.idx_to_id.insert(idx, id.clone());

        self.hnsw.insert((&vector, idx));
        self.storage.insert(id, vector);
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let results = self.hnsw.search(query, k, 50); // ef_search = 50

        results
            .into_iter()
            .filter_map(|neighbor| {
                self.idx_to_id
                    .get(&neighbor.d_id)
                    .map(|id| (id.clone(), neighbor.distance))
            })
            .collect()
    }

    pub async fn build_index(&mut self) -> std::io::Result<()> {
        let vec_refs: Vec<&Vec<f32>> = self.storage.values().collect();

        build_index(&vec_refs, 256, "vectors.bin", "graph.bin").await
    }
}

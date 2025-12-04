use std::collections::HashMap;
use std::vec;

pub struct VectorDB {
    pub storage: HashMap<String, Vec<f32>>,
    id_to_idx: HashMap<String, usize>,
    idx_to_id: HashMap<usize, String>,
    next_idx: usize,
}

impl VectorDB {
    pub fn new(_dim: usize) -> Self {
        Self {
            storage: HashMap::new(),
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

        self.storage.insert(id, vector);
    }

    pub fn search(&self, _query: &[f32], _k: usize) -> Vec<(String, f32)> {
        vec![]
    }

    pub async fn build_index(&mut self) -> std::io::Result<()> {
        //let vec_refs: Vec<&Vec<f32>> = self.storage.values().collect();

        Ok(())
        //build_index(&vec_refs, 256, "vectors.bin", "graph.bin").await
    }
}

use std::io::Error;

use system::metric::MetricType;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;

use crate::in_disk_index::InDiskIndex;

pub struct IndexView {
    in_disk_index: InDiskIndex,
}

impl IndexView {
    pub fn new() -> Result<Self, Error> {
        let in_disk_index = InDiskIndex::new(32, 1.2, 50, 20, MetricType::L2)?;
        Ok(IndexView { in_disk_index })
    }

    pub async fn insert(&self, point: &VectorPoint) -> Result<(), String> {
        self.in_disk_index.insert(point).await?;
        Ok(())
    }

    pub fn delete(&self, id: u64) {
        self.in_disk_index.delete(id);
    }

    pub async fn search(&self, query: &VectorData, k: usize, l: usize) -> Vec<u64> {
        self.in_disk_index.search(query, k, l).await
    }
}

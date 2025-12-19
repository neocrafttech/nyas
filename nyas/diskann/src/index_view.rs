use std::io::Error;

use system::metric::MetricType;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;

use crate::in_disk_index::InDiskIndex;

pub struct IndexView {
    in_disk_index: InDiskIndex,
}

impl IndexView {
    pub async fn new(index_name: &str) -> Result<Self, Error> {
        let in_disk_index =
            InDiskIndex::new(index_name, 32, 1.2, 128, 9999, MetricType::L2).await?;
        Ok(IndexView { in_disk_index })
    }

    pub async fn insert(&self, point: &VectorPoint) -> Result<(), String> {
        self.in_disk_index.insert(point).await?;
        Ok(())
    }

    pub fn delete(&self, id: u32) {
        self.in_disk_index.delete(id);
    }

    pub async fn search(&self, query: &VectorData, k: usize, l: usize) -> Vec<u32> {
        //TODO: Implement streaming merge as background task
        let result = self.in_disk_index.streaming_merge().await;
        println!("Streaming merge result: {:?}", result);
        self.in_disk_index.search(query, k, l).await
    }
}

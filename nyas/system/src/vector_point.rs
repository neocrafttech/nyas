use rkyv::{Archive, Deserialize, Serialize};

use crate::metric::{Distance, MetricType};
use crate::vector_data::VectorData;

#[derive(Debug, Clone, Serialize, Deserialize, Archive)]
pub struct VectorPoint {
    pub id: u32,
    pub vector: VectorData,
}

impl VectorPoint {
    pub fn new(id: u32, vector: VectorData) -> Self {
        VectorPoint { id, vector }
    }

    pub fn size_bytes(&self) -> usize {
        std::mem::size_of::<u32>() + self.vector.size_bytes()
    }

    pub fn distance(&self, other: &VectorPoint, metric: MetricType) -> f64 {
        self.vector.distance(&other.vector, metric)
    }

    pub fn distance_to_vector(&self, vector: &VectorData, metric: MetricType) -> f64 {
        self.vector.distance(vector, metric)
    }
}

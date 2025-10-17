use std::hash::Hash;

use std::fmt::Debug;

use crate::metric::MetricType;
use crate::{metric::Distance, vector_data::VectorData};
use std::sync::Arc;

#[derive(Clone)]
pub struct VectorPoint<ID>
where
    ID: Debug + Clone + Hash + Eq + PartialEq + Send + Sync + 'static,
{
    pub id: Arc<ID>,
    pub vector: VectorData,
}

impl<ID> VectorPoint<ID>
where
    ID: Debug + Clone + Hash + Eq + PartialEq + Send + Sync + 'static,
{
    pub fn new(id: ID, vector: VectorData) -> Self {
        VectorPoint { id: Arc::new(id), vector }
    }

    pub fn distance(&self, other: &VectorPoint<ID>, metric: MetricType) -> f64 {
        self.vector.distance(&other.vector, metric)
    }

    pub fn distance_to_vector(&self, vector: &VectorData, metric: MetricType) -> f64 {
        self.vector.distance(vector, metric)
    }
}

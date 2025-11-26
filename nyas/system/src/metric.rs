use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Archive)]
pub enum MetricType {
    L2,
    L2Squared,
    Cosine,
    Dot,
}

pub trait Distance {
    fn distance(&self, other: &Self, metric: MetricType) -> f64;
}

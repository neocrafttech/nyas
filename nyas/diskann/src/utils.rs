use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

/// Priority queue element for search
#[derive(Clone)]
pub struct SearchCandidate<ID>
where
    ID: Debug + Clone + Hash + PartialEq + Eq,
{
    pub point_id: ID,
    pub distance: f64,
}

impl<ID> PartialEq for SearchCandidate<ID>
where
    ID: Debug + Clone + Hash + PartialEq + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<ID> Eq for SearchCandidate<ID> where ID: Debug + Clone + Hash + PartialEq + Eq {}

impl<ID> PartialOrd for SearchCandidate<ID>
where
    ID: Debug + Clone + Hash + PartialEq + Eq,
{
    #[allow(clippy::non_canonical_partial_ord_impl)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl<ID> Ord for SearchCandidate<ID>
where
    ID: Debug + Clone + Hash + PartialEq + Eq,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

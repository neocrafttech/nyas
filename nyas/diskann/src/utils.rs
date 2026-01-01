use std::cmp::Ordering;

/// Priority queue element for search
#[derive(Clone)]
pub struct SearchCandidate {
    pub point_id: u32,
    pub distance: f64,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.point_id == other.point_id
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (smaller distance = higher priority)
        match other.distance.partial_cmp(&self.distance) {
            Some(Ordering::Equal) => self.point_id.cmp(&other.point_id),
            Some(ord) => ord,
            None => match (self.distance.is_nan(), other.distance.is_nan()) {
                (false, true) => Ordering::Less,
                (true, false) => Ordering::Greater,
                _ => self.point_id.cmp(&other.point_id),
            },
        }
    }
}

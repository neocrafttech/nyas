use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use rayon::prelude::*;
use system::metric::MetricType;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;

use crate::utils::SearchCandidate;

pub struct InMemIndex {
    /// Graph adjacency lists (out-neighbors)
    pub graph: DashMap<u32, Vec<u32>>,
    /// Points stored in the index
    pub points: DashMap<u32, VectorPoint>,
    /// Start node for navigation
    pub start_node: Option<u32>,
    /// Maximum out-degree
    pub r: usize,
    /// Alpha parameter for RNG property
    pub alpha: f64,
    /// Search list size for building
    pub l_build: usize,
    /// Thread-safe locks for concurrent access
    pub locks: DashMap<u32, Arc<Mutex<()>>>,
    /// Metric type for distance calculation
    pub metric: MetricType,
}

impl InMemIndex {
    pub fn new(r: usize, alpha: f64, l_build: usize, metric: MetricType) -> Self {
        assert!(alpha > 1.0);
        InMemIndex {
            graph: DashMap::new(),
            points: DashMap::new(),
            start_node: None,
            r,
            alpha,
            l_build,
            locks: DashMap::new(),
            metric,
        }
    }

    pub fn insert(&mut self, point: &VectorPoint) {
        debug_assert!(self.alpha > 1.0);
        self.points.insert(point.id, point.clone());
        self.locks.insert(point.id, Arc::new(Mutex::new(())));
        //Let's initialize the graph with the first point
        if self.start_node.is_none() {
            self.start_node = Some(point.id);
            self.graph.insert(point.id, Vec::new());
            return; //Just inserted first node, so no need to continue
        }

        // Run greedy search to find candidates
        debug_assert!(self.start_node.is_some());
        let (_, visited) =
            self.greedy_search(self.start_node.unwrap(), &point.vector, 1, self.l_build);

        // Prune the visited point to get out-neighbors for the new point
        let out_neighbors = self.robust_prune(point.id, visited, self.alpha);

        self.graph.insert(point.id, out_neighbors.clone());

        // Add backward edges and prune if necessary
        for neighbor_id in &out_neighbors {
            if let Some(lock_ref) = self.locks.get(neighbor_id) {
                let _guard = lock_ref.lock().unwrap();

                // Collect the data we need, then drop the lock
                let needs_pruning = if let Some(mut entry) = self.graph.get_mut(neighbor_id) {
                    entry.push(point.id);
                    let needs_prune = entry.len() > self.r;
                    let candidates = if needs_prune { entry.clone() } else { Vec::new() };
                    drop(entry); // Explicitly drop the lock
                    (needs_prune, candidates)
                } else {
                    (false, Vec::new())
                };

                // Now the lock is dropped, safe to call robust_prune
                if needs_pruning.0 {
                    let pruned = self.robust_prune(*neighbor_id, needs_pruning.1, self.alpha);

                    // Reacquire the lock to update
                    if let Some(mut entry) = self.graph.get_mut(neighbor_id) {
                        *entry = pruned;
                    }
                }
            }
        }
    }

    pub fn greedy_search(
        &self, start_id: u32, query: &VectorData, k: usize, l: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut candidates = BinaryHeap::with_capacity(l + 1); //min-heap
        let mut top_l = BinaryHeap::with_capacity(l + 1); // max-heap (using Reverse)

        let mut visited = HashSet::with_capacity(l * 2);
        let mut expanded = Vec::with_capacity(l);

        let start_point = self.points.get(&start_id).expect("Start node not found");
        let start_dist = start_point.distance_to_vector(query, self.metric);

        let start = SearchCandidate { point_id: start_id, distance: start_dist };

        candidates.push(start.clone());
        top_l.push(std::cmp::Reverse(start));
        visited.insert(start_id);

        let mut worst_top_l_dist = start_dist;

        while let Some(curr) = candidates.pop() {
            if curr.distance > worst_top_l_dist && top_l.len() >= l {
                break;
            }
            expanded.push(curr.point_id);

            let Some(neighbors) = self.graph.get(&curr.point_id) else {
                continue;
            };

            for &neighbor_id in neighbors.value() {
                if !visited.insert(neighbor_id) {
                    continue;
                }

                let Some(neighbor_point) = self.points.get(&neighbor_id) else {
                    continue;
                };

                let dist = neighbor_point.distance_to_vector(query, self.metric);

                if top_l.len() < l {
                    let cand = SearchCandidate { point_id: neighbor_id, distance: dist };
                    candidates.push(cand.clone());
                    top_l.push(std::cmp::Reverse(cand));
                    if dist > worst_top_l_dist {
                        worst_top_l_dist = dist;
                    }
                } else if dist < worst_top_l_dist {
                    let cand = SearchCandidate { point_id: neighbor_id, distance: dist };
                    candidates.push(cand.clone());
                    top_l.pop();
                    top_l.push(std::cmp::Reverse(cand));
                    if let Some(worst) = top_l.peek() {
                        worst_top_l_dist = worst.0.distance;
                    }
                }
            }
        }

        let mut results: Vec<SearchCandidate> = top_l.into_iter().map(|r| r.0).collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        let top_k: Vec<u32> = results.iter().take(k).map(|c| c.point_id).collect();

        (top_k, expanded)
    }

    fn robust_prune(&self, point_id: u32, mut candidates: Vec<u32>, alpha: f64) -> Vec<u32> {
        debug_assert!(alpha >= 1.0, "alpha must be >= 1.0");

        let point = match self.points.get(&point_id) {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Step 1: V ← (V + N_out(p)) - {p}
        let mut seen = HashSet::new();
        seen.insert(point_id);
        candidates.retain(|id| *id != point_id);
        if let Some(neighbors) = self.graph.get(&point_id) {
            for n in neighbors.value() {
                if seen.insert(*n) {
                    candidates.push(*n);
                }
            }
        }

        let mut candidate_data: Vec<SearchCandidate> = candidates
            .into_par_iter()
            .filter_map(|id| {
                self.points.get(&id).map(|p| SearchCandidate {
                    point_id: id,
                    distance: point.distance(&p, self.metric),
                })
            })
            .collect();

        candidate_data.sort_by(|a, b| b.cmp(a));

        // Step 2 & 3: Prune candidates that don't satisfy the alpha-RNG condition
        let mut new_neighbors = Vec::with_capacity(self.r);
        let mut new_neighbor_points: Vec<VectorPoint> = Vec::with_capacity(self.r);

        for candidate in candidate_data {
            let Some(point_c) = self.points.get(&candidate.point_id) else {
                continue;
            };

            let keep = !new_neighbor_points.iter().any(|point_n| {
                alpha * point_n.distance(&point_c, self.metric) <= candidate.distance
            });

            if keep {
                new_neighbors.push(candidate.point_id);
                new_neighbor_points.push(point_c.clone());
            }
            if new_neighbors.len() >= self.r {
                break;
            }
        }

        new_neighbors
    }

    pub fn delete(&mut self, delete_list: &[u32]) {
        let delete_set: HashSet<_> = delete_list.iter().cloned().collect();

        let updates: Vec<_> = self
            .graph
            .par_iter()
            .filter_map(|point_ref| {
                let point_id = *point_ref.key();
                if delete_set.contains(&point_id) {
                    return None;
                }
                let neighbors = point_ref.value();
                let deleted_neighbors: Vec<_> =
                    neighbors.iter().filter(|&n| delete_set.contains(n)).cloned().collect();

                if !deleted_neighbors.is_empty() {
                    let mut candidates: HashSet<_> =
                        neighbors.iter().filter(|&n| !delete_set.contains(n)).cloned().collect();

                    // Add neighbors of deleted neighbors
                    for deleted_id in &deleted_neighbors {
                        if let Some(del_neighbors) = self.graph.get(deleted_id) {
                            for n in del_neighbors.value() {
                                if !delete_set.contains(n) {
                                    candidates.insert(*n);
                                }
                            }
                        }
                    }

                    let candidate_vec: Vec<_> = candidates.into_iter().collect();
                    let new_neighbors = self.robust_prune(point_id, candidate_vec, self.alpha);
                    Some((point_id, new_neighbors))
                } else {
                    None
                }
            })
            .collect();

        // Apply all updates after iteration completes
        for (point_id, new_neighbors) in updates {
            if let Some(mut neighbors_mut) = self.graph.get_mut(&point_id) {
                *neighbors_mut = new_neighbors;
            }
        }

        // Remove deleted points
        for point_id in &delete_set {
            self.graph.remove(point_id);
            self.points.remove(point_id);
            self.locks.remove(point_id);

            // Update start_node if it was deleted
            if self.start_node == Some(*point_id) {
                self.start_node = self.graph.iter().next().map(|r| *r.key());
            }
        }
    }

    /// Search for k nearest neighbors
    #[allow(dead_code)]
    pub(crate) fn search(&self, query: &VectorData, k: usize, l: usize) -> Vec<u32> {
        if let Some(start_id) = &self.start_node {
            let (result, _) = self.greedy_search(*start_id, query, k, l);
            result
        } else {
            Vec::new()
        }
    }

    pub(crate) fn greedy_search_for_lti(
        &self, query: &VectorData, l: usize, visited_map: &mut HashMap<u32, VectorPoint>,
    ) {
        if self.start_node.is_none() {
            return;
        }
        let start_id = self.start_node.unwrap();
        let (_, visited_ids) = self.greedy_search(start_id, query, l, l);

        for id in visited_ids {
            if let Some(point) = self.points.get(&id) {
                visited_map.insert(id, point.clone());
            }
        }
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }
}

#[cfg(test)]
mod in_mem_index_test {
    use system::metric::MetricType;
    use system::vector_data::VectorData;
    use system::vector_point::VectorPoint;

    use super::*;

    #[test]
    fn test_in_mem_index() {
        let mut index = InMemIndex::new(32, 1.2, 50, MetricType::L2);

        for i in 0..100 {
            let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.insert(&VectorPoint::new(i, vector));
        }

        let query = VectorData::from_f32(vec![50.0, 100.0, 150.0]);
        let results = index.search(&query, 5, 20);

        assert!(results.len() <= 5);
        println!("Search results: {:?}", results);
    }

    macro_rules! test_index_matrix {
        (
            $metric:expr,
            [
                $(
                    ( $test_name:ident, $dim:expr, $alpha:expr, $l_build:expr, $num_points:expr, $num_queries:expr )
                ),* $(,)?
            ]
        ) => {
            $(
                #[test]
                fn $test_name() {
                    use rand::Rng;
                    let mut rng = rand::rng();

                    let mut index = InMemIndex::new($dim, $alpha, $l_build, $metric);

                    for i in 0..$num_points {
                        let vec: Vec<f32> = (0..$dim).map(|_| rng.random::<f32>()).collect();
                        let point = VectorPoint::new(i, VectorData::from_f32(vec));
                        index.insert(&point);
                    }

                    for qid in 0..$num_queries {
                        let query_vec: Vec<f32> = (0..$dim).map(|_| rng.random::<f32>()).collect();
                        let query = VectorData::from_f32(query_vec);

                        let results = index.search(&query, 5, 20);

                        assert!(
                            results.len() <= 5,
                            "Expected ≤5 results but got {} (test: {})",
                            results.len(),
                            stringify!($test_name)
                        );

                        println!(
                            "[{}] metric={:?}, dim={}, alpha={}, l_build={}, query={}, results={:?}",
                            stringify!($test_name),
                            $metric,
                            $dim,
                            $alpha,
                            $l_build,
                            qid,
                            results
                        );
                    }
                }
            )*
        };
    }

    test_index_matrix!(
        MetricType::L2,
        [
            (test_index_small_l2, 32, 1.2, 50, 100, 3),
            (test_index_medium_l2, 128, 1.1, 1000, 200, 50),
            (test_index_large_l2, 1024, 1.5, 10000, 400, 100)
        ]
    );

    test_index_matrix!(
        MetricType::L2Squared,
        [
            (test_index_small_l2_squared, 32, 1.2, 50, 100, 3),
            (test_index_medium_l2_squared, 64, 1.1, 1000, 200, 50),
            (test_index_large_l2_squared, 1536, 1.5, 10000, 400, 100)
        ]
    );

    test_index_matrix!(
        MetricType::Dot,
        [
            (test_index_small_dot, 32, 1.2, 50, 100, 3),
            (test_index_medium_dot, 64, 1.1, 1000, 200, 50),
            (test_index_large_dot, 1024, 1.5, 10000, 300, 100)
        ]
    );

    test_index_matrix!(
        MetricType::Cosine,
        [
            (test_index_small_cosine, 32, 1.2, 50, 100, 3),
            (test_index_medium_cosine, 64, 1.1, 1000, 200, 50),
            (test_index_large_cosine, 896, 1.3, 10000, 400, 100)
        ]
    );
}

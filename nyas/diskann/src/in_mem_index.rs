use crate::utils::SearchCandidate;
use dashmap::DashMap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use storage::metric::MetricType;
use storage::vector_data::VectorData;
use storage::vector_point::VectorPoint;

pub struct InMemIndex<ID>
where
    ID: Debug + Clone + Hash + Eq + PartialEq + Send + Sync + 'static,
{
    /// Graph adjacency lists (out-neighbors)
    pub graph: DashMap<Arc<ID>, Vec<Arc<ID>>>,
    /// Points stored in the index
    pub points: DashMap<Arc<ID>, VectorPoint<ID>>,
    /// Start node for navigation
    pub start_node: Option<Arc<ID>>,
    /// Maximum out-degree
    pub r: usize,
    /// Alpha parameter for RNG property
    pub alpha: f64,
    /// Search list size for building
    pub l_build: usize,
    /// Thread-safe locks for concurrent access
    pub locks: DashMap<Arc<ID>, Arc<Mutex<()>>>,
    /// Metric type for distance calculation
    pub metric: MetricType,
}

impl<ID> InMemIndex<ID>
where
    ID: Debug + Clone + Hash + Eq + PartialEq + Send + Sync + 'static,
{
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

    pub fn insert(&mut self, point: &VectorPoint<ID>) {
        debug_assert!(self.alpha > 1.0);

        self.points.insert(Arc::clone(&point.id), point.clone());
        self.locks.insert(Arc::clone(&point.id), Arc::new(Mutex::new(())));

        //Let's initialize the graph with the first point
        if self.start_node.is_none() {
            self.start_node = Some(Arc::clone(&point.id));
            self.graph.insert(Arc::clone(&point.id), Vec::new());
            return; //Just inserted first node, so no need to continue
        }

        // Run greedy search to find candidates
        debug_assert!(self.start_node.is_some());
        let (_, visited) =
            self.greedy_search(&self.start_node.clone().unwrap(), &point.vector, 1, self.l_build);

        // Prune the visited point to get out-neighbors for the new point
        let out_neighbors = self.robust_prune(&point.id, visited, self.alpha);
        self.graph.insert(Arc::clone(&point.id), out_neighbors.clone());

        // Add backward edges and prune if necessary
        for neighbor_id in &out_neighbors {
            if let Some(lock_ref) = self.locks.get(neighbor_id) {
                let _guard = lock_ref.lock().unwrap();

                // Collect the data we need, then drop the lock
                let needs_pruning = if let Some(mut entry) = self.graph.get_mut(neighbor_id) {
                    entry.push(Arc::clone(&point.id));
                    let needs_prune = entry.len() > self.r;
                    let candidates = if needs_prune { entry.clone() } else { Vec::new() };
                    drop(entry); // Explicitly drop the lock
                    (needs_prune, candidates)
                } else {
                    (false, Vec::new())
                };

                // Now the lock is dropped, safe to call robust_prune
                if needs_pruning.0 {
                    let pruned = self.robust_prune(neighbor_id, needs_pruning.1, self.alpha);

                    // Reacquire the lock to update
                    if let Some(mut entry) = self.graph.get_mut(neighbor_id) {
                        *entry = pruned;
                    }
                }
            }
        }

        // Update start node to the most recently inserted point
        self.start_node = Some(Arc::clone(&point.id));
    }

    pub fn greedy_search(
        &self, start_id: &Arc<ID>, query: &VectorData, k: usize, l: usize,
    ) -> (Vec<Arc<ID>>, Vec<Arc<ID>>) {
        let mut candidates = BinaryHeap::new(); //min-heap
        let mut visited = HashSet::new();

        let start_point = self.points.get(start_id).unwrap();

        candidates.push(SearchCandidate {
            point_id: Arc::clone(start_id),
            distance: start_point.distance_to_vector(query, self.metric),
        });

        while let Some(current) = candidates.pop()
            && !visited.contains(&current.point_id)
        {
            visited.insert(Arc::clone(&current.point_id));

            if let Some(neighbors) = self.graph.get(&current.point_id) {
                for neighbor_id in neighbors.value() {
                    if !visited.contains(neighbor_id) {
                        self.push_candidate(neighbor_id, query, l, &mut candidates);
                    }
                }
            }

            if visited.len() >= l {
                break;
            }
        }

        let visited: Vec<Arc<ID>> = visited.into_iter().collect();
        let mut result_with_dist: Vec<_> = visited
            .iter()
            .filter_map(|id| {
                self.points.get(id).map(|p| (id, p.distance_to_vector(query, self.metric)))
            })
            .collect();

        if result_with_dist.len() > k {
            let _ = result_with_dist.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)
            });
            result_with_dist.truncate(k);
        }

        let top_k: Vec<Arc<ID>> =
            result_with_dist.into_iter().map(|(id, _)| Arc::clone(id)).collect();

        (top_k, visited)
    }

    fn push_candidate(
        &self, neighbor_id: &Arc<ID>, query: &VectorData, l: usize,
        candidates: &mut BinaryHeap<SearchCandidate<Arc<ID>>>,
    ) {
        if let Some(neighbor_point) = self.points.get(neighbor_id) {
            if candidates.len() < l {
                candidates.push(SearchCandidate {
                    point_id: Arc::clone(neighbor_id),
                    distance: neighbor_point.distance_to_vector(query, self.metric),
                });
            } else if let Some(mut worst) = candidates.peek_mut() {
                let distance = neighbor_point.distance_to_vector(query, self.metric);
                if distance < worst.distance {
                    *worst = SearchCandidate { point_id: Arc::clone(neighbor_id), distance };
                }
            }
        }
    }

    fn robust_prune(
        &self, point_id: &Arc<ID>, mut candidates: Vec<Arc<ID>>, alpha: f64,
    ) -> Vec<Arc<ID>> {
        debug_assert!(alpha >= 1.0, "alpha must be >= 1.0");

        let point = match self.points.get(point_id) {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Step 1: V ← (V + N_out(p)) - {p}
        let mut seen = HashSet::new();
        seen.insert(Arc::clone(point_id));
        candidates.retain(|id| id != point_id);
        if let Some(neighbors) = self.graph.get(point_id) {
            for n in neighbors.value() {
                if seen.insert(Arc::clone(n)) {
                    candidates.push(Arc::clone(n));
                }
            }
        }

        // Step 2: N_out(p) ← {}
        let mut new_neighbors = Vec::with_capacity(self.r);

        // Step 3: while V ≠ {}
        while !candidates.is_empty() {
            let (best_idx, _) = candidates
                .par_iter()
                .enumerate()
                .filter_map(|(i, cid)| {
                    self.points.get(cid).map(|cp| (i, point.distance(&cp, self.metric)))
                })
                .reduce(|| (usize::MAX, f64::MAX), |a, b| if a.1 < b.1 { a } else { b });

            if best_idx == usize::MAX {
                break;
            }

            let best_id = candidates.swap_remove(best_idx);
            new_neighbors.push(Arc::clone(&best_id));

            if new_neighbors.len() >= self.r {
                break;
            }

            // alpha-RNG filtering
            let Some(best_point) = self.points.get(&best_id) else {
                continue;
            };

            candidates = candidates
                .into_par_iter()
                .filter(|candidate_id| {
                    if let Some(candidate_point) = self.points.get(candidate_id) {
                        alpha * best_point.distance(&candidate_point, self.metric)
                            > point.distance(&candidate_point, self.metric)
                    } else {
                        false
                    }
                })
                .collect();
        }

        new_neighbors
    }

    pub fn delete(&mut self, delete_list: &[Arc<ID>]) {
        let delete_set: HashSet<_> = delete_list.iter().cloned().collect();

        let mut updates = Vec::new();

        for point_ref in self.graph.iter() {
            let point_id = point_ref.key();
            if delete_set.contains(point_id) {
                continue;
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
                                candidates.insert(Arc::clone(n));
                            }
                        }
                    }
                }

                let candidate_vec: Vec<_> = candidates.into_iter().collect();
                let new_neighbors = self.robust_prune(point_id, candidate_vec, self.alpha);
                updates.push((Arc::clone(point_id), new_neighbors));
            }
        }

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
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &VectorData, k: usize, l: usize) -> Vec<Arc<ID>> {
        if let Some(start_id) = &self.start_node {
            let (result, _) = self.greedy_search(start_id, query, k, l);
            result
        } else {
            Vec::new()
        }
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }
}

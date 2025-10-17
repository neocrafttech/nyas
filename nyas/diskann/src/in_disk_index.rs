use crate::in_mem_index::InMemIndex;
use dashmap::DashMap;
use dashmap::DashSet;
use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use storage::metric::MetricType;
use storage::vector_data::VectorData;
use storage::vector_point::VectorPoint;

use std::fmt::Debug;

pub struct InDiskIndex<ID>
where
    ID: Debug + Clone + Hash + Eq + PartialEq + Send + Sync + 'static,
{
    /// Long-term index on SSD (simplified as in-memory for this demo)
    lti: Arc<RwLock<InMemIndex<ID>>>,
    /// Read-write temporary index
    rw_temp_index: Arc<RwLock<InMemIndex<ID>>>,
    /// Read-only temporary indices
    ro_temp_indices: Arc<RwLock<Vec<InMemIndex<ID>>>>,
    /// Delete list
    delete_list: Arc<DashSet<Arc<ID>>>,
    /// Maximum size of temp index before merge
    max_temp_size: usize,
    /// Parameters
    r: usize,
    alpha: f64,
    l_build: usize,
    metric: MetricType,
}

impl<ID> InDiskIndex<ID>
where
    ID: Debug + Clone + Hash + Eq + PartialEq + Send + Sync + 'static,
{
    pub fn new(
        r: usize, alpha: f64, l_build: usize, max_temp_size: usize, metric: MetricType,
    ) -> Self {
        InDiskIndex {
            lti: Arc::new(RwLock::new(InMemIndex::new(r, alpha, l_build, metric))),
            rw_temp_index: Arc::new(RwLock::new(InMemIndex::new(r, alpha, l_build, metric))),
            ro_temp_indices: Arc::new(RwLock::new(Vec::new())),
            delete_list: Arc::new(DashSet::new()),
            max_temp_size,
            r,
            alpha,
            l_build,
            metric,
        }
    }

    pub fn insert(&self, point: &VectorPoint<ID>) -> Result<(), String> {
        let mut rw_temp = self.rw_temp_index.write().unwrap();
        rw_temp.insert(point);

        // Check if we need to snapshot
        if rw_temp.size() >= self.max_temp_size {
            drop(rw_temp);
            self.snapshot_temp_index()?;
        }

        Ok(())
    }

    pub fn delete(&self, point_id: Arc<ID>) {
        self.delete_list.insert(point_id);
    }

    pub fn search(&self, query: &VectorData, k: usize, l: usize) -> Vec<Arc<ID>> {
        let mut all_results = Vec::new();

        // Search LTI
        let lti = self.lti.read().unwrap();
        let lti_results = lti.search(query, k, l);
        all_results.extend(lti_results);

        // Search RW-TempIndex
        let rw_temp = self.rw_temp_index.read().unwrap();
        let rw_results = rw_temp.search(query, k, l);
        all_results.extend(rw_results);

        // Search RO-TempIndices
        let ro_temps = self.ro_temp_indices.read().unwrap();
        for ro_temp in ro_temps.iter() {
            let ro_results = ro_temp.search(query, k, l);
            all_results.extend(ro_results);
        }

        // Filter out deleted points
        all_results.retain(|id| !self.delete_list.contains(id));

        // Deduplicate and sort by distance
        let mut unique_results: Vec<_> =
            all_results.into_iter().collect::<DashSet<_>>().into_iter().collect();

        // Get distances and sort
        let lti_read = self.lti.read().unwrap();
        let rw_read = self.rw_temp_index.read().unwrap();

        unique_results.sort_by(|a, b| {
            let dist_a = self.get_point_distance(a.clone(), query, &lti_read, &rw_read);
            let dist_b = self.get_point_distance(b.clone(), query, &lti_read, &rw_read);
            dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
        });

        unique_results.into_iter().take(k).collect()
    }

    fn get_point_distance(
        &self, point_id: Arc<ID>, query: &VectorData, lti: &InMemIndex<ID>, rw: &InMemIndex<ID>,
    ) -> f64 {
        if let Some(p) = lti.points.get(&point_id) {
            return p.distance_to_vector(query, self.metric);
        }
        if let Some(p) = rw.points.get(&point_id) {
            return p.distance_to_vector(query, self.metric);
        }

        let ro_temps = self.ro_temp_indices.read().unwrap();
        for ro in ro_temps.iter() {
            if let Some(p) = ro.points.get(&point_id) {
                return p.distance_to_vector(query, self.metric);
            }
        }
        f64::MAX
    }

    /// Snapshot RW-TempIndex to RO-TempIndex
    fn snapshot_temp_index(&self) -> Result<(), String> {
        let mut rw_temp = self.rw_temp_index.write().unwrap();
        let mut ro_temps = self.ro_temp_indices.write().unwrap();

        // Clone the current RW index to RO
        let snapshot = InMemIndex {
            graph: rw_temp.graph.clone(),
            points: rw_temp.points.clone(),
            start_node: rw_temp.start_node.clone(),
            r: rw_temp.r,
            alpha: rw_temp.alpha,
            l_build: rw_temp.l_build,
            locks: DashMap::new(), // RO doesn't need locks
            metric: self.metric,
        };

        ro_temps.push(snapshot);

        // Create new empty RW-TempIndex
        *rw_temp = InMemIndex::new(self.r, self.alpha, self.l_build, self.metric);

        Ok(())
    }

    /// Streaming merge (simplified version)
    pub fn streaming_merge(&self) -> Result<(), String> {
        // Collect all points from RO-TempIndices
        let ro_temps = self.ro_temp_indices.read().unwrap();
        let mut all_inserts = Vec::new();

        for ro_temp in ro_temps.iter() {
            for point_ref in ro_temp.points.iter() {
                all_inserts.push(point_ref.value().clone());
            }
        }
        drop(ro_temps);

        // Get delete list
        let deletes: Vec<_> = self.delete_list.iter().map(|e| e.clone()).collect();

        // Apply deletes to LTI
        let mut lti = self.lti.write().unwrap();
        if !deletes.is_empty() {
            lti.delete(&deletes);
        }

        // Apply inserts to LTI
        for point in all_inserts {
            lti.insert(&point);
        }
        drop(lti);

        // Clear RO-TempIndices and DeleteList
        let mut ro_temps = self.ro_temp_indices.write().unwrap();
        ro_temps.clear();

        self.delete_list.clear();
        Ok(())
    }
}

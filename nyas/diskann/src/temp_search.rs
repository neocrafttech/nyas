use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

// Constants
const MAX_N_SECTOR_READS: usize = 128;
const SECTOR_LEN: usize = 4096;
const WASTE_THRESHOLD: f64 = 0.1;

/// Neighbor structure for search results
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
    pub flag: bool,
    pub visited: bool,
}

impl Neighbor {
    pub fn new(id: u32, distance: f32, flag: bool) -> Self {
        Self {
            id,
            distance,
            flag,
            visited: false,
        }
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// I/O request structure
#[derive(Debug)]
pub struct IORequest {
    pub offset: u64,
    pub size: usize,
    pub buf: *mut u8,
    pub u_loc_offset: u32,
    pub max_node_len: usize,
    pub finished: bool,
}

unsafe impl Send for IORequest {}
unsafe impl Sync for IORequest {}

impl IORequest {
    pub fn new(
        offset: u64,
        size: usize,
        buf: *mut u8,
        u_loc_offset: u32,
        max_node_len: usize,
    ) -> Self {
        Self {
            offset,
            size,
            buf,
            u_loc_offset,
            max_node_len,
            finished: false,
        }
    }
}

/// In-flight I/O tracker
#[derive(Debug)]
pub struct IoT {
    pub nbr: Neighbor,
    pub page_id: u32,
    pub loc: u32,
    pub read_req_idx: usize,
}

impl IoT {
    pub fn finished(&self, requests: &[IORequest]) -> bool {
        requests[self.read_req_idx].finished
    }
}

impl PartialEq for IoT {
    fn eq(&self, other: &Self) -> bool {
        self.nbr.distance == other.nbr.distance
    }
}

impl Eq for IoT {}

impl PartialOrd for IoT {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.nbr.distance.partial_cmp(&other.nbr.distance)
    }
}

impl Ord for IoT {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Query statistics
#[derive(Debug, Default, Clone)]
pub struct QueryStats {
    pub io_us: u64,
    pub io_us1: u64,
    pub cpu_us: u64,
    pub cpu_us1: u64,
    pub cpu_us2: u64,
    pub total_us: f64,
    pub n_ios: usize,
    pub n_cmps: usize,
}

/// Disk node representation
pub struct DiskNode<T> {
    pub coords: Vec<T>,
    pub nbrs: Vec<u32>,
    pub nnbrs: u32,
    pub labels: Vec<u32>,
}

/// Query buffer for thread-local data
pub struct QueryBuffer<T> {
    pub aligned_query_t: Vec<T>,
    pub coord_scratch: Vec<T>,
    pub sector_scratch: Vec<u8>,
    pub aligned_dist_scratch: Vec<f32>,
    pub visited: HashSet<u32>,
    pub sector_idx: usize,
    pub reqs: Vec<IORequest>,
}

impl<T: Default + Clone> QueryBuffer<T> {
    pub fn new(data_dim: usize, aligned_dim: usize) -> Self {
        Self {
            aligned_query_t: vec![T::default(); aligned_dim],
            coord_scratch: vec![T::default(); data_dim * 1024],
            sector_scratch: vec![0u8; MAX_N_SECTOR_READS * SECTOR_LEN],
            aligned_dist_scratch: vec![0.0f32; 1024],
            visited: HashSet::new(),
            sector_idx: 0,
            reqs: (0..MAX_N_SECTOR_READS)
                .map(|_| IORequest::new(0, 0, std::ptr::null_mut(), 0, 0))
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        self.visited.clear();
        self.sector_idx = 0;
    }
}

/// Abstract selector for filtering results
pub trait AbstractSelector {
    fn is_member(&self, id: u32, filter_data: Option<&[u8]>, labels: &[u32]) -> bool;
}

/// Distance comparison trait
pub trait DistanceCompare<T> {
    fn compare(&self, a: &[T], b: &[T], dim: usize) -> f32;
}

/// Neighbor handler trait for distance computations
pub trait NeighborHandler<T> {
    fn initialize_query(&mut self, query: &[T], query_buf: &mut QueryBuffer<T>);
    fn compute_dists(&mut self, query_buf: &mut QueryBuffer<T>, ids: &[u32], count: usize);
}

/// File reader trait for I/O operations
pub trait FileReader {
    fn get_ctx(&self, flags: Option<u32>) -> *mut u8;
    fn send_read_no_alloc(&self, req: &mut IORequest, ctx: *mut u8);
    fn poll_all(&self, ctx: *mut u8);
}

/// Memory index trait for hybrid search
pub trait MemoryIndex<T> {
    fn search_with_tags_fast(
        &self,
        query: &[T],
        k: usize,
        tags: &mut [u32],
        dists: &mut [f32],
    );
}

/// Insert neighbor into sorted pool
fn insert_into_pool(pool: &mut [Neighbor], cur_size: usize, nn: Neighbor) -> usize {
    if cur_size == 0 {
        pool[0] = nn;
        return 0;
    }

    // Binary search for insertion position
    let pos = pool[..cur_size]
        .binary_search_by(|a| {
            a.distance
                .partial_cmp(&nn.distance)
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or_else(|e| e);

    if pos < cur_size {
        // Shift elements to make room
        pool.copy_within(pos..cur_size, pos + 1);
        pool[pos] = nn;
    }

    pos
}

/// Main SSD Index implementation for pipe search
pub struct SSDIndex<T, TagT = u32> {
    pub meta: IndexMetadata,
    pub params: SearchParameters,
    pub idx_lock_table: Arc<LockTable>,
    aligned_dim: usize,
}

#[derive(Debug, Clone)]
pub struct IndexMetadata {
    pub data_dim: usize,
    pub entry_point_id: u32,
    pub max_node_len: usize,
}

#[derive(Debug, Clone)]
pub struct SearchParameters {
    pub r: usize,
}

pub struct LockTable {
    // Simplified lock table implementation
}

impl LockTable {
    pub fn new() -> Self {
        Self {}
    }

    pub fn lock_idx(&self, _id: u32, _neighbors: &[u32], _read_only: bool) {}
    pub fn unlock_idx(&self, _id: u32) {}
}

impl<T, TagT> SSDIndex<T, TagT>
where
    T: Default + Clone + Copy + Send + Sync + 'static,
    TagT: Default + Clone + Copy + Send + Sync + From<u32> + 'static,
{
    /// Pipelined search algorithm
    #[allow(clippy::too_many_arguments)]
    pub fn pipe_search<D, N, F, M, S>(
        &self,
        query: &[T],
        k_search: usize,
        mem_l: usize,
        l_search: usize,
        res_tags: &mut [TagT],
        distances: Option<&mut [f32]>,
        beam_width: usize,
        stats: Option<&mut QueryStats>,
        dist_cmp: &D,
        nbr_handler: &mut N,
        reader: &F,
        mem_index: Option<&M>,
        selector: Option<&S>,
        filter_data: Option<&[u8]>,
        relaxed_monotonicity_l: usize,
    ) -> usize
    where
        D: DistanceCompare<T>,
        N: NeighborHandler<T>,
        F: FileReader,
        M: MemoryIndex<T>,
        S: AbstractSelector,
    {
        if beam_width > MAX_N_SECTOR_READS {
            panic!("Beamwidth cannot be higher than MAX_N_SECTOR_READS");
        }

        let mut query_buf = QueryBuffer::new(self.meta.data_dim, self.aligned_dim);
        query_buf.aligned_query_t[..query.len()].copy_from_slice(query);
        query_buf.reset();

        let ctx = reader.get_ctx(Some(0)); // IORING_SETUP_SQPOLL flag

        // Relaxed monotonicity mode tracking
        let mut converge_size: i64 = -1;

        let query_ref = &query_buf.aligned_query_t;
        let mut retset = vec![Neighbor::new(0, 0.0, false); mem_l + self.params.r + l_search * 10];
        let mut cur_list_size = 0;
        let mut full_retset = Vec::with_capacity(l_search * 10);

        // Initialize statistics
        if let Some(s) = stats.as_mut() {
            s.io_us = 0;
            s.io_us1 = 0;
            s.cpu_us = 0;
            s.cpu_us1 = 0;
            s.cpu_us2 = 0;
            s.n_ios = 0;
            s.n_cmps = 0;
        }

        let mut n_computes = 0;
        let mut id_buf_map: HashMap<u32, DiskNode<T>> = HashMap::new();
        let mut on_flight_ios: VecDeque<IoT> = VecDeque::new();

        // Closure: compute exact distances and push to full_retset
        let mut compute_exact_dists_and_push = |node: &DiskNode<T>, id: u32| -> f32 {
            let node_fp_coords = &node.coords;
            let cur_expanded_dist =
                dist_cmp.compare(query_ref, node_fp_coords, self.aligned_dim);

            // Filter check
            if selector.is_none()
                || selector
                    .unwrap()
                    .is_member(id, filter_data, &node.labels)
            {
                full_retset.push(Neighbor::new(id, cur_expanded_dist, true));
            }

            cur_expanded_dist
        };

        // Closure: compute distances for neighbors and push to retset
        let mut compute_and_push_nbrs =
            |node: &mut DiskNode<T>, nk: &mut usize, visited: &mut HashSet<u32>| {
                let mut nbors_cand_size = 0;
                let mut candidates = Vec::new();

                for &nbr_id in node.nbrs.iter().take(node.nnbrs as usize) {
                    if !visited.contains(&nbr_id) {
                        candidates.push(nbr_id);
                        visited.insert(nbr_id);
                        nbors_cand_size += 1;
                    }
                }

                n_computes += nbors_cand_size;

                if nbors_cand_size > 0 {
                    node.nbrs[..nbors_cand_size].copy_from_slice(&candidates);
                    nbr_handler.compute_dists(&mut query_buf, &node.nbrs, nbors_cand_size);

                    for m in 0..nbors_cand_size {
                        let nbor_id = node.nbrs[m];
                        let nbor_dist = query_buf.aligned_dist_scratch[m];

                        if let Some(s) = stats.as_mut() {
                            s.n_cmps += 1;
                        }

                        if nbor_dist >= retset[cur_list_size - 1].distance
                            && cur_list_size == l_search
                        {
                            continue;
                        }

                        let nn = Neighbor::new(nbor_id, nbor_dist, true);
                        let r = insert_into_pool(&mut retset, cur_list_size, nn);

                        if cur_list_size < l_search {
                            cur_list_size += 1;
                            if cur_list_size >= retset.len() {
                                retset.resize(2 * cur_list_size, Neighbor::new(0, 0.0, false));
                            }
                        }

                        if r < *nk {
                            *nk = r;
                        }
                    }
                }
            };

        // Closure: add to retset
        let mut add_to_retset = |node_ids: &[u32], n_ids: usize, dists: &[f32]| {
            for i in 0..n_ids {
                retset[cur_list_size] = Neighbor::new(node_ids[i], dists[i], true);
                cur_list_size += 1;
                query_buf.visited.insert(node_ids[i]);
            }
        };

        // Initialize with memory index or entry point
        let mut cur_beam_width = beam_width.min(4);
        let mut mem_tags = vec![0u32; mem_l];
        let mut mem_dists = vec![0.0f32; mem_l];

        #[cfg(feature = "overlap_init")]
        {
            if mem_l > 0 && mem_index.is_some() {
                mem_index.unwrap().search_with_tags_fast(
                    query,
                    mem_l,
                    &mut mem_tags,
                    &mut mem_dists,
                );
                add_to_retset(&mem_tags, mem_l.min(l_search), &mem_dists);
            } else {
                nbr_handler.initialize_query(query, &mut query_buf);
                nbr_handler.compute_dists(
                    &mut query_buf,
                    &[self.meta.entry_point_id],
                    1,
                );
                add_to_retset(
                    &[self.meta.entry_point_id],
                    1,
                    &query_buf.aligned_dist_scratch,
                );
            }
        }

        #[cfg(not(feature = "overlap_init"))]
        {
            nbr_handler.initialize_query(query, &mut query_buf);
            if mem_l > 0 && mem_index.is_some() {
                mem_index.unwrap().search_with_tags_fast(
                    query,
                    mem_l,
                    &mut mem_tags,
                    &mut mem_dists,
                );
                nbr_handler.compute_dists(&mut query_buf, &mem_tags, mem_l);
                add_to_retset(&mem_tags, mem_l.min(l_search), &query_buf.aligned_dist_scratch);
            } else {
                nbr_handler.compute_dists(
                    &mut query_buf,
                    &[self.meta.entry_point_id],
                    1,
                );
                add_to_retset(
                    &[self.meta.entry_point_id],
                    1,
                    &query_buf.aligned_dist_scratch,
                );
            }
            retset[..cur_list_size].sort_unstable();
        }

        // Closure: send read request
        let send_read_req = |item: &mut Neighbor,
                             on_flight: &mut VecDeque<IoT>,
                             qbuf: &mut QueryBuffer<T>,
                             stats_ref: &mut Option<&mut QueryStats>|
         -> bool {
            item.flag = false;

            self.idx_lock_table.lock_idx(item.id, &[], true);
            let loc = self.id2loc(item.id);
            let pid = self.loc_sector_no(loc);

            let cur_buf_idx = qbuf.sector_idx;
            let buf_ptr = unsafe {
                qbuf.sector_scratch
                    .as_mut_ptr()
                    .add(cur_buf_idx * self.size_per_io())
            };

            qbuf.reqs[cur_buf_idx] = IORequest::new(
                (pid as u64) * SECTOR_LEN as u64,
                self.size_per_io(),
                buf_ptr,
                self.u_loc_offset(loc),
                self.meta.max_node_len,
            );

            reader.send_read_no_alloc(&mut qbuf.reqs[cur_buf_idx], ctx);

            on_flight.push_back(IoT {
                nbr: *item,
                page_id: pid,
                loc,
                read_req_idx: cur_buf_idx,
            });

            qbuf.sector_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

            if let Some(s) = stats_ref.as_mut() {
                s.n_ios += 1;
            }

            true
        };

        // Closure: poll all I/O completions
        let poll_all = |on_flight: &mut VecDeque<IoT>,
                        id_map: &mut HashMap<u32, DiskNode<T>>,
                        qbuf: &QueryBuffer<T>|
         -> (usize, usize) {
            reader.poll_all(ctx);
            let mut n_in = 0;
            let mut n_out = 0;

            while let Some(io) = on_flight.front() {
                if !io.finished(&qbuf.reqs) {
                    break;
                }

                let io = on_flight.pop_front().unwrap();
                let node = self.node_from_page(
                    unsafe { &*io.read_req_idx as *const u8 as *const i8 },
                    io.loc,
                );
                id_map.insert(io.nbr.id, node);

                if io.nbr.distance <= retset[cur_list_size - 1].distance {
                    n_in += 1;
                } else {
                    n_out += 1;
                }

                self.idx_lock_table.unlock_idx(io.nbr.id);
            }

            (n_in, n_out)
        };

        // Closure: send best read requests
        let send_best_read_req = |n: usize,
                                  marker: &mut usize,
                                  on_flight: &mut VecDeque<IoT>,
                                  id_map: &HashMap<u32, DiskNode<T>>,
                                  qbuf: &mut QueryBuffer<T>,
                                  stats_ref: &mut Option<&mut QueryStats>|
         -> bool {
            let mut n_sent = 0;

            while *marker < cur_list_size && n_sent < n {
                while *marker < cur_list_size
                    && (!retset[*marker].flag || id_map.contains_key(&retset[*marker].id))
                {
                    retset[*marker].flag = false;
                    *marker += 1;
                }

                if *marker >= cur_list_size {
                    break;
                }

                if send_read_req(&mut retset[*marker], on_flight, qbuf, stats_ref) {
                    n_sent += 1;
                }
            }

            n_sent != 0
        };

        // Closure: calculate best node
        let calc_best_node =
            |marker: &mut usize, id_map: &HashMap<u32, DiskNode<T>>| -> usize {
                let mut nk = cur_list_size;

                // Process already-read nodes
                for m in *marker..cur_list_size {
                    if !retset[m].visited && id_map.contains_key(&retset[m].id) {
                        retset[m].flag = false;
                        retset[m].visited = true;

                        if let Some(node) = id_map.get(&retset[m].id) {
                            compute_exact_dists_and_push(node, retset[m].id);
                            compute_and_push_nbrs(
                                &mut node.clone(),
                                &mut nk,
                                &mut query_buf.visited,
                            );
                        }
                        break;
                    }
                }

                // Find first unvisited vector
                let mut first_unvisited_eager = cur_list_size;
                for i in 0..cur_list_size {
                    if !retset[i].visited
                        && retset[i].flag
                        && !id_map.contains_key(&retset[i].id)
                    {
                        first_unvisited_eager = i;
                        break;
                    }
                }

                first_unvisited_eager
            };

        // Closure: get first unvisited
        let get_first_unvisited = || -> Option<usize> {
            for i in 0..cur_list_size {
                if !retset[i].visited {
                    return Some(i);
                }
            }
            None
        };

        // Main search loop
        let mut marker = 0;
        let mut max_marker = 0;
        let mut cur_n_in = 0;
        let mut cur_tot = 0;

        send_best_read_req(
            cur_beam_width - on_flight_ios.len(),
            &mut marker,
            &mut on_flight_ios,
            &id_buf_map,
            &mut query_buf,
            &mut stats,
        );

        #[cfg(feature = "overlap_init")]
        {
            if mem_l != 0 {
                nbr_handler.initialize_query(query, &mut query_buf);
                nbr_handler.compute_dists(&mut query_buf, &mem_tags, mem_l);
                for i in 0..cur_list_size {
                    retset[i].distance = query_buf.aligned_dist_scratch[i];
                }
                retset[..cur_list_size].sort_unstable();
            }
        }

        while get_first_unvisited().is_some() {
            let (n_in, n_out) = poll_all(&mut on_flight_ios, &mut id_buf_map, &query_buf);

            if max_marker >= 5 && n_in + n_out > 0 {
                cur_n_in += n_in;
                cur_tot += n_in + n_out;

                // Tune beam width based on waste
                if (cur_tot - cur_n_in) as f64 / cur_tot as f64 <= WASTE_THRESHOLD {
                    cur_beam_width = (cur_beam_width + 1).max(4).min(beam_width);
                }

                // Record converge size for relaxed monotonicity
                if relaxed_monotonicity_l > 0 && converge_size < 0 {
                    converge_size = full_retset.len() as i64;
                }
            }

            // Check relaxed monotonicity termination
            if relaxed_monotonicity_l > 0
                && converge_size >= 0
                && full_retset.len() >= (converge_size as usize + relaxed_monotonicity_l)
            {
                break;
            }

            if on_flight_ios.len() < cur_beam_width {
                send_best_read_req(
                    1,
                    &mut marker,
                    &mut on_flight_ios,
                    &id_buf_map,
                    &mut query_buf,
                    &mut stats,
                );
            }

            marker = calc_best_node(&mut marker, &id_buf_map);
            max_marker = max_marker.max(marker);
        }

        // Drain remaining I/Os in relaxed monotonicity mode
        if relaxed_monotonicity_l > 0 {
            while !on_flight_ios.is_empty() {
                reader.poll_all(ctx);
                while let Some(io) = on_flight_ios.front() {
                    if !io.finished(&query_buf.reqs) {
                        break;
                    }
                    let io = on_flight_ios.pop_front().unwrap();
                    let node = self.node_from_page(
                        unsafe { &*io.read_req_idx as *const u8 as *const i8 },
                        io.loc,
                    );
                    id_buf_map.insert(io.nbr.id, node);
                    self.idx_lock_table.unlock_idx(io.nbr.id);
                }
            }

            // Process remaining unvisited nodes
            for i in 0..cur_list_size {
                if !retset[i].visited && id_buf_map.contains_key(&retset[i].id) {
                    retset[i].visited = true;
                    if let Some(node) = id_buf_map.get(&retset[i].id) {
                        compute_exact_dists_and_push(node, retset[i].id);
                    }
                }
            }
        }

        if let Some(s) = stats.as_mut() {
            s.cpu_us = n_computes as u64;
        }

        // Sort and deduplicate results
        full_retset.sort_unstable();

        let mut t = 0;
        for i in 0..full_retset.len() {
            if t >= k_search {
                break;
            }
            if i > 0 && full_retset[i].id == full_retset[i - 1].id {
                continue;
            }
            res_tags[t] = self.id2tag(full_retset[i].id);
            if let Some(dists) = distances.as_mut() {
                dists[t] = full_retset[i].distance;
            }
            t += 1;
        }

        t
    }

    // Helper methods
    fn id2loc(&self, _id: u32) -> u32 {
        // Implementation needed
        0
    }

    fn loc_sector_no(&self, _loc: u32) -> u32 {
        // Implementation needed
        0
    }

    fn u_loc_offset(&self, _loc: u32) -> u32 {
        // Implementation needed
        0
    }

    fn size_per_io(&self) -> usize {
        SECTOR_LEN
    }

    fn node_from_page(&self, _page: *const i8, _loc: u32) -> DiskNode<T> {
        // Implementation needed
        DiskNode {
            coords: vec![],
            nbrs: vec![],
            nnbrs: 0,
            labels: vec![],
        }
    }

    fn id2tag(&self, id: u32) -> TagT {
        TagT::from(id)
    }
}
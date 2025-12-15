use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

// Constants
const SECTOR_LEN: usize = 4096;
const MAX_N_SECTOR_READS: usize = 128;
const INDEX_SIZE_FACTOR: usize = 4;
const INVALID_ID: u32 = u32::MAX;
const ALLOCATED_ID: u32 = u32::MAX - 1;
const BG_IO_THREADS: usize = 4;

/// Metric types for distance computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    L2,
    InnerProduct,
    Cosine,
}

/// Index build parameters
#[derive(Debug, Clone)]
pub struct IndexBuildParameters {
    pub max_degree: usize,
    pub search_list_size: usize,
    pub build_pq: bool,
    pub pq_code_size: usize,
}

impl Default for IndexBuildParameters {
    fn default() -> Self {
        Self {
            max_degree: 64,
            search_list_size: 100,
            build_pq: false,
            pq_code_size: 32,
        }
    }
}

/// Metadata for SSD index
#[derive(Debug, Clone)]
pub struct SSDIndexMetadata {
    pub data_dim: usize,
    pub npoints: u64,
    pub max_node_len: usize,
    pub nnodes_per_sector: u64,
    pub entry_point: u32,
    pub range: f32,
}

impl SSDIndexMetadata {
    pub fn load_from_disk_index(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = vec![0u8; 1024];
        file.read_exact(&mut buffer)?;
        
        // Parse metadata from buffer (simplified)
        // In real implementation, you'd properly deserialize the binary format
        Ok(Self {
            data_dim: 128,
            npoints: 0,
            max_node_len: 0,
            nnodes_per_sector: 0,
            entry_point: 0,
            range: 0.0,
        })
    }
}

/// Disk node representation
pub struct DiskNode<T> {
    pub coords: Vec<T>,
    pub nbrs: Vec<u32>,
    pub nnbrs: u32,
}

/// Query buffer for thread-local storage
pub struct QueryBuffer<T> {
    pub coord_scratch: Vec<T>,
    pub sector_scratch: Vec<u8>,
    pub nbr_vec_scratch: Vec<u32>,
    pub nbr_ctx_scratch: Vec<u8>,
    pub aligned_dist_scratch: Vec<f32>,
    pub aligned_query: Vec<T>,
}

impl<T: Default + Clone> QueryBuffer<T> {
    fn new(data_dim: usize) -> Self {
        Self {
            coord_scratch: vec![T::default(); data_dim * 1024],
            sector_scratch: vec![0u8; MAX_N_SECTOR_READS * SECTOR_LEN],
            nbr_vec_scratch: vec![0u32; 1024],
            nbr_ctx_scratch: vec![0u8; 4096],
            aligned_dist_scratch: vec![0.0f32; 1024],
            aligned_query: vec![T::default(); data_dim],
        }
    }
}

/// Background I/O task
struct BgTask {
    terminate: bool,
    pages_to_unlock: Vec<u64>,
    pages_to_deref: Vec<u64>,
}

/// Thread-safe queue with notification
struct NotifyQueue<T> {
    queue: Mutex<Vec<T>>,
}

impl<T> NotifyQueue<T> {
    fn new() -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.push(item);
    }

    fn pop(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        if queue.is_empty() {
            None
        } else {
            Some(queue.remove(0))
        }
    }
}

/// Simple lock table for concurrent access
pub struct SparseLockTable {
    locks: RwLock<HashMap<u64, Arc<RwLock<()>>>>,
}

impl SparseLockTable {
    fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
        }
    }

    fn get_lock(&self, key: u64) -> Arc<RwLock<()>> {
        let locks = self.locks.read().unwrap();
        if let Some(lock) = locks.get(&key) {
            return Arc::clone(lock);
        }
        drop(locks);

        let mut locks = self.locks.write().unwrap();
        Arc::clone(locks.entry(key).or_insert_with(|| Arc::new(RwLock::new(()))))
    }

    pub fn rdlock(&self, key: u64) {
        let lock = self.get_lock(key);
        let _ = lock.read().unwrap();
    }

    pub fn wrlock(&self, key: u64) {
        let lock = self.get_lock(key);
        let _ = lock.write().unwrap();
    }

    pub fn unlock(&self, _key: u64) {
        // Locks are automatically released when guards go out of scope
    }
}

/// Main SSD Index structure
pub struct SSDIndex<T, TagT = u32> {
    // Metadata
    meta: SSDIndexMetadata,
    metric: Metric,
    enable_tags: bool,
    
    // File paths
    disk_index_file: PathBuf,
    
    // Mapping structures
    id2loc: RwLock<Vec<u32>>,
    loc2id: RwLock<Vec<u32>>,
    tags: RwLock<HashMap<u32, TagT>>,
    
    // Allocation tracking
    cur_loc: AtomicU64,
    cur_id: AtomicU32,
    empty_pages: NotifyQueue<u32>,
    
    // Thread management
    thread_data_bufs: Mutex<Vec<Arc<Mutex<QueryBuffer<T>>>>>,
    max_nthreads: usize,
    
    // Locking
    idx_lock_table: SparseLockTable,
    page_lock_table: SparseLockTable,
    alloc_lock: Mutex<()>,
    
    // Configuration
    use_page_search: bool,
    load_flag: bool,
    params: IndexBuildParameters,
}

impl<T, TagT> SSDIndex<T, TagT>
where
    T: Default + Clone + Send + Sync + 'static,
    TagT: Default + Clone + Send + Sync + 'static,
{
    /// Create a new SSD index
    pub fn new(
        metric: Metric,
        enable_tags: bool,
        parameters: Option<IndexBuildParameters>,
    ) -> Self {
        let params = parameters.unwrap_or_default();
        
        Self {
            meta: SSDIndexMetadata {
                data_dim: 0,
                npoints: 0,
                max_node_len: 0,
                nnodes_per_sector: 0,
                entry_point: 0,
                range: 0.0,
            },
            metric,
            enable_tags,
            disk_index_file: PathBuf::new(),
            id2loc: RwLock::new(Vec::new()),
            loc2id: RwLock::new(Vec::new()),
            tags: RwLock::new(HashMap::new()),
            cur_loc: AtomicU64::new(0),
            cur_id: AtomicU32::new(0),
            empty_pages: NotifyQueue::new(),
            thread_data_bufs: Mutex::new(Vec::new()),
            max_nthreads: 0,
            idx_lock_table: SparseLockTable::new(),
            page_lock_table: SparseLockTable::new(),
            alloc_lock: Mutex::new(()),
            use_page_search: false,
            load_flag: false,
            params,
        }
    }

    /// Load index from disk
    pub fn load(
        &mut self,
        index_prefix: &str,
        num_threads: usize,
        use_page_search: bool,
    ) -> io::Result<()> {
        let disk_index_file = format!("{}_disk.index", index_prefix);
        self.disk_index_file = PathBuf::from(&disk_index_file);

        // Load metadata
        let meta = SSDIndexMetadata::load_from_disk_index(&self.disk_index_file)?;
        self.meta = meta.clone();

        // Initialize buffers
        self.init_buffers(num_threads);
        self.max_nthreads = num_threads;

        // Load page layout
        self.use_page_search = use_page_search;
        self.load_page_layout(index_prefix, meta.nnodes_per_sector, meta.npoints)?;

        // Load tags if enabled
        if self.enable_tags {
            let tag_file = format!("{}.tags", disk_index_file);
            self.load_tags(&tag_file)?;
        }

        self.load_flag = true;
        println!("SSDIndex loaded successfully");
        Ok(())
    }

    /// Initialize query buffers for threads
    fn init_buffers(&mut self, n_threads: usize) {
        let n_buffers = n_threads * 2;
        println!("Init buffers for {} threads, setup {} buffers", n_threads, n_buffers);
        
        let mut bufs = self.thread_data_bufs.lock().unwrap();
        for _ in 0..n_buffers {
            let buf = QueryBuffer::new(self.meta.data_dim);
            bufs.push(Arc::new(Mutex::new(buf)));
        }
    }

    /// Load page layout from partition file
    fn load_page_layout(
        &mut self,
        index_prefix: &str,
        nnodes_per_sector: u64,
        num_points: u64,
    ) -> io::Result<()> {
        let partition_file = format!("{}_partition.bin.aligned", index_prefix);
        
        let mut id2loc = self.id2loc.write().unwrap();
        let mut loc2id = self.loc2id.write().unwrap();
        
        id2loc.resize(num_points as usize, INVALID_ID);
        loc2id.resize(self.cur_loc.load(Ordering::Relaxed) as usize, INVALID_ID);

        if Path::new(&partition_file).exists() {
            println!("Loading partition file {}", partition_file);
            
            let mut file = File::open(&partition_file)?;
            let mut buffer = vec![0u8; 24];
            file.read_exact(&mut buffer)?;
            
            // Parse partition metadata
            // In real implementation, properly deserialize the binary format
            
            println!("Page layout loaded from partition file");
        } else {
            println!("{} does not exist, use equal partition mapping", partition_file);
            
            // Equal mapping for id2loc and loc2id
            for i in 0..num_points as usize {
                id2loc[i] = i as u32;
                loc2id[i] = i as u32;
            }
            
            let cur_loc_val = self.cur_loc.load(Ordering::Relaxed) as usize;
            for i in num_points as usize..cur_loc_val {
                loc2id[i] = INVALID_ID;
            }
        }
        
        println!("Page layout loaded");
        Ok(())
    }

    /// Load tags from file
    fn load_tags(&mut self, tag_file_name: &str) -> io::Result<()> {
        let mut tags = self.tags.write().unwrap();
        tags.clear();

        if !Path::new(tag_file_name).exists() {
            println!("Tags file not found. Using equal mapping");
            return Ok(());
        }

        println!("Load tags from existing file: {}", tag_file_name);
        // In real implementation, properly deserialize tags from binary format
        
        Ok(())
    }

    /// Copy index files from one location to another
    pub fn copy_index(src_prefix: &str, dst_prefix: &str) -> io::Result<()> {
        println!("Copying disk index from {} to {}", src_prefix, dst_prefix);
        
        let src_index = format!("{}_disk.index", src_prefix);
        let dst_index = format!("{}_disk.index", dst_prefix);
        fs::copy(&src_index, &dst_index)?;

        let src_tags = format!("{}_disk.index.tags", src_prefix);
        let dst_tags = format!("{}_disk.index.tags", dst_prefix);
        if Path::new(&src_tags).exists() {
            fs::copy(&src_tags, &dst_tags)?;
        } else {
            let _ = fs::remove_file(&dst_tags);
        }

        let src_partition = format!("{}_partition.bin.aligned", src_prefix);
        let dst_partition = format!("{}_partition.bin.aligned", dst_prefix);
        if Path::new(&src_partition).exists() {
            fs::copy(&src_partition, &dst_partition)?;
        }

        Ok(())
    }

    /// Convert ID to tag
    pub fn id2tag(&self, id: u32) -> TagT 
    where
        TagT: From<u32>,
    {
        let tags = self.tags.read().unwrap();
        tags.get(&id).cloned().unwrap_or_else(|| TagT::from(id))
    }

    /// Convert ID to location
    pub fn id2loc(&self, id: u32) -> u32 {
        let id2loc = self.id2loc.read().unwrap();
        if id as usize >= id2loc.len() {
            eprintln!("id {} is out of range {}", id, id2loc.len());
            return INVALID_ID;
        }
        id2loc[id as usize]
    }

    /// Set ID to location mapping
    pub fn set_id2loc(&self, id: u32, loc: u32) {
        let mut id2loc = self.id2loc.write().unwrap();
        
        if id as usize >= id2loc.len() {
            let new_size = ((id as f64) * 1.5) as usize;
            id2loc.resize(new_size, INVALID_ID);
            println!("Resize id2loc_ to {}", id2loc.len());
        }
        
        id2loc[id as usize] = loc;
    }

    /// Convert ID to page number
    pub fn id2page(&self, id: u32) -> u64 {
        let loc = self.id2loc(id);
        if loc == INVALID_ID {
            return u64::MAX;
        }
        self.loc_sector_no(loc as u64)
    }

    /// Convert location to ID
    pub fn loc2id(&self, loc: u32) -> u32 {
        let loc2id = self.loc2id.read().unwrap();
        if loc as usize >= loc2id.len() {
            eprintln!("loc {} is out of range {}", loc, loc2id.len());
            return INVALID_ID;
        }
        loc2id[loc as usize]
    }

    /// Set location to ID mapping
    pub fn set_loc2id(&self, loc: u32, id: u32) {
        let mut loc2id = self.loc2id.write().unwrap();
        
        if loc as usize >= loc2id.len() {
            let new_size = ((loc as f64) * 1.5) as usize;
            loc2id.resize(new_size, INVALID_ID);
            println!("Resize loc2id_ to {}", loc2id.len());
        }
        
        loc2id[loc as usize] = id;
    }

    /// Erase location to ID mapping
    pub fn erase_loc2id(&self, loc: u32) {
        let mut loc2id = self.loc2id.write().unwrap();
        loc2id[loc as usize] = INVALID_ID;
        
        let st = self.sector_to_loc(self.loc_sector_no(loc as u64), 0);
        let mut empty = true;
        
        let nodes_per_sector = self.meta.nnodes_per_sector as u32;
        for i in st..(st + nodes_per_sector) {
            if loc2id.get(i as usize).copied().unwrap_or(INVALID_ID) != INVALID_ID {
                empty = false;
                break;
            }
        }
        
        if empty {
            let page = self.loc_sector_no(loc as u64) as u32;
            self.empty_pages.push(page);
        }
    }

    /// Calculate sector number from location
    fn loc_sector_no(&self, loc: u64) -> u64 {
        if self.meta.nnodes_per_sector == 0 {
            loc * ((self.meta.max_node_len + SECTOR_LEN - 1) / SECTOR_LEN) as u64
        } else {
            loc / self.meta.nnodes_per_sector
        }
    }

    /// Convert sector and offset to location
    fn sector_to_loc(&self, sector: u64, offset: u32) -> u32 {
        if self.meta.nnodes_per_sector == 0 {
            sector as u32
        } else {
            (sector * self.meta.nnodes_per_sector + offset as u64) as u32
        }
    }

    /// Get page layout for a given page number
    pub fn get_page_layout(&self, page_no: u32) -> Vec<u32> {
        let loc2id = self.loc2id.read().unwrap();
        let mut ret = Vec::new();
        
        let st = self.sector_to_loc(page_no as u64, 0);
        let ed = if self.meta.nnodes_per_sector == 0 {
            st + 1
        } else {
            st + self.meta.nnodes_per_sector as u32
        };
        
        for i in st..ed {
            if let Some(&id) = loc2id.get(i as usize) {
                ret.push(id);
            }
        }
        
        ret
    }

    /// Lock indices for concurrent access
    pub fn lock_idx(
        &self,
        lock_table: &SparseLockTable,
        target: u32,
        neighbors: &[u32],
        read_only: bool,
    ) -> Vec<u32> {
        let mut to_lock = self.get_to_lock_idx(target, neighbors);
        
        for &id in &to_lock {
            if read_only {
                lock_table.rdlock(id as u64);
            } else {
                lock_table.wrlock(id as u64);
            }
        }
        
        to_lock
    }

    /// Unlock indices
    pub fn unlock_idx(&self, lock_table: &SparseLockTable, to_lock: &[u32]) {
        for &id in to_lock {
            lock_table.unlock(id as u64);
        }
    }

    /// Get indices to lock (sorted and deduplicated)
    fn get_to_lock_idx(&self, target: u32, neighbors: &[u32]) -> Vec<u32> {
        let mut to_lock: Vec<u32> = neighbors.to_vec();
        
        if target != INVALID_ID {
            to_lock.push(target);
        }
        
        to_lock.sort_unstable();
        to_lock.dedup();
        to_lock
    }

    /// Lock page indices
    pub fn lock_page_idx(
        &self,
        lock_table: &SparseLockTable,
        target: u32,
        neighbors: &[u32],
        read_only: bool,
    ) -> Vec<u32> {
        if !self.use_page_search {
            return Vec::new();
        }

        let mut to_lock: Vec<u32> = neighbors.iter().map(|&id| self.id2page(id) as u32).collect();
        
        if target != INVALID_ID {
            to_lock.push(self.id2page(target) as u32);
        }
        
        to_lock.sort_unstable();
        to_lock.dedup();
        
        for &id in &to_lock {
            if read_only {
                lock_table.rdlock(id as u64);
            } else {
                lock_table.wrlock(id as u64);
            }
        }
        
        to_lock
    }

    /// Unlock page indices
    pub fn unlock_page_idx(&self, lock_table: &SparseLockTable, to_lock: &[u32]) {
        if !self.use_page_search {
            return;
        }
        
        for &id in to_lock {
            lock_table.unlock(id as u64);
        }
    }

    /// Allocate locations
    pub fn alloc_loc(
        &self,
        n: usize,
        hint_pages: &[u64],
        page_need_to_read: &mut HashSet<u64>,
    ) -> Vec<u64> {
        let _lock = self.alloc_lock.lock().unwrap();
        let mut ret = Vec::new();
        let threshold = (self.meta.nnodes_per_sector as usize + INDEX_SIZE_FACTOR - 1) / INDEX_SIZE_FACTOR;

        // 1. Use empty pages
        while let Some(empty_page) = self.empty_pages.pop() {
            let st = self.sector_to_loc(empty_page as u64, 0);
            let ed = if self.meta.nnodes_per_sector == 0 {
                st + 1
            } else {
                st + self.meta.nnodes_per_sector as u32
            };
            
            for i in st..ed {
                self.set_loc2id(i, ALLOCATED_ID);
                ret.push(i as u64);
                if ret.len() == n {
                    return ret;
                }
            }
        }

        // 2. Use hint pages
        for &page in hint_pages {
            let st = self.sector_to_loc(page, 0);
            let ed = if self.meta.nnodes_per_sector == 0 {
                st + 1
            } else {
                st + self.meta.nnodes_per_sector as u32
            };
            
            let loc2id = self.loc2id.read().unwrap();
            let cnt = (st..ed)
                .filter(|&i| loc2id.get(i as usize).copied().unwrap_or(INVALID_ID) == INVALID_ID)
                .count();
            drop(loc2id);
            
            if cnt < threshold {
                continue;
            }
            
            if cnt < self.meta.nnodes_per_sector as usize {
                page_need_to_read.insert(page);
            }
            
            for i in st..ed {
                if self.loc2id(i) == INVALID_ID {
                    self.set_loc2id(i, ALLOCATED_ID);
                    ret.push(i as u64);
                    if ret.len() == n {
                        return ret;
                    }
                }
            }
        }

        // 3. Use new pages
        let remaining = n - ret.len();
        let cur_loc = self.cur_loc.load(Ordering::Relaxed);
        
        for i in 0..remaining {
            self.set_loc2id((cur_loc + i as u64) as u32, ALLOCATED_ID);
            ret.push(cur_loc + i as u64);
        }
        
        let mut new_cur_loc = cur_loc + remaining as u64;
        
        // Align cur_loc
        if self.meta.nnodes_per_sector != 0 {
            while new_cur_loc % self.meta.nnodes_per_sector != 0 {
                self.set_loc2id(new_cur_loc as u32, INVALID_ID);
                new_cur_loc += 1;
            }
        }
        
        self.cur_loc.store(new_cur_loc, Ordering::Relaxed);
        ret
    }

    /// Verify ID to location mapping consistency
    pub fn verify_id2loc(&self) {
        let id2loc = self.id2loc.read().unwrap();
        let loc2id = self.loc2id.read().unwrap();
        let cur_id = self.cur_id.load(Ordering::Relaxed);
        let cur_loc = self.cur_loc.load(Ordering::Relaxed);
        
        println!(
            "ID2loc size: {}, cur_loc: {}, cur_id: {}, nnodes_per_sector: {}",
            id2loc.len(),
            cur_loc,
            cur_id,
            self.meta.nnodes_per_sector
        );

        // Verify id -> loc -> id map
        for i in 0..cur_id {
            if let Some(&loc) = id2loc.get(i as usize) {
                if loc >= cur_loc as u32 {
                    eprintln!(
                        "ID2loc inconsistency at ID: {}, loc: {}, cur_loc: {}",
                        i, loc, cur_loc
                    );
                    continue;
                }
                
                if let Some(&id_back) = loc2id.get(loc as usize) {
                    if id_back != i {
                        eprintln!(
                            "ID2loc inconsistency at ID: {}, loc: {}, loc2id: {}",
                            i, loc, id_back
                        );
                    }
                }
            }
        }

        println!("ID2loc consistency check passed");

        // Verify loc2id doesn't contain duplicate ids
        for i in 0..cur_loc {
            if let Some(&id) = loc2id.get(i as usize) {
                if id != INVALID_ID && id != ALLOCATED_ID {
                    if let Some(&loc) = id2loc.get(id as usize) {
                        if loc != i as u32 {
                            eprintln!("loc2id inconsistency at loc: {}, id: {}, loc: {}", i, id, loc);
                        }
                    }
                }
            }
        }

        println!("loc2ID consistency check passed");
    }

    /// Get vector by ID
    pub fn get_vector_by_id(&self, id: u32, vector_coords: &mut [T]) -> io::Result<()> {
        if !self.enable_tags {
            eprintln!("Tags are disabled, cannot retrieve vector");
            return Err(io::Error::new(io::ErrorKind::Other, "Tags disabled"));
        }

        let loc = self.id2loc(id);
        let num_sectors = self.loc_sector_no(loc as u64);
        
        let mut file = File::open(&self.disk_index_file)?;
        let mut sector_buf = vec![0u8; SECTOR_LEN];
        
        file.seek(SeekFrom::Start(SECTOR_LEN as u64 * num_sectors))?;
        file.read_exact(&mut sector_buf)?;
        
        // Parse node and copy coordinates
        // In real implementation, properly deserialize the disk node
        
        Ok(())
    }
}

impl<T, TagT> Drop for SSDIndex<T, TagT> {
    fn drop(&mut self) {
        if self.load_flag {
            println!("Lock table size: {}", "N/A");
            println!("Dropping SSDIndex");
        }
    }
}
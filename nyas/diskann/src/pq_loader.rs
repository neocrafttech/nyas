use std::collections::HashMap;
use std::io;

use memmap2::Mmap;
use rkyv::{Archived, Deserialize, rancor};
use system::metric::{Distance, MetricType};
use system::vector_data::VectorData;

use crate::product_quantizer::ProductQuantizer;

pub struct MmappedCodes {
    _mmap: Mmap,
    map_ptr: *const Archived<HashMap<u32, Vec<u8>>>,
}

// SAFETY: Mmap ensures the memory is valid for the life of the struct.
// Moving the struct does not move the mmapped memory.
unsafe impl Send for MmappedCodes {}
unsafe impl Sync for MmappedCodes {}

impl MmappedCodes {
    pub fn new(mmap: Mmap) -> Self {
        let map_ptr =
            unsafe { rkyv::access_unchecked::<Archived<HashMap<u32, Vec<u8>>>>(&mmap) as *const _ };
        Self { _mmap: mmap, map_ptr }
    }

    fn archived_map(&self) -> &Archived<HashMap<u32, Vec<u8>>> {
        unsafe { &*self.map_ptr }
    }

    pub fn get(&self, key: u32) -> Option<&[u8]> {
        let key_le = rkyv::rend::u32_le::from_native(key);
        self.archived_map().get(&key_le).map(|v| v.as_slice())
    }
}

pub async fn load_codes(codes_file_path: &str) -> io::Result<Option<MmappedCodes>> {
    if !std::path::Path::new(codes_file_path).exists() {
        return Ok(None);
    }

    let file = tokio::fs::File::open(codes_file_path).await?;
    let std_file = file.into_std().await;
    let mmap = unsafe { Mmap::map(&std_file)? };

    Ok(Some(MmappedCodes::new(mmap)))
}

pub struct MmappedPQ {
    _mmap: Mmap,
    pq_ptr: *const Archived<ProductQuantizer>,
}

// SAFETY: Mmap ensures the memory is valid for the life of the struct.
// Moving the struct does not move the mmapped memory.
unsafe impl Send for MmappedPQ {}
unsafe impl Sync for MmappedPQ {}

impl MmappedPQ {
    pub fn new(mmap: Mmap) -> Self {
        let pq_ptr =
            unsafe { rkyv::access_unchecked::<Archived<ProductQuantizer>>(&mmap) as *const _ };
        Self { _mmap: mmap, pq_ptr }
    }

    pub fn as_ref(&self) -> &Archived<ProductQuantizer> {
        unsafe { &*self.pq_ptr }
    }

    pub fn compute_lut(&self, query: &VectorData) -> Option<Vec<Vec<f64>>> {
        let pq = self.as_ref();
        let mut lut = vec![
            vec![0.0; pq.centroids_per_subspace.to_native() as usize];
            pq.num_subspaces.to_native() as usize
        ];

        let mut binding = ();
        let err = rancor::Strategy::<_, rancor::Panic>::wrap(&mut binding);

        for (sub_idx, item) in lut.iter_mut().enumerate() {
            let start = sub_idx * pq.subspace_dim.to_native() as usize;
            let query_sub_vec = query.slice(start, start + pq.subspace_dim.to_native() as usize);
            let subspace_codebook = &pq.codebook[sub_idx];

            for (k_idx, centroid) in subspace_codebook.iter().enumerate() {
                let vector_data: VectorData = centroid.deserialize(err).ok()?;
                item[k_idx] = query_sub_vec.distance(&vector_data, MetricType::L2Squared);
            }
        }
        Some(lut)
    }
}

pub async fn load_pq_data(pq_file_path: &str) -> io::Result<Option<MmappedPQ>> {
    if !std::path::Path::new(pq_file_path).exists() {
        return Ok(None);
    }

    let file = tokio::fs::File::open(pq_file_path).await?;
    let std_file = file.into_std().await;
    let mmap = unsafe { Mmap::map(&std_file)? };

    Ok(Some(MmappedPQ::new(mmap)))
}

pub fn estimate_distance(lut: &[Vec<f64>], codes: &[u8]) -> f64 {
    let mut dist_approx = 0.0;

    for (sub_idx, centroid_idx) in codes.iter().enumerate() {
        dist_approx += lut[sub_idx][*centroid_idx as usize];
    }
    dist_approx
}

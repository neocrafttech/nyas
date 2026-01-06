use std::collections::HashSet;
use std::path::Path;

use diskann::index_view::IndexView;
use system::vector_data::VectorData;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, BufReader};

#[derive(Debug)]
pub struct SiftDataset {
    pub dimension: u32,
    pub vectors: Vec<VectorData>,
}

impl SiftDataset {
    pub async fn from_fvecs<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path).await?;
        let mut reader = BufReader::new(file);

        let mut vectors = Vec::new();
        let mut dimension = 0u32;
        let mut buffer = [0u8; 4];

        if reader.read_exact(&mut buffer).await.is_ok() {
            dimension = u32::from_le_bytes(buffer);

            // Read first vector
            let mut vec = vec![0f32; dimension as usize];
            for item in vec.iter_mut().take(dimension as usize) {
                reader.read_exact(&mut buffer).await?;
                *item = f32::from_le_bytes(buffer);
            }
            let vec_data = VectorData::from_f32(vec);
            vectors.push(vec_data);
        }

        loop {
            match reader.read_exact(&mut buffer).await {
                Ok(_) => {
                    let dim = u32::from_le_bytes(buffer);
                    if dim != dimension {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Inconsistent vector dimensions",
                        ));
                    }

                    let mut vec = vec![0f32; dimension as usize];
                    for item in vec.iter_mut().take(dimension as usize) {
                        reader.read_exact(&mut buffer).await?;
                        *item = f32::from_le_bytes(buffer);
                    }
                    let vec_data = VectorData::from_f32(vec);
                    vectors.push(vec_data);
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        Ok(SiftDataset { dimension, vectors })
    }

    #[allow(dead_code)]
    pub async fn from_bvecs<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path).await?;
        let mut reader = BufReader::new(file);

        let mut vectors = Vec::new();
        let mut dimension = 0u32;
        let mut buffer = [0u8; 4];

        if reader.read_exact(&mut buffer).await.is_ok() {
            dimension = u32::from_le_bytes(buffer);

            let mut vec = vec![0u8; dimension as usize];
            reader.read_exact(&mut vec).await?;
            let vec = vec.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            let vec_data = VectorData::from_f32(vec);
            vectors.push(vec_data);
        }

        loop {
            match reader.read_exact(&mut buffer).await {
                Ok(_) => {
                    let dim = u32::from_le_bytes(buffer);
                    if dim != dimension {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Inconsistent vector dimensions",
                        ));
                    }

                    let mut vec = vec![0u8; dimension as usize];
                    reader.read_exact(&mut vec).await?;
                    let vec = vec.iter().map(|&x| x as f32).collect::<Vec<f32>>();
                    let vec_data = VectorData::from_f32(vec);
                    vectors.push(vec_data);
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        Ok(SiftDataset { dimension, vectors })
    }

    pub async fn from_ivecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u32>>> {
        let file = File::open(path).await?;
        let mut reader = BufReader::new(file);

        let mut results = Vec::new();
        let mut buffer = [0u8; 4];

        loop {
            match reader.read_exact(&mut buffer).await {
                Ok(_) => {
                    let k = u32::from_le_bytes(buffer);
                    let mut neighbors = vec![0u32; k as usize];

                    for item in neighbors.iter_mut().take(k as usize) {
                        reader.read_exact(&mut buffer).await?;
                        *item = u32::from_le_bytes(buffer);
                    }
                    results.push(neighbors);
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }

        Ok(results)
    }
}

pub fn compute_recall(results: &[Vec<u32>], ground_truth: &[Vec<u32>], k: usize) -> f64 {
    assert_eq!(results.len(), ground_truth.len(), "Results and groundtruth must have same length");

    let num_queries = results.len();
    let mut total_matches = 0;

    for (result, truth) in results.iter().zip(ground_truth.iter()) {
        let truth_set: HashSet<u32> = truth.iter().take(k).copied().collect();

        let matches = result.iter().take(k).filter(|&&idx| truth_set.contains(&idx)).count();
        println!("Matches: {}", matches);
        total_matches += matches;
    }

    total_matches as f64 / (num_queries * k) as f64
}

pub async fn compute_recall_at_k(
    index_view: &IndexView, query: &SiftDataset, ground_truth: &[Vec<u32>], k: usize,
) {
    let mut results = Vec::new();
    for q in query.vectors.iter() {
        let res = index_view.search(q, k, 128).await;
        results.push(res);
    }

    let recall = compute_recall(&results, ground_truth, k);
    println!("Recall  for k: {}: {:?}", k, recall);
}

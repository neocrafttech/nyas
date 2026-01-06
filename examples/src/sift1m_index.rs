use std::path::Path;
use std::time::{Duration, SystemTime};

use cpu_time::ProcessTime;
use diskann::index_view::IndexView;
use system::vector_point::VectorPoint;
use tokio::io;

use crate::index_utils::SiftDataset;
mod index_utils;

#[tokio::main]
async fn main() -> io::Result<()> {
    let base_folder = "examples/data/sift";
    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    let (base, query, ground_truth) = tokio::try_join!(
        SiftDataset::from_fvecs(format!("{}/sift_base.fvecs", base_folder)),
        SiftDataset::from_fvecs(format!("{}/sift_query.fvecs", base_folder)),
        SiftDataset::from_ivecs(format!("{}/sift_groundtruth.ivecs", base_folder))
    )?;

    println!("Base dataset: {} vectors of dimension {}", base.vectors.len(), base.dimension);
    println!("Query dataset: {} vectors of dimension {}", query.vectors.len(), query.dimension);
    println!("Ground truth: {} queries", ground_truth.len());

    let index_name = "sift1m";
    let index_view = IndexView::new(index_name).await.expect("Failed to create IndexView");

    let path = Path::new(index_name);

    if !path.exists() {
        for (index, vector) in base.vectors.iter().enumerate() {
            index_view.insert(&VectorPoint::new(index as u32, vector.clone())).await.unwrap();
        }
    }

    let index_cpu_time = start_cpu.elapsed();
    let index_wall_time = start_wall.elapsed().unwrap();
    println!("Indexing time: CPU {:?}, Wall {:?}", index_cpu_time, index_wall_time);

    for k in [1, 10, 100] {
        index_utils::compute_recall_at_k(&index_view, &query, &ground_truth, k).await;
    }

    let cpu_time: Duration = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();

    println!("CPU time: {:?}", cpu_time);
    println!("Wall time: {:?}", wall_time);

    Ok(())
}

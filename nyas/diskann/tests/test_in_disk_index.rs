use diskann::in_disk_index::InDiskIndex;
use std::sync::Arc;
use storage::metric::MetricType;
use storage::vector_data::VectorData;
use storage::vector_point::VectorPoint;

#[test]
fn test_in_disk_index() {
    let system = InDiskIndex::new(32, 1.2, 50, 20, MetricType::L2);

    for i in 0..50 {
        let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32]);
        system.insert(&VectorPoint::new(i, vector)).unwrap();
    }

    system.delete(Arc::new(5));
    system.delete(Arc::new(10));

    let query = VectorData::from_f32(vec![25.0, 50.0]);
    let results = system.search(&query, 5, 20);

    assert!(!results.contains(&Arc::new(5)));
    assert!(!results.contains(&Arc::new(10)));

    println!("Search results: {:?}", results);
}

use diskann::index_view::IndexView;
use system::vector_data::VectorData;
use system::vector_point::VectorPoint;

#[test]
fn test_index_view() {
    let index_view = IndexView::new().expect("Failed to create IndexView");

    for i in 0..50 {
        let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32]);
        index_view.insert(&VectorPoint::new(i, vector)).unwrap();
    }

    index_view.delete(5);
    index_view.delete(10);

    let query = VectorData::from_f32(vec![25.0, 50.0]);
    let results = index_view.search(&query, 5, 20);

    assert!(!results.contains(&5));
    assert!(!results.contains(&10));

    println!("Search results: {:?}", results);
}

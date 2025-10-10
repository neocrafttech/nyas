use diskann::index::DiskANNIndex;
use diskann::index_builder::build_index;
use rand::random;
use tokio::fs;

#[tokio::test]
async fn test_search_vector() {
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..16).map(|_| random::<f32>()).collect::<Vec<f32>>())
        .collect();
    let vectors: Vec<&Vec<f32>> = vectors.iter().collect();

    let vectors_path = "test_vectors.bin";
    let graph_path = "test_graph.bin";

    build_index(&vectors, 8, vectors_path, graph_path)
        .await
        .expect("Failed to build index");

    assert!(
        fs::metadata(vectors_path).await.is_ok(),
        "vectors.bin not created"
    );
    assert!(
        fs::metadata(graph_path).await.is_ok(),
        "graph.bin not created"
    );

    let index = DiskANNIndex::load(vectors_path, graph_path, 16, 8)
        .await
        .expect("Failed to load index");

    let top_k = 5;
    let results = (&index).search(&vectors[0], top_k, 10).await;

    assert!(!results.is_empty(), "Search returned no results");
    assert!(results.len() <= top_k, "Returned more than top_k results");

    fs::remove_file(vectors_path).await.unwrap();
    fs::remove_file(graph_path).await.unwrap();

    assert!(!results.is_empty(), "Search returned no results");
}

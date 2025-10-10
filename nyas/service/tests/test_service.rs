use service::VectorService;
use service::vector::InsertVectorRequest;
use service::vector::SearchVectorRequest;
use service::vector::vector_db_server::VectorDb;
use tonic::Request;

#[tokio::test]
async fn test_insert_vector() {
    let service = VectorService::default();

    let req = InsertVectorRequest {
        id: "vec1".to_string(),
        vector: vec![0.1, 0.2, 0.3],
    };

    let response = service.insert_vector(Request::new(req)).await.unwrap();
    let res_inner = response.into_inner();

    assert!(res_inner.success, "Insert should succeed");
}

#[tokio::test]
async fn test_search_vector() {
    let service = VectorService::default();

    let req = InsertVectorRequest {
        id: "vec1".to_string(),
        vector: vec![0.1, 0.2, 0.3],
    };
    service.insert_vector(Request::new(req)).await.unwrap();

    let req = SearchVectorRequest {
        vector: vec![0.1, 0.2, 0.3],
        top_k: 1,
    };

    let response = service.search_vector(Request::new(req)).await.unwrap();
    let res_inner = response.into_inner();

    assert_eq!(res_inner.ids.len(), 1, "Should return one id");
    assert_eq!(res_inner.distances.len(), 1, "Should return one distance");
    assert_eq!(res_inner.ids[0], "vec1");
    assert_eq!(res_inner.distances[0], 0.0);
}

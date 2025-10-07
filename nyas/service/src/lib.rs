use crate::vector::vector_db_server::VectorDb;
use crate::vector::{
    InsertVectorRequest, InsertVectorResponse, SearchVectorRequest, SearchVectorResponse,
};
use std::sync::Arc;
use storage::vector_db::VectorDB;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

pub mod vector {
    tonic::include_proto!("vector");
}

pub struct VectorService {
    pub db: Arc<Mutex<VectorDB>>,
}

impl Default for VectorService {
    fn default() -> Self {
        Self {
            db: Arc::new(Mutex::new(VectorDB::new(128))),
        }
    }
}

#[tonic::async_trait]
impl VectorDb for VectorService {
    async fn insert_vector(
        &self,
        request: Request<InsertVectorRequest>,
    ) -> Result<Response<InsertVectorResponse>, Status> {
        let req = request.into_inner();
        let mut db = self.db.lock().await;
        db.insert(req.id, req.vector);
        Ok(Response::new(InsertVectorResponse { success: true }))
    }

    async fn search_vector(
        &self,
        request: Request<SearchVectorRequest>,
    ) -> Result<Response<SearchVectorResponse>, Status> {
        let req = request.into_inner();
        let db = self.db.lock().await;
        let results = db.search(&req.vector, req.top_k as usize);
        let (ids, distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        Ok(Response::new(SearchVectorResponse { ids, distances }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}

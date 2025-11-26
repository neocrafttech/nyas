use std::sync::Arc;

use system::vector_db::VectorDB;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use crate::vector::vector_db_server::VectorDb;
use crate::vector::{
    BuildIndexRequest, BuildIndexResponse, IndexType, InsertVectorRequest, InsertVectorResponse,
    SearchVectorRequest, SearchVectorResponse,
};

pub mod vector {
    tonic::include_proto!("vector");
}

pub struct VectorService {
    pub db: Arc<Mutex<VectorDB>>,
}

impl Default for VectorService {
    fn default() -> Self {
        Self { db: Arc::new(Mutex::new(VectorDB::new(128))) }
    }
}

impl BuildIndexRequest {
    pub fn index_type_enum(&self) -> IndexType {
        IndexType::try_from(self.index_type).unwrap_or(IndexType::Unspecified)
    }
}
#[tonic::async_trait]
impl VectorDb for VectorService {
    async fn insert_vector(
        &self, request: Request<InsertVectorRequest>,
    ) -> Result<Response<InsertVectorResponse>, Status> {
        let req = request.into_inner();
        let mut db = self.db.lock().await;
        db.insert(req.id, req.vector);
        Ok(Response::new(InsertVectorResponse { success: true }))
    }

    async fn search_vector(
        &self, request: Request<SearchVectorRequest>,
    ) -> Result<Response<SearchVectorResponse>, Status> {
        let req = request.into_inner();
        let db = self.db.lock().await;
        let results = db.search(&req.vector, req.top_k as usize);
        let (ids, distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        Ok(Response::new(SearchVectorResponse { ids, distances }))
    }

    async fn build_index(
        &self, request: Request<BuildIndexRequest>,
    ) -> Result<Response<BuildIndexResponse>, Status> {
        let req = request.into_inner();
        let mut db = self.db.lock().await;
        match req.index_type_enum() {
            IndexType::Flat => {
                unimplemented!()
            }
            IndexType::Hnsw => {
                unimplemented!()
            }
            IndexType::Diskann => match db.build_index().await {
                Ok(_) => Ok(Response::new(BuildIndexResponse { success: true })),
                Err(e) => Err(Status::internal(e.to_string())),
            },
            _ => Err(Status::invalid_argument("Invalid index type")),
        }
    }
}

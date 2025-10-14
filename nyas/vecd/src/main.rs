use mimalloc::MiMalloc;
use service::VectorService;
use service::vector;
use std::sync::Arc;
use storage::vector_db::VectorDB;
use tokio::sync::Mutex;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let db = Arc::new(Mutex::new(VectorDB::new(128))); // dimension 128
    let svc = VectorService { db: db.clone() };

    let addr = "127.0.0.1:50051".parse()?;
    println!("VectorDB gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(vector::vector_db_server::VectorDbServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}

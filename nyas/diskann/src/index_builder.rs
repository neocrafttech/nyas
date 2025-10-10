use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};

pub type NodeId = u32;

pub async fn build_index(
    vectors: &Vec<&Vec<f32>>,
    max_neighbors: usize,
    vectors_path: &str,
    graph_path: &str,
) -> std::io::Result<()> {
    let vec_file = File::create(vectors_path).await?;
    let mut vec_writer = BufWriter::new(vec_file);

    for vec in vectors.iter() {
        for &f in vec.iter() {
            vec_writer.write_all(&f.to_le_bytes()).await?;
        }
    }
    vec_writer.flush().await?;

    let graph_file = File::create(graph_path).await?;
    let mut graph_writer = BufWriter::new(graph_file);

    for (i, _) in vectors.iter().enumerate() {
        let mut neighbors: Vec<NodeId> = (0..vectors.len() as u32)
            .filter(|&x| x != i as u32)
            .collect();
        neighbors.truncate(max_neighbors);

        for &nid in &neighbors {
            graph_writer.write_all(&nid.to_le_bytes()).await?;
        }
    }
    graph_writer.flush().await?;

    Ok(())
}

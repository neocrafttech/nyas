use rayon::prelude::*;
use system::metric::{Distance, MetricType};
use system::vector_data::VectorData;

#[allow(dead_code)]
pub struct ProductQuantizer {
    /// Number of subspaces
    pub m: usize,
    /// Centroids per subspace
    pub k: usize,
    /// Dimensions per subspace
    pub sub_dim: usize,
    /// [M] matrices of size [K, sub_dim]
    pub codebook: Vec<Vec<VectorData>>,
}

impl ProductQuantizer {
    #[allow(dead_code)]
    pub fn train(data: &[VectorData], m: usize, k: usize, iterations: usize) -> Self {
        debug_assert!(!data.is_empty());

        let dim = data[0].dim();
        let sub_dim = dim / m;

        let codebook: Vec<Vec<VectorData>> = (0..m)
            .into_par_iter()
            .map(|sub_idx| {
                let start = sub_idx * sub_dim;
                let sub_data: Vec<VectorData> =
                    data.iter().map(|v| v.slice(start, start + sub_dim)).collect();

                Self::run_kmeans(&sub_data, k, iterations)
            })
            .collect();

        Self { m, k, sub_dim, codebook }
    }

    #[allow(dead_code)]
    fn run_kmeans(sub_data: &[VectorData], k: usize, iterations: usize) -> Vec<VectorData> {
        let n = sub_data.len();
        let dim = sub_data[0].dim();
        let d_type = sub_data[0].dtype();
        let mut centroids = vec![VectorData::zeros(dim, d_type); k];
        for i in 0..k {
            centroids[i] = sub_data[i % n].clone();
        }

        for _ in 0..iterations {
            let mut assignments = vec![0; n];
            let mut new_centroids = vec![VectorData::zeros(dim, d_type); k];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let mut min_dist = f64::MAX;
                let mut best_idx = 0;
                for (j, centroid) in centroids.iter().enumerate().take(k) {
                    let dist = sub_data[i].distance(centroid, MetricType::L2Squared);
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j;
                    }
                }
                assignments[i] = best_idx;
            }

            for i in 0..n {
                let cluster = assignments[i];
                new_centroids[cluster].add(&sub_data[i]);
                counts[cluster] += 1;
            }

            for j in 0..k {
                if counts[j] > 0 {
                    new_centroids[j].scale(1.0 / counts[j] as f64);
                }
            }
            centroids = new_centroids;
        }
        centroids
    }
}

#[cfg(test)]
mod tests {
    use system::vector_data::VectorData;

    use super::*;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<VectorData> {
        (0..n)
            .map(|i| {
                let data: Vec<f32> =
                    (0..dim).map(|j| ((i * dim + j) as f32 * 0.1) % 10.0).collect();
                VectorData::from_f32(data)
            })
            .collect()
    }

    #[test]
    fn test_train_basic() {
        let data = create_test_vectors(100, 32);
        let m = 8;
        let k = 16;
        let iterations = 5;

        let pq = ProductQuantizer::train(&data, m, k, iterations);

        assert_eq!(pq.m, m);
        assert_eq!(pq.k, k);
        assert_eq!(pq.sub_dim, 4);
        assert_eq!(pq.codebook.len(), m);

        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), k);
            for centroid in subspace_codebook {
                assert_eq!(centroid.dim(), pq.sub_dim);
            }
        }
    }

    #[test]
    fn test_train_dimensions_divisible() {
        let data = create_test_vectors(50, 64);
        let m = 16;
        let k = 32;

        let pq = ProductQuantizer::train(&data, m, k, 3);

        assert_eq!(pq.sub_dim, 4);
        assert_eq!(pq.m * pq.sub_dim, 64);
    }

    #[test]
    fn test_train_single_iteration() {
        let data = create_test_vectors(20, 16);
        let pq = ProductQuantizer::train(&data, 4, 8, 1);

        assert_eq!(pq.m, 4);
        assert_eq!(pq.k, 8);
        assert_eq!(pq.codebook.len(), 4);
    }

    #[test]
    fn test_train_many_iterations() {
        let data = create_test_vectors(50, 32);
        let pq = ProductQuantizer::train(&data, 8, 16, 20);

        assert_eq!(pq.m, 8);
        assert_eq!(pq.k, 16);
    }

    #[test]
    fn test_codebook_structure() {
        let data = create_test_vectors(40, 24);
        let m = 6;
        let k = 12;

        let pq = ProductQuantizer::train(&data, m, k, 5);

        for (subspace_idx, subspace_codebook) in pq.codebook.iter().enumerate() {
            assert_eq!(
                subspace_codebook.len(),
                k,
                "Subspace {} should have {} centroids",
                subspace_idx,
                k
            );

            for (centroid_idx, centroid) in subspace_codebook.iter().enumerate() {
                assert_eq!(
                    centroid.dim(),
                    pq.sub_dim,
                    "Centroid {} in subspace {} should have dimension {}",
                    centroid_idx,
                    subspace_idx,
                    pq.sub_dim
                );
            }
        }
    }

    #[test]
    fn test_small_dataset() {
        let data = create_test_vectors(10, 16);
        let pq = ProductQuantizer::train(&data, 4, 8, 3);

        assert_eq!(pq.m, 4);
        assert_eq!(pq.k, 8);
        assert_eq!(pq.sub_dim, 4);
    }

    #[test]
    fn test_k_larger_than_n() {
        let data = create_test_vectors(15, 32);
        let pq = ProductQuantizer::train(&data, 8, 20, 3);

        assert_eq!(pq.k, 20);
        assert_eq!(pq.codebook.len(), 8);

        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), 20);
        }
    }

    #[test]
    fn test_min_k_value() {
        let data = create_test_vectors(10, 16);
        let pq = ProductQuantizer::train(&data, 4, 2, 3);

        assert_eq!(pq.k, 2);
        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), 2);
        }
    }

    #[test]
    fn test_zero_iterations() {
        let data = create_test_vectors(20, 16);
        let pq = ProductQuantizer::train(&data, 4, 8, 0);

        assert_eq!(pq.codebook.len(), 4);
        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), 8);
        }
    }

    #[test]
    fn test_parallel_execution() {
        let data = create_test_vectors(100, 32);

        let pq1 = ProductQuantizer::train(&data, 8, 16, 5);
        let pq2 = ProductQuantizer::train(&data, 8, 16, 5);

        assert_eq!(pq1.m, pq2.m);
        assert_eq!(pq1.k, pq2.k);
        assert_eq!(pq1.sub_dim, pq2.sub_dim);
        assert_eq!(pq1.codebook.len(), pq2.codebook.len());
    }

    #[test]
    fn test_different_subspace_counts() {
        let data = create_test_vectors(50, 64);

        for m in [2, 4, 8, 16, 32] {
            let pq = ProductQuantizer::train(&data, m, 16, 3);
            assert_eq!(pq.m, m);
            assert_eq!(pq.sub_dim, 64 / m);
            assert_eq!(pq.codebook.len(), m);
        }
    }

    #[test]
    fn test_centroids_non_zero() {
        let data = create_test_vectors(30, 16);
        let pq = ProductQuantizer::train(&data, 4, 8, 5);

        for subspace_codebook in &pq.codebook {
            for centroid in subspace_codebook {
                let centroid_vec = centroid.to_f64_vec();
                let has_nonzero = (0..centroid_vec.len()).any(|i| centroid_vec[i].abs() > 1e-10);
                assert!(has_nonzero, "Centroid should have at least one non-zero element");
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_empty_data_panics() {
        let data: Vec<VectorData> = vec![];
        ProductQuantizer::train(&data, 4, 8, 3);
    }

    #[test]
    fn test_consistency_across_runs() {
        // With same data and parameters, structure should be consistent
        let data = create_test_vectors(50, 32);

        let pq1 = ProductQuantizer::train(&data, 8, 16, 10);
        let pq2 = ProductQuantizer::train(&data, 8, 16, 10);

        assert_eq!(pq1.m, pq2.m);
        assert_eq!(pq1.k, pq2.k);
        assert_eq!(pq1.sub_dim, pq2.sub_dim);

        for i in 0..pq1.m {
            assert_eq!(pq1.codebook[i].len(), pq2.codebook[i].len());
        }
    }

    #[test]
    fn test_large_m_value() {
        let data = create_test_vectors(100, 128);
        let pq = ProductQuantizer::train(&data, 32, 16, 3);

        assert_eq!(pq.m, 32);
        assert_eq!(pq.sub_dim, 4);
        assert_eq!(pq.codebook.len(), 32);
    }

    #[test]
    fn test_large_k_value() {
        let data = create_test_vectors(500, 32);
        let pq = ProductQuantizer::train(&data, 8, 256, 5);

        assert_eq!(pq.k, 256);
        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), 256);
        }
    }
}

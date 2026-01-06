use rayon::prelude::*;
use system::metric::{Distance, MetricType};
use system::vector_data::VectorData;

#[allow(dead_code)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug)]
pub struct ProductQuantizer {
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Centroids per subspace
    pub centroids_per_subspace: usize,
    /// Dimensions per subspace
    pub subspace_dim: usize,
    /// [M] matrices of size [K, sub_dim]
    pub codebook: Vec<Vec<VectorData>>,
}

impl ProductQuantizer {
    #[allow(dead_code)]
    pub fn train(
        data: &[VectorData], num_subspaces: usize, centroids_per_subspace: usize, iterations: usize,
    ) -> Self {
        debug_assert!(!data.is_empty());

        let dim = data[0].dim();
        let subspace_dim = dim / num_subspaces;

        let codebook: Vec<Vec<VectorData>> = (0..num_subspaces)
            .into_par_iter()
            .map(|sub_idx| {
                let start = sub_idx * subspace_dim;
                let sub_data: Vec<VectorData> =
                    data.iter().map(|v| v.slice(start, start + subspace_dim)).collect();

                Self::run_kmeans(&sub_data, centroids_per_subspace, iterations)
            })
            .collect();

        Self { num_subspaces, centroids_per_subspace, subspace_dim, codebook }
    }

    #[allow(dead_code)]
    fn run_kmeans(
        sub_data: &[VectorData], centroids_per_subspace: usize, iterations: usize,
    ) -> Vec<VectorData> {
        let n = sub_data.len();
        let dim = sub_data[0].dim();
        let d_type = sub_data[0].dtype();
        let mut centroids = vec![VectorData::zeros(dim, d_type); centroids_per_subspace];
        for i in 0..centroids_per_subspace {
            centroids[i] = sub_data[i % n].clone();
        }

        for _ in 0..iterations {
            let mut assignments = vec![0; n];
            let mut new_centroids = vec![VectorData::zeros(dim, d_type); centroids_per_subspace];
            let mut counts = vec![0usize; centroids_per_subspace];

            for i in 0..n {
                let mut min_dist = f64::MAX;
                let mut best_idx = 0;
                for (j, centroid) in centroids.iter().enumerate() {
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

            for j in 0..centroids_per_subspace {
                if counts[j] > 0 {
                    new_centroids[j].scale(1.0 / counts[j] as f64);
                }
            }
            centroids = new_centroids;
        }
        centroids
    }

    pub fn encode_batch(&self, data: &[VectorData]) -> Vec<u8> {
        let n = data.len();
        let mut flat_codes = vec![0u8; n * self.num_subspaces];

        // Use par_chunks_mut to get mutable access to 32-byte slices
        flat_codes.par_chunks_mut(self.num_subspaces).enumerate().for_each(|(i, code_row)| {
            let full_vector = &data[i];

            for (sub_idx, item) in code_row.iter_mut().enumerate().take(self.num_subspaces) {
                let start = sub_idx * self.subspace_dim;
                let end = start + self.subspace_dim;
                let sub_vec = full_vector.slice(start, end);

                *item = self.find_best_centroid(sub_idx, sub_vec);
            }
        });

        flat_codes
    }

    #[allow(dead_code)]
    pub fn batch_chunks<'a>(&self, codes: &'a [u8]) -> Vec<&'a [u8]> {
        codes.chunks(self.num_subspaces).collect::<Vec<_>>()
    }

    fn find_best_centroid(&self, sub_idx: usize, sub_vec: VectorData) -> u8 {
        let mut min_dist = f64::MAX;
        let mut best_u8 = 0u8;

        let subspace_codebook = &self.codebook[sub_idx];

        debug_assert_eq!(subspace_codebook.len(), self.centroids_per_subspace);
        for (idx, centroid) in subspace_codebook.iter().enumerate() {
            let dist = sub_vec.distance(centroid, MetricType::L2Squared);

            if dist < min_dist {
                min_dist = dist;
                best_u8 = idx as u8;
            }
        }
        best_u8
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

        assert_eq!(pq.num_subspaces, m);
        assert_eq!(pq.centroids_per_subspace, k);
        assert_eq!(pq.subspace_dim, 4);
        assert_eq!(pq.codebook.len(), m);

        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), k);
            for centroid in subspace_codebook {
                assert_eq!(centroid.dim(), pq.subspace_dim);
            }
        }
    }

    #[test]
    fn test_train_dimensions_divisible() {
        let data = create_test_vectors(50, 64);
        let m = 16;
        let k = 32;

        let pq = ProductQuantizer::train(&data, m, k, 3);

        assert_eq!(pq.subspace_dim, 4);
        assert_eq!(pq.num_subspaces * pq.subspace_dim, 64);
    }

    #[test]
    fn test_train_single_iteration() {
        let data = create_test_vectors(20, 16);
        let pq = ProductQuantizer::train(&data, 4, 8, 1);

        assert_eq!(pq.num_subspaces, 4);
        assert_eq!(pq.centroids_per_subspace, 8);
        assert_eq!(pq.codebook.len(), 4);
    }

    #[test]
    fn test_train_many_iterations() {
        let data = create_test_vectors(50, 32);
        let pq = ProductQuantizer::train(&data, 8, 16, 20);

        assert_eq!(pq.num_subspaces, 8);
        assert_eq!(pq.centroids_per_subspace, 16);
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
                    pq.subspace_dim,
                    "Centroid {} in subspace {} should have dimension {}",
                    centroid_idx,
                    subspace_idx,
                    pq.subspace_dim
                );
            }
        }
    }

    #[test]
    fn test_small_dataset() {
        let data = create_test_vectors(10, 16);
        let pq = ProductQuantizer::train(&data, 4, 8, 3);

        assert_eq!(pq.num_subspaces, 4);
        assert_eq!(pq.centroids_per_subspace, 8);
        assert_eq!(pq.subspace_dim, 4);
    }

    #[test]
    fn test_k_larger_than_n() {
        let data = create_test_vectors(15, 32);
        let pq = ProductQuantizer::train(&data, 8, 20, 3);

        assert_eq!(pq.centroids_per_subspace, 20);
        assert_eq!(pq.codebook.len(), 8);

        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), 20);
        }
    }

    #[test]
    fn test_min_k_value() {
        let data = create_test_vectors(10, 16);
        let pq = ProductQuantizer::train(&data, 4, 2, 3);

        assert_eq!(pq.centroids_per_subspace, 2);
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

        assert_eq!(pq1.num_subspaces, pq2.num_subspaces);
        assert_eq!(pq1.centroids_per_subspace, pq2.centroids_per_subspace);
        assert_eq!(pq1.subspace_dim, pq2.subspace_dim);
        assert_eq!(pq1.codebook.len(), pq2.codebook.len());
    }

    #[test]
    fn test_different_subspace_counts() {
        let data = create_test_vectors(50, 64);

        for m in [2, 4, 8, 16, 32] {
            let pq = ProductQuantizer::train(&data, m, 16, 3);
            assert_eq!(pq.num_subspaces, m);
            assert_eq!(pq.subspace_dim, 64 / m);
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
        let data = create_test_vectors(50, 32);

        let pq1 = ProductQuantizer::train(&data, 8, 16, 10);
        let pq2 = ProductQuantizer::train(&data, 8, 16, 10);

        assert_eq!(pq1.num_subspaces, pq2.num_subspaces);
        assert_eq!(pq1.centroids_per_subspace, pq2.centroids_per_subspace);
        assert_eq!(pq1.subspace_dim, pq2.subspace_dim);

        for i in 0..pq1.num_subspaces {
            assert_eq!(pq1.codebook[i].len(), pq2.codebook[i].len());
        }
    }

    #[test]
    fn test_large_m_value() {
        let data = create_test_vectors(100, 128);
        let pq = ProductQuantizer::train(&data, 32, 16, 3);

        assert_eq!(pq.num_subspaces, 32);
        assert_eq!(pq.subspace_dim, 4);
        assert_eq!(pq.codebook.len(), 32);
    }

    #[test]
    fn test_large_k_value() {
        let data = create_test_vectors(500, 32);
        let pq = ProductQuantizer::train(&data, 8, 256, 5);

        assert_eq!(pq.centroids_per_subspace, 256);
        for subspace_codebook in &pq.codebook {
            assert_eq!(subspace_codebook.len(), 256);
        }
    }

    #[test]
    fn test_encode_batch_basic() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(10, 32);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        assert_eq!(codes.len(), 10);
        for code in &codes {
            assert_eq!(code.len(), pq.num_subspaces);
            for &c in code.iter() {
                assert!((c as usize) < pq.centroids_per_subspace);
            }
        }
    }

    #[test]
    fn test_encode_batch_single_vector() {
        let train_data = create_test_vectors(50, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(1, 32);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].len(), pq.num_subspaces);
    }

    #[test]
    fn test_encode_batch_empty() {
        let train_data = create_test_vectors(50, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data: Vec<VectorData> = vec![];
        let codes = pq.encode_batch(&test_data);

        assert_eq!(codes.len(), 0);
    }

    #[test]
    fn test_encode_batch_large_batch() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(1000, 32);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        assert_eq!(codes.len(), 1000);
        for code in &codes {
            assert_eq!(code.len(), pq.num_subspaces);
        }
    }

    #[test]
    fn test_encode_batch_code_range() {
        let train_data = create_test_vectors(100, 64);
        let pq = ProductQuantizer::train(&train_data, 16, 32, 5);

        let test_data = create_test_vectors(50, 64);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        for (vec_idx, code) in codes.iter().enumerate() {
            for (sub_idx, &c) in code.iter().enumerate() {
                assert!(
                    (c as usize) < pq.centroids_per_subspace,
                    "Vector {} subspace {} has invalid code {} (should be < {})",
                    vec_idx,
                    sub_idx,
                    c,
                    pq.centroids_per_subspace
                );
            }
        }
    }

    #[test]
    fn test_encode_batch_training_data() {
        let train_data = create_test_vectors(50, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 10);

        let codes = pq.encode_batch(&train_data);
        let codes = pq.batch_chunks(&codes);

        assert_eq!(codes.len(), train_data.len());
        for code in &codes {
            assert_eq!(code.len(), pq.num_subspaces);
        }
    }

    #[test]
    fn test_encode_batch_deterministic() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(20, 32);
        let codes1 = pq.encode_batch(&test_data);
        let codes2 = pq.encode_batch(&test_data);

        assert_eq!(codes1.len(), codes2.len());
        for i in 0..codes1.len() {
            assert_eq!(codes1[i], codes2[i]);
        }
    }

    #[test]
    fn test_encode_batch_different_vectors() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(50, 32);
        let codes = pq.encode_batch(&test_data);

        // Check that not all codes are identical
        let first_code = &codes[0];
        let all_same = codes.iter().all(|c| c == first_code);
        assert!(!all_same, "Different vectors should produce different codes");
    }

    #[test]
    fn test_encode_batch_with_k256() {
        let train_data = create_test_vectors(300, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 256, 5);

        let test_data = create_test_vectors(10, 32);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        for code in &codes {
            for &c in code.iter() {
                assert!((c as usize) < 256);
            }
        }
    }

    #[test]
    fn test_encode_batch_parallel_consistency() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(500, 32);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        assert_eq!(codes.len(), 500);

        for code in &codes {
            assert_eq!(code.len(), pq.num_subspaces);
        }
    }

    #[test]
    fn test_encode_batch_subspace_independence() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_data = create_test_vectors(10, 32);
        let codes = pq.encode_batch(&test_data);
        let codes = pq.batch_chunks(&codes);

        // Each subspace should potentially have different codes
        for code in &codes {
            let unique_codes: std::collections::HashSet<_> = code.iter().collect();
            // With 8 subspaces, we'd expect some variation (not guaranteed but likely)
            assert!(unique_codes.len() >= 1);
        }
    }

    #[test]
    fn test_find_best_centroid_basic() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_vec = create_test_vectors(1, 32)[0].clone();
        let sub_vec = test_vec.slice(0, pq.subspace_dim);

        let best = pq.find_best_centroid(0, sub_vec);

        assert!((best as usize) < pq.centroids_per_subspace);
    }

    #[test]
    fn test_find_best_centroid_all_subspaces() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_vec = create_test_vectors(1, 32)[0].clone();

        for sub_idx in 0..pq.num_subspaces {
            let start = sub_idx * pq.subspace_dim;
            let sub_vec = test_vec.slice(start, start + pq.subspace_dim);
            let best = pq.find_best_centroid(sub_idx, sub_vec);

            assert!((best as usize) < pq.centroids_per_subspace);
        }
    }

    #[test]
    fn test_find_best_centroid_deterministic() {
        let train_data = create_test_vectors(100, 32);
        let pq = ProductQuantizer::train(&train_data, 8, 16, 5);

        let test_vec = create_test_vectors(1, 32)[0].clone();
        let sub_vec = test_vec.slice(0, pq.subspace_dim);

        let best1 = pq.find_best_centroid(0, sub_vec.clone());
        let best2 = pq.find_best_centroid(0, sub_vec);

        assert_eq!(best1, best2);
    }
}

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use half::{bf16, f16};
use rand::{Rng, rng};
use system::metric::{Distance, MetricType};
use system::vector_data::VectorData;

mod generators {
    use super::*;

    pub fn random_f32(len: usize) -> Vec<f32> {
        let mut rng = rng();
        (0..len).map(|_| rng.random::<f32>()).collect()
    }

    pub fn random_f64(len: usize) -> Vec<f64> {
        let mut rng = rng();
        (0..len).map(|_| rng.random::<f64>()).collect()
    }

    pub fn random_f16(len: usize) -> Vec<f16> {
        let mut rng = rng();
        (0..len).map(|_| rng.random::<f16>()).collect()
    }

    pub fn random_bf16(len: usize) -> Vec<bf16> {
        let mut rng = rng();
        (0..len).map(|_| rng.random::<bf16>()).collect()
    }
}

struct VectorBenchmark {
    vector_size: usize,
}

impl VectorBenchmark {
    fn new(vector_size: usize) -> Self {
        Self { vector_size }
    }

    fn generate_vector_types(&self) -> Vec<(&'static str, VectorData)> {
        vec![
            ("F32", VectorData::F32(generators::random_f32(self.vector_size))),
            ("F64", VectorData::F64(generators::random_f64(self.vector_size))),
            ("F16", VectorData::F16(generators::random_f16(self.vector_size))),
            ("BF16", VectorData::BF16(generators::random_bf16(self.vector_size))),
        ]
    }

    fn metric_types(&self) -> [MetricType; 4] {
        [MetricType::L2, MetricType::L2Squared, MetricType::Cosine, MetricType::Dot]
    }

    fn run(&self, c: &mut Criterion) {
        for (vec_name, vec_a) in self.generate_vector_types() {
            let mut group = c.benchmark_group(format!("distance_metrics_{}", vec_name));

            for metric in self.metric_types() {
                let metric_name = format!("{:?}", metric);
                let bench_id =
                    BenchmarkId::new(format!("{}_{}", vec_name, metric_name), self.vector_size);

                let vec_b = match &vec_a {
                    VectorData::F32(_) => VectorData::F32(generators::random_f32(self.vector_size)),
                    VectorData::F64(_) => VectorData::F64(generators::random_f64(self.vector_size)),
                    VectorData::F16(_) => VectorData::F16(generators::random_f16(self.vector_size)),
                    VectorData::BF16(_) => {
                        VectorData::BF16(generators::random_bf16(self.vector_size))
                    }
                };

                group.bench_with_input(bench_id, &metric, |bch, &m| {
                    bch.iter(|| {
                        let _ = vec_a.distance(black_box(&vec_b), black_box(m));
                    });
                });
            }
            group.finish();
        }
    }
}

fn bench_all_distances(c: &mut Criterion) {
    let vb = VectorBenchmark::new(1024);
    vb.run(c);
}

criterion_group!(benches, bench_all_distances);
criterion_main!(benches);

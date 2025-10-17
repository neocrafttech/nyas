use diskann::in_mem_index::InMemIndex;
use storage::metric::MetricType;
use storage::vector_data::VectorData;
use storage::vector_point::VectorPoint;

#[test]
fn test_in_mem_index() {
    let mut index = InMemIndex::new(32, 1.2, 50, MetricType::L2);

    for i in 0..100 {
        let vector = VectorData::from_f32(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
        index.insert(&VectorPoint::new(i, vector));
    }

    let query = VectorData::from_f32(vec![50.0, 100.0, 150.0]);
    let results = index.search(&query, 5, 20);

    assert!(results.len() <= 5);
    println!("Search results: {:?}", results);
}

macro_rules! test_index_matrix {
    (
        $metric:expr,
        [
            $(
                ( $test_name:ident, $dim:expr, $alpha:expr, $l_build:expr, $num_points:expr, $num_queries:expr )
            ),* $(,)?
        ]
    ) => {
        $(
            #[test]
            fn $test_name() {
                use rand::Rng;
                use crate::{InMemIndex, VectorPoint, VectorData, MetricType};

                let mut rng = rand::rng();

                let mut index = InMemIndex::new($dim, $alpha, $l_build, $metric);

                for i in 0..$num_points {
                    let vec: Vec<f32> = (0..$dim).map(|_| rng.random::<f32>()).collect();
                    let point = VectorPoint::new(i, VectorData::from_f32(vec));
                    index.insert(&point);
                }

                for qid in 0..$num_queries {
                    let query_vec: Vec<f32> = (0..$dim).map(|_| rng.random::<f32>()).collect();
                    let query = VectorData::from_f32(query_vec);

                    let results = index.search(&query, 5, 20);

                    assert!(
                        results.len() <= 5,
                        "Expected ≤5 results but got {} (test: {})",
                        results.len(),
                        stringify!($test_name)
                    );

                    println!(
                        "[{}] metric={:?}, dim={}, alpha={}, l_build={}, query={}, results={:?}",
                        stringify!($test_name),
                        $metric,
                        $dim,
                        $alpha,
                        $l_build,
                        qid,
                        results
                    );
                }
            }
        )*
    };
}

test_index_matrix!(
    MetricType::L2,
    [
        (test_index_small_l2, 32, 1.2, 50, 100, 3),
        (test_index_medium_l2, 128, 1.1, 1000, 200, 50),
        (test_index_large_l2, 1024, 1.5, 10000, 400, 100)
    ]
);

test_index_matrix!(
    MetricType::L2Squared,
    [
        (test_index_small_l2_squared, 32, 1.2, 50, 100, 3),
        (test_index_medium_l2_squared, 64, 1.1, 1000, 200, 50),
        (test_index_large_l2_squared, 1536, 1.5, 10000, 400, 100)
    ]
);

test_index_matrix!(
    MetricType::Dot,
    [
        (test_index_small_dot, 32, 1.2, 50, 100, 3),
        (test_index_medium_dot, 64, 1.1, 1000, 200, 50),
        (test_index_large_dot, 1024, 1.5, 10000, 300, 100)
    ]
);

test_index_matrix!(
    MetricType::Cosine,
    [
        (test_index_small_cosine, 32, 1.2, 50, 100, 3),
        (test_index_medium_cosine, 64, 1.1, 1000, 200, 50),
        (test_index_large_cosine, 896, 1.3, 10000, 400, 100)
    ]
);

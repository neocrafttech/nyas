use half::{bf16, f16};
use nalgebra::DVector;
use rand;
use rand::Rng;
use storage::metric::Distance;
use storage::metric::MetricType;
use storage::vector_data::VectorData;

fn norm(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    (a - b).norm()
}

fn norm_squared(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    (a - b).norm_squared()
}

fn dot_product(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    a.dot(&b)
}

fn cosine_distance(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let dot = a.dot(&b);
    let norm_a = a.norm();
    let norm_b = b.norm();
    1.0 - dot / (norm_a * norm_b)
}

macro_rules! test_distance_matrix {
    (
        $metric:expr,
        $nalgebra_op:expr,
        $dim:expr,
        $epsilon:expr,
        [
            $(
                ( $test_name:ident, $data_type:ty, $vector_data_variant:path )
            ),* $(,)?
        ]
    ) => {
        $(
            test_distance_fn!(
                $test_name,
                $metric,
                $nalgebra_op,
                $data_type,
                $vector_data_variant,
                $dim,
                $epsilon
            );
        )*
    };
}

macro_rules! test_distance_fn {
    (
           $test_name:ident,
           $metric:expr,
           $nalgebra_op:expr,
           $data_type:ty,
           $vector_data_variant:path,
           $dim:expr,
           $epsilon:expr
       ) => {
        #[test]
        fn $test_name() {
            let mut rng = rand::rng();

            // Generate random vectors of the specified type
            let a_vec: Vec<$data_type> = (0..$dim).map(|_| rng.random::<$data_type>()).collect();
            let b_vec: Vec<$data_type> = (0..$dim).map(|_| rng.random::<$data_type>()).collect();

            let a_vd = $vector_data_variant(a_vec.clone());
            let b_vd = $vector_data_variant(b_vec.clone());

            let result = a_vd.distance(&b_vd, $metric);

            // Convert to f64 for nalgebra calculation to maintain precision
            let a_f64: Vec<f64> = a_vec.iter().map(|&x| f64::from(x)).collect();
            let b_f64: Vec<f64> = b_vec.iter().map(|&x| f64::from(x)).collect();

            let a_nalgebra = DVector::from_vec(a_f64);
            let b_nalgebra = DVector::from_vec(b_f64);

            // Calculate the expected result using nalgebra
            let expected = $nalgebra_op(&a_nalgebra, &b_nalgebra);

            assert!(
                (result - expected).abs() < $epsilon,
                "Test failed for {}: Result: {}, Expected: {}, Epsilon: {}",
                stringify!($test_name),
                result,
                expected,
                $epsilon
            );
        }
    };
}

test_distance_matrix!(
    MetricType::L2,
    norm,
    1024,
    1e-3,
    [
        (test_l2_distance_bf16, bf16, VectorData::BF16),
        (test_l2_distance_f16, f16, VectorData::F16),
        (test_l2_distance_f32, f32, VectorData::F32),
        (test_l2_distance_f64, f64, VectorData::F64),
    ]
);

test_distance_matrix!(
    MetricType::L2Squared,
    norm_squared,
    1536,
    1e-3,
    [
        (test_l2_sq_distance_bf16, bf16, VectorData::BF16),
        (test_l2_sq_distance_f16, f16, VectorData::F16),
        (test_l2_sq_distance_f32, f32, VectorData::F32),
        (test_l2_sq_distance_f64, f64, VectorData::F64),
    ]
);

test_distance_matrix!(
    MetricType::Cosine,
    cosine_distance,
    800,
    1e-5,
    [
        (test_cos_distance_bf16, bf16, VectorData::BF16),
        (test_cos_distance_f16, f16, VectorData::F16),
        (test_cos_distance_f32, f32, VectorData::F32),
        (test_cos_distance_f64, f64, VectorData::F64),
    ]
);

test_distance_matrix!(
    MetricType::Dot,
    dot_product,
    25,
    1e-5,
    [
        (test_dot_distance_bf16, bf16, VectorData::BF16),
        (test_dot_distance_f16, f16, VectorData::F16),
        (test_dot_distance_f32, f32, VectorData::F32),
        (test_dot_distance_f64, f64, VectorData::F64),
    ]
);

#[test]
fn test_mixed_types_l2_squared() {
    let a_f32 = VectorData::F32(vec![1.0, 2.0, 3.0]);
    let b_f16 = VectorData::F16(vec![
        f16::from_f32(4.0),
        f16::from_f32(5.0),
        f16::from_f32(6.0),
    ]);

    let result = a_f32.distance(&b_f16, MetricType::L2Squared);
    let expected = (1.0 - 4.0f32).powi(2) + (2.0 - 5.0f32).powi(2) + (3.0 - 6.0f32).powi(2);

    assert!((result - expected as f64).abs() < 1e-6);
}

#[test]
fn test_mixed_types_f64_cosine() {
    let a_f64 = VectorData::F64(vec![1.0, 2.0, 3.0, 4.0]);
    let b_f32 = VectorData::F32(vec![5.0, 6.0, 7.0, 8.0]);

    let result = a_f64.distance(&b_f32, MetricType::Cosine);

    let a_nalgebra = DVector::from_vec(a_f64.to_f64_vec());
    let b_nalgebra = DVector::from_vec(b_f32.to_f64_vec());

    let expected = 1.0 - (a_nalgebra.dot(&b_nalgebra) / (a_nalgebra.norm() * b_nalgebra.norm()));

    assert!((result - expected).abs() < 1e-6);
}

#[test]
fn cosine_distance_with_zero_vector() {
    let a = VectorData::F32(vec![1.0, 2.0, 3.0]);
    let b = VectorData::F32(vec![0.0, 0.0, 0.0]);

    // Cosine distance is 1 if one vector is zero
    assert_eq!(a.distance(&b, MetricType::Cosine), 1.0);
    assert_eq!(b.distance(&a, MetricType::Cosine), 1.0);
}

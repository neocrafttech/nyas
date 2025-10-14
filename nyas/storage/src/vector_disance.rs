use crate::metric::{Distance, MetricType};
use crate::vector_data::VectorData;
use simsimd::{self, SpatialSimilarity};

macro_rules! calculate {
    ($method:ident, $a:expr, $b:expr) => {
        SpatialSimilarity::$method($a, $b).expect(concat!(
            "Failed to compute ",
            stringify!($method),
            " distance"
        ))
    };
    ($method:ident, $a:expr, $b:expr,  $dtype:ty) => {{
        let a_cast: &[$dtype] =
            unsafe { std::slice::from_raw_parts($a.as_ptr() as *const $dtype, $a.len()) };
        let b_cast: &[$dtype] =
            unsafe { std::slice::from_raw_parts($b.as_ptr() as *const $dtype, $b.len()) };

        calculate!($method, a_cast, b_cast)
    }};
}

macro_rules! distance {
    ($method:ident, $a:expr, $b:expr) => {{
        use $crate::vector_data::VectorData;
        let a = $a;
        let b = $b;

        assert_eq!(a.dim(), b.dim(), "Vector dimensions must match");

        match (a, b) {
            (VectorData::BF16(av), VectorData::BF16(bv)) => {
                calculate!($method, av, bv, simsimd::bf16)
            }
            (VectorData::F16(av), VectorData::F16(bv)) => {
                calculate!($method, av, bv, simsimd::f16)
            }
            (VectorData::F32(av), VectorData::F32(bv)) => {
                calculate!($method, av, bv)
            }
            (VectorData::F64(av), VectorData::F64(bv)) => {
                calculate!($method, av, bv)
            }
            (_, VectorData::F64(bv)) => {
                calculate!($method, &a.to_f64_vec(), bv)
            }
            (VectorData::F64(av), _) => {
                calculate!($method, av, &b.to_f64_vec())
            }
            _ => {
                let av = a.to_f32_vec();
                let bv = b.to_f32_vec();
                calculate!($method, &av, &bv)
            }
        }
    }};
}

impl Distance for VectorData {
    fn distance(&self, other: &Self, metric: MetricType) -> f64 {
        match metric {
            MetricType::L2 => distance!(l2, self, other),
            MetricType::L2Squared => distance!(l2sq, self, other),
            MetricType::Cosine => distance!(cos, self, other),
            MetricType::Dot => distance!(dot, self, other),
        }
    }
}

use nalgebra::DVector;
use storage::vector_data::VectorData;
use storage::vector_data::VectorDataType;

#[test]
fn test_conversions() {
    let f32_data = vec![1.0, 2.0, 3.0, 4.0];
    let vec_data = VectorData::from_f32(f32_data.clone());

    assert_eq!(vec_data.dim(), 4);
    assert_eq!(vec_data.dtype(), VectorDataType::F32);

    let converted = vec_data.to_f32();
    assert_eq!(converted.as_slice(), f32_data.as_slice());
}

#[test]
fn test_dvector_conversion() {
    let dv = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let vec_data = VectorData::from_dvector_f32(dv.clone());

    assert_eq!(vec_data.dim(), 3);
    assert_eq!(vec_data.to_f32(), dv);
}

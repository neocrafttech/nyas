use std::{fmt, mem};

use bytemuck::cast_slice;
use half::{bf16, f16};
use nalgebra::{DVector, DVectorView};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Debug, Deserialize, Serialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum VectorData {
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl VectorData {
    #[inline]
    pub fn from_bf16(data: Vec<bf16>) -> Self {
        Self::BF16(data)
    }

    #[inline]
    pub fn from_f16(data: Vec<f16>) -> Self {
        Self::F16(data)
    }

    #[inline]
    pub fn from_f32(data: Vec<f32>) -> Self {
        Self::F32(data)
    }

    #[inline]
    pub fn from_f64(data: Vec<f64>) -> Self {
        Self::F64(data)
    }

    #[inline]
    pub fn from_dvector_f32(data: DVector<f32>) -> Self {
        Self::F32(data.data.into())
    }

    #[inline]
    pub fn from_dvector_f64(data: DVector<f64>) -> Self {
        Self::F64(data.data.into())
    }

    #[inline]
    pub fn from_slice_f32(data: &[f32]) -> Self {
        Self::F32(data.to_vec())
    }

    /// Create zeros with specified dimension and type
    pub fn zeros(dim: usize, dtype: VectorDataType) -> Self {
        match dtype {
            VectorDataType::BF16 => Self::BF16(vec![bf16::ZERO; dim]),
            VectorDataType::F16 => Self::F16(vec![f16::ZERO; dim]),
            VectorDataType::F32 => Self::F32(vec![0.0; dim]),
            VectorDataType::F64 => Self::F64(vec![0.0; dim]),
        }
    }

    /// Create ones with specified dimension and type
    pub fn ones(dim: usize, dtype: VectorDataType) -> Self {
        match dtype {
            VectorDataType::BF16 => Self::BF16(vec![bf16::ONE; dim]),
            VectorDataType::F16 => Self::F16(vec![f16::ONE; dim]),
            VectorDataType::F32 => Self::F32(vec![1.0; dim]),
            VectorDataType::F64 => Self::F64(vec![1.0; dim]),
        }
    }

    /// Get the dimensionality (length) of the vector
    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            Self::BF16(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Check if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dim() == 0
    }

    /// Get the data type
    #[inline]
    pub fn dtype(&self) -> VectorDataType {
        match self {
            Self::BF16(_) => VectorDataType::BF16,
            Self::F16(_) => VectorDataType::F16,
            Self::F32(_) => VectorDataType::F32,
            Self::F64(_) => VectorDataType::F64,
        }
    }

    /// Get the size in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::BF16(v) => v.len() * mem::size_of::<bf16>(),
            Self::F16(v) => v.len() * mem::size_of::<f16>(),
            Self::F32(v) => v.len() * mem::size_of::<f32>(),
            Self::F64(v) => v.len() * mem::size_of::<f64>(),
        }
    }

    /// Get raw bytes in native endian format (zero-copy view)
    #[inline]
    pub fn as_raw_bytes(&self) -> &[u8] {
        match self {
            Self::BF16(v) => cast_slice(v),
            Self::F16(v) => cast_slice(v),
            Self::F32(v) => cast_slice(v),
            Self::F64(v) => cast_slice(v),
        }
    }

    /// Get a value at index as f32 (without full conversion)
    pub fn get_f32(&self, idx: usize) -> Option<f32> {
        match self {
            Self::BF16(v) => v.get(idx).map(|x| x.to_f32()),
            Self::F16(v) => v.get(idx).map(|x| x.to_f32()),
            Self::F32(v) => v.get(idx).copied(),
            Self::F64(v) => v.get(idx).map(|&x| x as f32),
        }
    }

    /// Get a slice view (if the data is f32)
    #[inline]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Get a mutable slice view (if the data is f32)
    #[inline]
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Convert to f32 DVector (consuming self)
    pub fn into_f32(self) -> DVector<f32> {
        let vec = match self {
            Self::BF16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            Self::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            Self::F32(v) => v,
            Self::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        };
        DVector::from_vec(vec)
    }

    /// Convert to f32 DVector (cloning if necessary)
    pub fn to_f32(&self) -> DVector<f32> {
        match self {
            Self::F32(v) => DVector::from_vec(v.clone()),
            _ => self.clone().into_f32(),
        }
    }

    /// Convert to f32 Vec (consuming self)
    pub fn into_f32_vec(self) -> Vec<f32> {
        match self {
            Self::BF16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            Self::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            Self::F32(v) => v,
            Self::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        }
    }

    /// Convert to f32 Vec (cloning if necessary)
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::F32(v) => v.clone(),
            Self::F64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::BF16(v) => v.iter().map(|x| x.to_f64()).collect(),
            Self::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
            Self::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::F64(v) => v.clone(),
        }
    }

    /// Get DVectorView<f32> without copying (only works for F32 variant)
    #[inline]
    pub fn as_f32_view(&self) -> Option<DVectorView<'_, f32>> {
        match self {
            Self::F32(v) => Some(DVectorView::from_slice(v, v.len())),
            _ => None,
        }
    }

    /// Convert to f64 DVector (consuming self)
    pub fn into_f64(self) -> DVector<f64> {
        let vec = match self {
            Self::BF16(v) => v.into_iter().map(|x| x.to_f32() as f64).collect(),
            Self::F16(v) => v.into_iter().map(|x| x.to_f32() as f64).collect(),
            Self::F32(v) => v.into_iter().map(|x| x as f64).collect(),
            Self::F64(v) => v,
        };
        DVector::from_vec(vec)
    }

    /// Convert to f64 DVector (cloning if necessary)
    pub fn to_f64(&self) -> DVector<f64> {
        match self {
            Self::F64(v) => DVector::from_vec(v.clone()),
            _ => self.clone().into_f64(),
        }
    }

    /// Convert to a different data type
    pub fn convert_to(self, target: VectorDataType) -> Self {
        if self.dtype() == target {
            return self;
        }

        let f32_vec = self.into_f32_vec();
        match target {
            VectorDataType::BF16 => Self::BF16(f32_vec.into_iter().map(bf16::from_f32).collect()),
            VectorDataType::F16 => Self::F16(f32_vec.into_iter().map(f16::from_f32).collect()),
            VectorDataType::F32 => Self::F32(f32_vec),
            VectorDataType::F64 => Self::F64(f32_vec.into_iter().map(|x| x as f64).collect()),
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Self::BF16(v) => Self::BF16(v[start..end].to_vec()),
            Self::F16(v) => Self::F16(v[start..end].to_vec()),
            Self::F32(v) => Self::F32(v[start..end].to_vec()),
            Self::F64(v) => Self::F64(v[start..end].to_vec()),
        }
    }

    //TODO : Use Simd
    pub fn add(&mut self, other: &Self) {
        match (self, other) {
            (Self::BF16(a), Self::BF16(b)) => {
                for (x, &y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
            (Self::F16(a), Self::F16(b)) => {
                for (x, &y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
            (Self::F32(a), Self::F32(b)) => {
                for (x, &y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
            (Self::F64(a), Self::F64(b)) => {
                for (x, &y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
            _ => panic!("VectorData types do not match"),
        }
    }

    //TODO : Use Simd
    pub fn scale(&mut self, factor: f64) {
        match self {
            Self::BF16(v) => {
                for x in v {
                    *x *= bf16::from_f64(factor);
                }
            }
            Self::F16(v) => {
                for x in v {
                    *x *= f16::from_f64(factor);
                }
            }
            Self::F32(v) => {
                for x in v {
                    *x *= factor as f32;
                }
            }
            Self::F64(v) => {
                for x in v {
                    *x *= factor;
                }
            }
        }
    }
}

/// Represents the numeric type of VectorData
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorDataType {
    BF16,
    F16,
    F32,
    F64,
}

impl VectorDataType {
    /// Get the size in bytes for this type
    #[inline]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::BF16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Get the name of the type
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for VectorDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl From<Vec<bf16>> for VectorData {
    #[inline]
    fn from(v: Vec<bf16>) -> Self {
        Self::BF16(v)
    }
}

impl From<Vec<f16>> for VectorData {
    #[inline]
    fn from(v: Vec<f16>) -> Self {
        Self::F16(v)
    }
}

impl From<Vec<f32>> for VectorData {
    #[inline]
    fn from(v: Vec<f32>) -> Self {
        Self::F32(v)
    }
}

impl From<Vec<f64>> for VectorData {
    #[inline]
    fn from(v: Vec<f64>) -> Self {
        Self::F64(v)
    }
}

impl From<DVector<f32>> for VectorData {
    #[inline]
    fn from(dv: DVector<f32>) -> Self {
        Self::F32(dv.data.into())
    }
}

impl From<DVector<f64>> for VectorData {
    #[inline]
    fn from(dv: DVector<f64>) -> Self {
        Self::F64(dv.data.into())
    }
}

impl Default for VectorData {
    fn default() -> Self {
        Self::F32(Vec::new())
    }
}

impl fmt::Display for VectorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorData<{}>({} dims)", self.dtype(), self.dim())
    }
}

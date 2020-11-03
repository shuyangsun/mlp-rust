use ndarray_stats::MaybeNan;
use num_traits::{Float, FromPrimitive};
use rand::distributions::uniform::SampleUniform;
use std::cmp::PartialOrd;

pub trait MLPFloat:
    'static
    + Float
    + FromPrimitive
    + PartialOrd
    + MaybeNan
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display // TODO: remove theses, just for debugging.
{
}
pub trait MLPFLoatRandSampling: MLPFloat + SampleUniform {}

impl MLPFloat for f32 {}
impl MLPFloat for f64 {}

impl MLPFLoatRandSampling for f32 {}
impl MLPFLoatRandSampling for f64 {}

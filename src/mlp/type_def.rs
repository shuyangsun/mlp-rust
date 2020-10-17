use ndarray_stats::MaybeNan;
use num_traits::{Float, FromPrimitive};
use std::cmp::PartialOrd;

pub trait MLPFloat: Float + FromPrimitive + PartialOrd + MaybeNan {}

impl MLPFloat for f32 {}
impl MLPFloat for f64 {}

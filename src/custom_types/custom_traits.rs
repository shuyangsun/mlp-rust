extern crate ndarray;
use ndarray::prelude::*;

use ndarray_stats::MaybeNan;
use num_traits::{Float, FromPrimitive};
use std::cmp::PartialOrd;

pub trait MLPFloat: Float + FromPrimitive + PartialOrd + MaybeNan {}

pub trait Tensor<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: &Array2<T>) -> Array2<T>;
    fn backward_batch(&self, output: &Array2<T>) -> Array2<T>;

    fn backward(&self, output: &Array2<T>) -> Array1<T> {
        self.backward_batch(output).mean_axis(Axis(0)).unwrap()
    }
}

impl MLPFloat for f32 {}
impl MLPFloat for f64 {}

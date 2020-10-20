extern crate ndarray;
use ndarray::prelude::*;

use ndarray_stats::MaybeNan;
use num_traits::{Float, FromPrimitive};
use rand::distributions::uniform::SampleUniform;
use std::cmp::PartialOrd;

pub trait MLPFloat: 'static + Float + FromPrimitive + PartialOrd + MaybeNan {}
pub trait MLPFLoatRandSampling: MLPFloat + SampleUniform {}

pub trait Tensor<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> Box<ArrayD<T>>;
    fn backward_batch(&self, output: ArrayViewD<T>) -> Box<ArrayD<T>>;

    fn backward(&self, output: ArrayViewD<T>) -> Box<ArrayD<T>> {
        Box::new(
            self.backward_batch(output)
                .as_ref()
                .mean_axis(Axis(0))
                .unwrap(),
        )
    }
}

pub trait TensorUpdatable<T>: Tensor<T>
where
    T: MLPFloat,
{
    fn is_frozen(&self) -> bool {
        false
    }
    fn updatable_mat(&mut self) -> ArrayViewMutD<T>;
    fn update(&mut self, gradient_last_layer: ArrayViewD<T>) {
        if self.is_frozen() {
            return;
        }
        let mut original_mat = self.updatable_mat();
        let update_res: ArrayD<T> = &original_mat - &gradient_last_layer;
        original_mat.assign(&update_res);
    }
}

impl MLPFloat for f32 {}
impl MLPFloat for f64 {}

impl MLPFLoatRandSampling for f32 {}
impl MLPFLoatRandSampling for f64 {}

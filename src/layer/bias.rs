extern crate ndarray;
use super::super::custom_types::numerical_traits::MLPFloat;
use super::super::custom_types::tensor_traits::{TensorComputable, TensorUpdatable};
use ndarray::prelude::*;

pub struct Bias<T>
where
    T: MLPFloat,
{
    bias_arr: ArrayD<T>, // 2D array with shape 1 x N, N = number of neurons
}

impl<T> TensorComputable<T> for Bias<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        &input + &self.bias_arr.view()
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }
}

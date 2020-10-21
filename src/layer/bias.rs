extern crate ndarray;
use super::super::custom_types::numerical_traits::MLPFloat;
use super::super::custom_types::tensor_traits::{Tensor, TensorUpdatable};
use ndarray::prelude::*;

pub struct Bias<T>
where
    T: MLPFloat,
{
    bias_arr: Box<Array2<T>>,
}

impl<T> Tensor<T> for Bias<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<'_, T>) -> Box<ArrayD<T>> {
        unimplemented!()
    }

    fn backward_batch(&self, output: ArrayViewD<'_, T>) -> Box<ArrayD<T>> {
        unimplemented!()
    }
}

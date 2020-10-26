extern crate ndarray;
use super::super::custom_types::numerical_traits::{MLPFLoatRandSampling, MLPFloat};
use super::super::custom_types::tensor_traits::{TensorComputable, TensorUpdatable};
use ndarray::prelude::*;

pub struct Bias<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
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

impl<T> TensorUpdatable<T> for Bias<T>
where
    T: MLPFloat,
{
    fn is_frozen(&self) -> bool {
        self.is_frozen
    }
    fn updatable_mat(&mut self) -> ArrayViewMutD<'_, T> {
        self.bias_arr.view_mut()
    }
}

impl<T> Bias<T>
where
    T: MLPFloat + MLPFLoatRandSampling,
{
    fn new(size: usize) -> Self {
        Self {
            is_frozen: false,
            bias_arr: Array2::zeros((1, size)).into_dyn(),
        }
    }
}

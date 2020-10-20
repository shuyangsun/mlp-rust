extern crate ndarray;
use super::super::custom_types::custom_traits::{
    MLPFLoatRandSampling, MLPFloat, Tensor, TensorUpdatable,
};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct Weight<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    weight_mat: Box<Array2<T>>,
}

impl<T> Tensor<T> for Weight<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> Box<ArrayD<T>> {
        assert_eq!(input.shape().len(), 2);
        let input_2d: ArrayView2<T> = input.into_dimensionality().unwrap();
        let mat_mul_res = self
            .weight_mat
            .view()
            .dot(&input_2d)
            .into_dimensionality()
            .unwrap();
        Box::new(mat_mul_res)
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> Box<ArrayD<T>> {
        unimplemented!()
    }
}

impl<T> TensorUpdatable<T> for Weight<T>
where
    T: MLPFloat,
{
    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn updatable_mat(&mut self) -> ArrayViewMutD<T> {
        self.weight_mat.view_mut().into_dyn()
    }
}

impl<T> Weight<T>
where
    T: MLPFloat + MLPFLoatRandSampling,
{
    fn new(from_layer_size: usize, to_layer_size: usize) -> Self {
        Self {
            is_frozen: false,
            weight_mat: Box::new(Array::random(
                (from_layer_size, to_layer_size),
                Uniform::new(-T::one(), T::one()),
            )),
        }
    }
}

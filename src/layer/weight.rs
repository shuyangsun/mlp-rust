extern crate ndarray;
use super::super::custom_types::numerical_traits::{MLPFLoatRandSampling, MLPFloat};
use super::super::custom_types::tensor_traits::{TensorComputable, TensorUpdatable};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct Weight<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    weight_mat: ArrayD<T>, // n1 x n2
}

impl<T> TensorComputable<T> for Weight<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        assert_eq!(input.ndim(), 2);
        let input_2d: ArrayView2<T> = input.into_dimensionality().unwrap();
        let weight_2d: ArrayView2<T> = self.weight_mat.view().into_dimensionality().unwrap();
        let mat_mul_res = weight_2d.dot(&input_2d).into_dimensionality().unwrap();
        assert_eq!(mat_mul_res.ndim(), 2);
        mat_mul_res
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T> {
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
            weight_mat: Array::random(
                (from_layer_size, to_layer_size),
                Uniform::new(-T::one(), T::one()),
            )
            .into_dyn(),
        }
    }
}

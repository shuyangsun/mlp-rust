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
        let mat_mul_res = input_2d.dot(&weight_2d).into_dimensionality().unwrap();
        assert_eq!(mat_mul_res.ndim(), 2);
        mat_mul_res
    }

    fn backward_batch(&self, _: ArrayViewD<T>) -> ArrayD<T> {
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
        self.weight_mat.view_mut()
    }
}

impl<T> Weight<T>
where
    T: MLPFloat + MLPFLoatRandSampling,
{
    fn new_random_uniform(from_layer_size: usize, to_layer_size: usize) -> Self {
        Self {
            is_frozen: false,
            weight_mat: Array::random(
                (from_layer_size, to_layer_size),
                Uniform::new(-T::one(), T::one()),
            )
            .into_dyn(),
        }
    }

    fn new_random_uniform_frozen(from_layer_size: usize, to_layer_size: usize) -> Self {
        let mut res = Self::new_random_uniform(from_layer_size, to_layer_size);
        res.is_frozen = true;
        res
    }
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;

    use super::super::super::custom_types::tensor_traits::{TensorComputable, TensorUpdatable};
    use super::Weight;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_weights_forward() {
        let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let weights = Weight::new_random_uniform(2, 5);
        let forward_res = weights.forward(arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[3usize, 5usize]);
    }

    #[test]
    fn test_weights_forward_rand() {
        let shape = [1000, 100];
        let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let weights = Weight::new_random_uniform(100, 50);
        let forward_res = weights.forward(rand_arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[1000usize, 50usize]);
    }
}

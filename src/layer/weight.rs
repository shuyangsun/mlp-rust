extern crate ndarray;
use super::super::custom_types::numerical_traits::{MLPFLoatRandSampling, MLPFloat};
use super::super::custom_types::tensor_traits::{
    TensorComputable, TensorForwardBatchIndependent, TensorUpdatable,
};
use super::super::util::linalg;
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
        linalg::mat_mul(
            &input.into_dimensionality::<Ix2>().unwrap(),
            &self.weight_mat.view().into_dimensionality::<Ix2>().unwrap(),
        )
        .into_dyn()
    }

    fn backward_batch(&self, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        linalg::par_mat_mul(
            &input.into_dimensionality::<Ix2>().unwrap(),
            &self.weight_mat.view().into_dimensionality::<Ix2>().unwrap(),
        )
        .into_dyn()
    }
}

impl<T> TensorForwardBatchIndependent<T> for Weight<T> where T: MLPFloat {}

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
    pub fn new_random_uniform(from_layer_size: usize, to_layer_size: usize) -> Self {
        Self {
            is_frozen: false,
            weight_mat: Array::random(
                (from_layer_size, to_layer_size),
                Uniform::new(-T::one(), T::one()),
            )
            .into_dyn(),
        }
    }

    pub fn new_random_uniform_frozen(from_layer_size: usize, to_layer_size: usize) -> Self {
        let mut res = Self::new_random_uniform(from_layer_size, to_layer_size);
        res.is_frozen = true;
        res
    }
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;

    use super::super::super::custom_types::tensor_traits::TensorComputable;
    use super::Weight;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_weights_forward() {
        let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let weights = Weight::new_random_uniform(2, 5);
        let forward_res = weights.forward(arr.view());
        let par_forward_res = weights.par_forward(arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[3usize, 5usize]);
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_weights_forward_rand_1() {
        let shape = [1024, 100];
        let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let weights = Weight::new_random_uniform(100, 50);
        let forward_res = weights.forward(rand_arr.view());
        let par_forward_res = weights.par_forward(rand_arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[1024usize, 50usize]);
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_weights_forward_rand_2() {
        let shape = [997, 100];
        let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let weights = Weight::new_random_uniform(100, 50);
        let forward_res = weights.forward(rand_arr.view());
        let par_forward_res = weights.par_forward(rand_arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[997usize, 50usize]);
        assert_eq!(forward_res, par_forward_res);
    }
}

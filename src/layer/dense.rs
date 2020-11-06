use super::super::utility::{array::mat_mul, array::par_mat_mul, counter::CounterEst};
use crate::{MLPFLoatRandSampling, MLPFloat, Optimizer, Tensor};
use ndarray::{Array2, ArrayView2, Ix2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

pub struct Dense<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    weight_mat: Array2<T>,
}

impl<T> Tensor<T, Ix2, Ix2> for Dense<T>
where
    T: MLPFloat,
{
    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn forward(&self, input: ArrayView2<T>) -> Array2<T> {
        mat_mul(&input, &self.weight_view())
    }

    fn par_forward(&self, input: ArrayView2<T>) -> Array2<T> {
        par_mat_mul(&input, &self.weight_view())
    }

    fn backward_respect_to_input(
        &self,
        _: ArrayView2<T>,
        layer_output: ArrayView2<T>,
    ) -> Array2<T> {
        par_mat_mul(&layer_output, &self.weight_view().t())
    }

    fn backward_update(
        &mut self,
        input: ArrayView2<T>,
        output_gradient: ArrayView2<T>,
        optimizer: &Box<dyn Optimizer<T, Ix2>>,
    ) {
        let weight_gradient = mat_mul(&input.t(), &output_gradient);
        optimizer.change_values(&mut self.weight_mat.view_mut(), weight_gradient.view());
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.weight_mat.len())
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.weight_mat.len() * 2 - self.weight_mat.shape()[0])
    }
}

impl<T> Dense<T>
where
    T: MLPFloat,
{
    pub fn weight_view(&self) -> ArrayView2<T> {
        self.weight_mat.view()
    }
}

impl<T> Dense<T>
where
    T: MLPFloat + MLPFLoatRandSampling,
{
    pub fn new_random_uniform(from_layer_size: usize, to_layer_size: usize) -> Self {
        Self {
            is_frozen: false,
            weight_mat: Array2::random(
                (from_layer_size, to_layer_size),
                Uniform::new(-T::one(), T::one()),
            ),
        }
    }

    pub fn new_random_uniform_frozen(from_layer_size: usize, to_layer_size: usize) -> Self {
        let mut res = Self::new_random_uniform(from_layer_size, to_layer_size);
        res.is_frozen = true;
        res
    }
}

#[macro_export]
macro_rules! dense {
    ($a:expr, $b:expr) => {{
        Box::new(Dense::new_random_uniform($a, $b))
    }};
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;

    use super::super::super::traits::tensor_traits::Tensor;
    use super::Dense;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_weights_forward() {
        let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]);
        let weights = Dense::new_random_uniform(2, 5);
        let forward_res = weights.forward(arr.view());
        let par_forward_res = weights.par_forward(arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[3usize, 5usize]);
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_weights_forward_rand_1() {
        let shape = [997, 100]; // 997 is a prime number, testing parallel splitting.
        let output_size = 50;
        let rand_arr = &Array::random(shape, Uniform::new(0., 10.));
        let weights = Dense::new_random_uniform(shape[1], output_size);
        let forward_res = weights.forward(rand_arr.view());
        let par_forward_res = weights.par_forward(rand_arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(forward_res.shape(), &[shape[0], output_size]);
        assert_eq!(forward_res, par_forward_res);
    }
}

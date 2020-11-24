use super::super::utility::{
    counter::CounterEst, linalg::mat_mul, linalg::par_mat_mul, math::to_2d_view,
};
use crate::{MLPFLoatRandSampling, MLPFloat, Optimizer, Tensor};
use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct Dense<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    weight_mat: ArrayD<T>, // n1 x n2
}

impl<T> Tensor<T> for Dense<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        mat_mul(&to_2d_view(input), &self.weight_view_2d()).into_dyn()
    }

    fn backward_respect_to_input(
        &self,
        _: ArrayViewD<T>,
        layer_output: ArrayViewD<T>,
    ) -> ArrayD<T> {
        par_mat_mul(&to_2d_view(layer_output), &self.weight_view_2d().t()).into_dyn()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        par_mat_mul(&to_2d_view(input), &self.weight_view_2d()).into_dyn()
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn backward_update(
        &mut self,
        input: ArrayViewD<T>,           // m x n1
        output_gradient: ArrayViewD<T>, // m x n2
        optimizer: &Box<dyn Optimizer<T>>,
    ) {
        let weight_gradient =
            par_mat_mul(&to_2d_view(input).t(), &to_2d_view(output_gradient)).into_dyn();
        optimizer.change_values(&mut self.weight_mat.view_mut(), weight_gradient.view());
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.weight_mat.len())
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.weight_mat.len() * 2 - self.weight_mat.shape()[0])
    }

    fn to_frozen(&self) -> Box<dyn Tensor<T> + Sync> {
        Box::new(Self {
            is_frozen: true,
            weight_mat: self.weight_mat.into_owned(),
        })
    }
}

impl<T> Dense<T>
where
    T: MLPFloat,
{
    pub fn weight_view_2d(&self) -> ArrayView2<T> {
        to_2d_view(self.weight_mat.view())
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

#[macro_export]
macro_rules! dense {
    ($a:expr, $b:expr) => {{
        Box::new(Dense::new_random_uniform($a, $b))
    }};
}

extern crate ndarray;
use super::super::traits::numerical_traits::{MLPFLoatRandSampling, MLPFloat};
use super::super::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::prelude::*;

pub struct Bias<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    bias_arr: ArrayD<T>, // 2D array with shape 1 x N, N = number of neurons
}

impl<T> Tensor<T> for Bias<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        &input + &self.bias_arr.view()
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn updatable_mat(&mut self) -> ArrayViewMutD<'_, T> {
        self.bias_arr.view_mut()
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.bias_arr.len())
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.bias_arr.len())
    }
}

impl<T> Bias<T>
where
    T: MLPFloat + MLPFLoatRandSampling,
{
    pub fn new(size: usize) -> Self {
        Self {
            is_frozen: false,
            bias_arr: Array2::zeros((1, size)).into_dyn(),
        }
    }

    pub fn new_frozen(size: usize) -> Self {
        let mut res = Self::new(size);
        res.is_frozen = true;
        res
    }
}

#[macro_export]
macro_rules! bias {
    ($x:expr) => {{
        crate::traits::tensor_traits::TensorTraitObjWrapper::ForwardParallel(Box::new(Bias::new(
            $x,
        )))
    }};
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;

    use super::super::super::traits::tensor_traits::Tensor;
    use super::Bias;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_bias_forward() {
        let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let bias = Bias::new(2);
        let forward_res = bias.forward(arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(&forward_res, arr);
    }

    #[test]
    fn test_bias_forward_rand() {
        let shape = [1000, 100];
        let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let weights = Bias::new(100);
        let forward_res = weights.forward(rand_arr.view());
        assert_eq!(forward_res.ndim(), 2usize);
        assert_eq!(&forward_res, rand_arr);
    }
}

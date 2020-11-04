extern crate ndarray;
use super::super::traits::numerical_traits::MLPFloat;
use super::super::traits::tensor_traits::Tensor;
use crate::utility::{counter::CounterEst, math::eps};
use ndarray::prelude::*;
use std::cell::RefCell;

pub struct BatchNormalization<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    size: usize,
    moving_mean: RefCell<Option<ArrayD<T>>>,     // n
    moving_variance: RefCell<Option<ArrayD<T>>>, // n
    last_mean: RefCell<Option<ArrayD<T>>>,       // n
    last_variance: RefCell<Option<ArrayD<T>>>,   // n
    moving_batch_size: RefCell<usize>,
    last_batch_size: RefCell<usize>,
    gama: ArrayD<T>, // 1 x n
    beta: ArrayD<T>, // 1 x n
}

impl<T> BatchNormalization<T>
where
    T: MLPFloat,
{
    fn forward_helper(&self, input: ArrayViewD<T>, is_parallel: bool) -> ArrayD<T> {
        *self.last_batch_size.borrow_mut() = input.shape()[0];
        let mean = input.mean_axis(Axis(0)).unwrap();
        let mut variance: ArrayD<T> = (&input - &mean.view()).into_dimensionality().unwrap();
        if is_parallel {
            variance.par_mapv_inplace(|ele| ele * ele);
        } else {
            variance.mapv_inplace(|ele| ele * ele);
        }
        variance = variance.mean_axis(Axis(0)).unwrap();
        self.update_last_mean(mean);
        self.update_last_variance(variance);

        if self.moving_mean.borrow().is_none() {
            *self.moving_batch_size.borrow_mut() = input.shape()[0];
            *self.moving_mean.borrow_mut() =
                Some(self.last_mean.borrow().as_ref().unwrap().clone());
            // Assuming moving variance is None as well.
            *self.moving_variance.borrow_mut() =
                Some(self.last_variance.borrow().as_ref().unwrap().clone());
        }

        let mut std_stable = self.last_variance.borrow().as_ref().unwrap().clone();
        if is_parallel {
            std_stable.par_mapv_inplace(|ele| (ele + eps()).sqrt());
        } else {
            std_stable.mapv_inplace(|ele| (ele + eps()).sqrt());
        }
        let input_normalized =
            (&input - &self.last_mean.borrow().as_ref().unwrap().view()) / &std_stable.view();
        input_normalized * &self.gama + &self.beta
    }
}

impl<T> Tensor<T> for BatchNormalization<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false)
    }

    fn backward_respect_to_input(&self, _: ArrayViewD<T>, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, true)
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate((self.size + 1) * 2)
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.size * 2)
    }
}

impl<T> BatchNormalization<T>
where
    T: MLPFloat,
{
    pub fn new(size: usize) -> Self {
        Self {
            is_frozen: false,
            size,
            moving_mean: RefCell::new(None),
            moving_variance: RefCell::new(None),
            last_mean: RefCell::new(None),
            last_variance: RefCell::new(None),
            moving_batch_size: RefCell::new(0),
            last_batch_size: RefCell::new(0),
            gama: Array2::ones((1, size)).into_dyn(),
            beta: Array2::zeros((1, size)).into_dyn(),
        }
    }

    pub fn new_frozen(size: usize) -> Self {
        let mut res = Self::new(size);
        res.is_frozen = true;
        res
    }

    fn update_last_mean(&self, mean: ArrayD<T>) {
        assert_eq!(mean.len(), self.size);
        *self.last_mean.borrow_mut() = Some(mean);
    }

    fn update_last_variance(&self, variance: ArrayD<T>) {
        assert_eq!(variance.len(), self.size);
        *self.last_variance.borrow_mut() = Some(variance);
    }
}

#[macro_export]
macro_rules! batch_norm {
    ($x:expr) => {{
        Box::new(BatchNormalization::new($x))
    }};
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::super::super::traits::tensor_traits::Tensor;
    use super::BatchNormalization;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_batch_norm_forward_random_arr() {
        let shape = [10, 5];
        let rand_arr = Array2::random(shape, Uniform::new(-10., 10.)).into_dyn();
        let batch_norm = BatchNormalization::new(5);
        let forward_res = batch_norm.forward(rand_arr.view());
        assert_eq!(forward_res.shape(), &shape);
    }

    #[test]
    fn test_batch_norm_forward_consistency() {
        let arr_1 = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let arr_2 = &arr2(&[[1.1, -2.], [-1.3, 2.1], [100., 0.5]]).into_dyn();
        let batch_norm_1 = BatchNormalization::new(2);
        let forward_res_1 = batch_norm_1.forward(arr_1.view());
        let batch_norm_2 = BatchNormalization::new(2);
        let forward_res_2 = batch_norm_2.forward(arr_2.view());
        assert_eq!(
            forward_res_1.index_axis(Axis(1), 1), // forward_res_1[:, 1]
            forward_res_2.index_axis(Axis(1), 1)  // forward_res_2[:, 1]
        );
    }
}

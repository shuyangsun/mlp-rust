use crate::utility::{counter::CounterEst, math::calculate_std_from_variance};
use crate::{MLPFloat, Tensor};
use ndarray::{Array2, ArrayD, ArrayViewD, Axis};
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

#[derive(Clone)]
struct BatchNormalizationFrozen<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    size: usize,
    mean: ArrayD<T>,       // n
    std_stable: ArrayD<T>, // n
    gama: ArrayD<T>,       // 1 x n
    beta: ArrayD<T>,       // 1 x n
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
            variance.par_mapv_inplace(|ele| ele.powi(2));
        } else {
            variance.mapv_inplace(|ele| ele.powi(2));
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

        let std_stable = calculate_std_from_variance(
            &self.last_variance.borrow().as_ref().unwrap(),
            is_parallel,
        );
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

    fn to_frozen(&self) -> Box<dyn Tensor<T> + Sync> {
        let mean = if self.moving_mean.borrow().as_ref().is_some() {
            self.moving_mean.borrow().as_ref().unwrap().clone()
        } else {
            ArrayD::zeros(vec![self.size])
        };
        let std_stable = if self.moving_variance.borrow().as_ref().is_some() {
            calculate_std_from_variance(self.moving_variance.borrow().as_ref().unwrap(), true)
        } else {
            ArrayD::ones(vec![self.size])
        };
        Box::new(BatchNormalizationFrozen {
            is_frozen: true,
            size: self.size,
            mean,
            std_stable,
            gama: self.gama.clone(),
            beta: self.beta.clone(), // 1 x n
        })
    }
}

impl<T> Tensor<T> for BatchNormalizationFrozen<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        let input_normalized = (&input - &self.mean.view()) / &self.std_stable.view();
        input_normalized * &self.gama + &self.beta
    }

    fn backward_respect_to_input(&self, _: ArrayViewD<T>, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn is_frozen(&self) -> bool {
        true
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate((self.size + 1) * 2)
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.size * 2)
    }

    fn to_frozen(&self) -> Box<dyn Tensor<T> + Sync> {
        Box::new(self.clone())
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

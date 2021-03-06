use crate::utility::{counter::CounterEst, math::calculate_std_from_variance};
use crate::{MLPFloat, Tensor};
use ndarray::{ArrayD, ArrayViewD, Axis};
use std::cell::RefCell;

pub struct InputNormalization<T>
where
    T: MLPFloat,
{
    size: usize,
    moving_min: RefCell<Option<ArrayD<T>>>,      // n
    moving_variance: RefCell<Option<ArrayD<T>>>, // n
    last_min: RefCell<Option<ArrayD<T>>>,        // n
    last_variance: RefCell<Option<ArrayD<T>>>,   // n
    moving_batch_size: RefCell<usize>,
    last_batch_size: RefCell<usize>,
}

#[derive(Clone)]
struct InputNormalizationFrozen<T>
where
    T: MLPFloat,
{
    size: usize,
    min: ArrayD<T>,
    std_stable: ArrayD<T>,
}

impl<T> InputNormalization<T>
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
        self.update_last_min(mean);
        self.update_last_variance(variance);

        if self.moving_min.borrow().is_none() {
            *self.moving_batch_size.borrow_mut() = input.shape()[0];
            *self.moving_min.borrow_mut() = Some(self.last_min.borrow().as_ref().unwrap().clone());
            // Assuming moving variance is None as well.
            *self.moving_variance.borrow_mut() =
                Some(self.last_variance.borrow().as_ref().unwrap().clone());
        }

        let std_stable = calculate_std_from_variance(
            &self.last_variance.borrow().as_ref().unwrap(),
            is_parallel,
        );
        (&input - &self.last_min.borrow().as_ref().unwrap().view()) / &std_stable.view()
    }
}

impl<T> Tensor<T> for InputNormalization<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false)
    }

    fn backward_respect_to_input(&self, _: ArrayViewD<T>, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!() // Should not be ran since it's only on the input layer.
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, true)
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate((self.size + 1) * 2)
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.size * 2)
    }

    fn to_frozen(&self) -> Box<dyn Tensor<T> + Sync> {
        let min = if self.moving_min.borrow().as_ref().is_some() {
            self.moving_min.borrow().as_ref().unwrap().clone()
        } else {
            ArrayD::zeros(vec![self.size])
        };
        let std_stable = if self.moving_variance.borrow().as_ref().is_some() {
            calculate_std_from_variance(self.moving_variance.borrow().as_ref().unwrap(), true)
        } else {
            ArrayD::ones(vec![self.size])
        };

        Box::new(InputNormalizationFrozen {
            size: self.size,
            min,
            std_stable,
        })
    }
}

impl<T> Tensor<T> for InputNormalizationFrozen<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        (&input - &self.min.view()) / &self.std_stable.view()
    }

    fn backward_respect_to_input(&self, _: ArrayViewD<T>, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!() // Should not be ran since it's only on the input layer.
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

impl<T> InputNormalization<T>
where
    T: MLPFloat,
{
    pub fn new(size: usize) -> Self {
        Self {
            size,
            moving_min: RefCell::new(None),
            moving_variance: RefCell::new(None),
            last_min: RefCell::new(None),
            last_variance: RefCell::new(None),
            moving_batch_size: RefCell::new(0),
            last_batch_size: RefCell::new(0),
        }
    }

    fn update_last_min(&self, mean: ArrayD<T>) {
        assert_eq!(mean.len(), self.size);
        *self.last_min.borrow_mut() = Some(mean);
    }

    fn update_last_variance(&self, variance: ArrayD<T>) {
        assert_eq!(variance.len(), self.size);
        *self.last_variance.borrow_mut() = Some(variance);
    }
}

#[macro_export]
macro_rules! input_norm {
    ($x:expr) => {{
        Box::new(InputNormalization::new($x))
    }};
}

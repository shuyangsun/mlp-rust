use crate::utility::{
    array::add_1_dim_to_shape, counter::CounterEst, math::calculate_std_from_variance,
};
use crate::{MLPFloat, Tensor};
use ndarray::{Array, ArrayView, Axis, Dimension, Shape, ShapeBuilder};
use std::cell::RefCell;

pub struct InputNormalization<T, D>
where
    T: MLPFloat,
{
    shape: Shape<D>,
    moving_min: RefCell<Option<Array<T, D>>>,
    moving_variance: RefCell<Option<Array<T, D>>>,
    last_min: RefCell<Option<Array<T, D>>>,
    last_variance: RefCell<Option<Array<T, D>>>,
    moving_batch_size: RefCell<usize>,
    last_batch_size: RefCell<usize>,
}

impl<T, D> InputNormalization<T, D>
where
    T: MLPFloat,
{
    fn forward_helper(&self, input: ArrayView<T, D>, is_parallel: bool) -> Array<T, D> {
        *self.last_batch_size.borrow_mut() = input.shape()[0];
        let mean = input.mean_axis(Axis(0)).unwrap();
        let mut variance: Array<T, D> = (&input - &mean.view()).into_dimensionality().unwrap();
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
            *self.moving_min.borrow_mut() =
                Some(self.last_min.borrow().as_ref().unwrap().into_owned());
            // Assuming moving variance is None as well.
            *self.moving_variance.borrow_mut() =
                Some(self.last_variance.borrow().as_ref().unwrap().into_owned());
        }

        let std_stable = calculate_std_from_variance(
            &self.last_variance.borrow().as_ref().unwrap(),
            is_parallel,
        );
        (&input - &self.last_min.borrow().as_ref().unwrap().view()) / &std_stable.view()
    }
}

impl<T, D> Tensor<T, D, D> for InputNormalization<T, D>
where
    D: 'static,
    T: MLPFloat,
{
    fn forward(&self, input: ArrayView<T, D>) -> Array<T, D> {
        self.forward_helper(input, false)
    }

    fn par_forward(&self, input: ArrayView<T, D>) -> Array<T, D> {
        self.forward_helper(input, true)
    }

    fn backward_respect_to_input(&self, _: ArrayView<T, D>, _: ArrayView<T, D>) -> Array<T, D> {
        unimplemented!() // Should not be ran since it's only on the input layer.
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate((self.size + 1) * 2)
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.size * 2)
    }
}

impl<T, D> InputNormalization<T, D>
where
    T: MLPFloat,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        D: Dimension,
        Sh: ShapeBuilder<Dim = D::Smaller>,
    {
        let new_shape = add_1_dim_to_shape(shape);
        assert_eq!(new_shape.size(), shape.size());
        Self {
            shape: new_shape,
            moving_min: RefCell::new(None),
            moving_variance: RefCell::new(None),
            last_min: RefCell::new(None),
            last_variance: RefCell::new(None),
            moving_batch_size: RefCell::new(0),
            last_batch_size: RefCell::new(0),
        }
    }

    fn update_last_min(&self, mean: Array<T, D>) {
        assert_eq!(mean.len(), self.size);
        *self.last_min.borrow_mut() = Some(mean);
    }

    fn update_last_variance(&self, variance: Array<T, D>) {
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

use crate::traits::{numerical_traits::MLPFloat, optimizer_traits::Optimizer};
use ndarray::prelude::ArrayViewMut;

pub struct GradientDescent<T>
where
    T: MLPFloat,
{
    learning_rate: T,
}

impl<T> GradientDescent<T>
where
    T: MLPFloat,
{
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

impl<T, D> Optimizer<T, D> for GradientDescent<T>
where
    T: MLPFloat,
{
    fn modify_inplace(&self, gradient: &mut ArrayViewMut<T, D>) {
        gradient.par_map_inplace(|ele| *ele = *ele * self.learning_rate);
    }

    // TODO: uncomment to test performance
    // fn modify<D>(&self, gradient: ArrayView<T, D>) -> Array<T, D> {
    //     gradient.mapv(|ele| ele * self.learning_rate)
    // }
}

#[macro_export]
macro_rules! gradient_descent {
    ($x:expr) => {{
        Box::new(crate::optimizer::gradient_descent::GradientDescent::new($x))
    }};
}

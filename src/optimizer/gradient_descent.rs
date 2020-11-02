use crate::traits::{numerical_traits::MLPFloat, optimizer_traits::Optimizer};
extern crate ndarray;
use ndarray::prelude::*;
use rayon::prelude::*;

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

impl<T> Optimizer<T> for GradientDescent<T>
where
    T: MLPFloat,
{
    fn modify_inplace(&self, gradient: &mut ArrayViewMutD<'_, T>) {
        gradient.par_map_inplace(|ele| *ele = *ele * self.learning_rate);
    }

    // fn modify(&self, gradient: ArrayViewD<'_, T>) -> ArrayD<T> {
    //     gradient.mapv(|ele| ele * self.learning_rate)
    // }
}

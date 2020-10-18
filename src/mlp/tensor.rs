extern crate ndarray;
use super::type_def::MLPFloat;
use ndarray::prelude::*;

pub trait Tensor<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: &Array2<T>) -> Array2<T>;
    fn backward_batch(&self, output: &Array2<T>) -> Array2<T>;

    fn backward(&self, output: &Array2<T>) -> Array1<T> {
        self.backward_batch(output).mean_axis(Axis(0)).unwrap()
    }
}

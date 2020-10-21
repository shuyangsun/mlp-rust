extern crate ndarray;
use super::numerical_traits::MLPFloat;
use ndarray::prelude::*;

pub trait Tensor<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> Box<ArrayD<T>>;
    fn backward_batch(&self, output: ArrayViewD<T>) -> Box<ArrayD<T>>;

    fn backward(&self, output: ArrayViewD<T>) -> Box<ArrayD<T>> {
        Box::new(
            self.backward_batch(output)
                .as_ref()
                .mean_axis(Axis(0))
                .unwrap(),
        )
    }
}

pub trait TensorUpdatable<T>
where
    T: MLPFloat,
{
    fn is_frozen(&self) -> bool {
        false
    }
    fn updatable_mat(&mut self) -> ArrayViewMutD<T>;
    fn update(&mut self, gradient_last_layer: ArrayViewD<T>) {
        if self.is_frozen() {
            return;
        }
        let mut original_mat = self.updatable_mat();
        let update_res: ArrayD<T> = &original_mat - &gradient_last_layer;
        original_mat.assign(&update_res);
    }
}

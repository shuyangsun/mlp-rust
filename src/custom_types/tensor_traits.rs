extern crate ndarray;
use super::numerical_traits::MLPFloat;
use ndarray::prelude::*;

pub trait TensorComputable<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T>;
    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T>;

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward(input)
    }

    fn backward(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        self.backward_batch(output).mean_axis(Axis(0)).unwrap()
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
        assert_eq!(update_res.shape(), original_mat.shape());
        original_mat.assign(&update_res);
    }
}

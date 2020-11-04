use crate::traits::optimizer_traits::Optimizer;
use ndarray::{ArrayD, ArrayViewD};

pub trait Model<T> {
    fn train(
        &mut self,
        max_num_iter: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
    );

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T>;

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.predict(input)
    }
}

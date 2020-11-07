use crate::{DataSet, Optimizer};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub trait Model<T> {
    fn train(
        &mut self,
        data: &mut Box<dyn DataSet<T, IxDyn>>,
        batch_size: usize,
        max_num_epoch: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        should_print: bool,
    );

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T>;

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.predict(input)
    }
}

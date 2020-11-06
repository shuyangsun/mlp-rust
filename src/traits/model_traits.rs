use crate::{DataSet, Optimizer};
use ndarray::{Array, ArrayView};

pub trait Model<T, InputD, OutputD> {
    fn train<'data, 'model>(
        &'model mut self,
        data: &'data mut Box<dyn DataSet<'data, T, InputD, OutputD>>,
        max_num_epoch: usize,
        batch_size: usize,
        optimizer: &Box<dyn Optimizer<T, InputD>>,
        should_print_loss: bool,
    ) where
        'data: 'model;

    fn predict(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD>;

    fn par_predict(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD> {
        self.predict(input)
    }
}

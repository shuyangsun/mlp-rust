use crate::data_set::utility::{DataBatch, InputOutputData};

pub trait DataSet<'data, T, D>
where
    T: 'static,
{
    fn next_training_batch(&'data self, batch_size: usize) -> DataBatch<'data, T, D>;

    fn test_data(&'data self) -> InputOutputData<'data, T, D>;

    fn num_samples(&self) -> usize;

    fn num_training_samples(&self) -> usize;

    fn num_test_samples(&self) -> usize {
        self.num_samples() - self.num_training_samples()
    }
}

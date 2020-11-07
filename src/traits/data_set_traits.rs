use crate::data_set::utility::{DataBatch, InputOutputData};

pub trait DataSet<'dset, 'dview, T, D>
where
    T: 'static,
    'dset: 'dview,
{
    fn next_train_batch(&'dset self, batch_size: usize) -> DataBatch<'dview, T, D>;

    fn train_data(&'dset self) -> InputOutputData<'dview, T, D>;
    fn test_data(&'dset self) -> InputOutputData<'dview, T, D>;

    fn num_samples(&self) -> usize;

    fn num_training_samples(&self) -> usize;

    fn shuffle_all(&mut self);

    fn shuffle_train(&mut self);

    fn num_test_samples(&self) -> usize {
        self.num_samples() - self.num_training_samples()
    }
}

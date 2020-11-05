use crate::InputOutputData;

pub trait DataSet<'a, T, D> {
    fn next_training_batch(&'a self) -> InputOutputData<'a, T, D>;

    fn test_data(&'a self) -> InputOutputData<'a, T, D>;

    fn num_samples(&self) -> usize;

    fn num_training_samples(&self) -> usize;

    fn num_test_samples(&self) -> usize {
        self.num_samples() - self.num_test_samples()
    }
}

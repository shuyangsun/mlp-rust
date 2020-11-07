use ndarray::ArrayViewD;

pub trait DataSet<T, D> {
    fn next_train_batch(&self, batch_size: usize) -> Vec<(ArrayViewD<T>, ArrayViewD<T>)>;

    fn train_data(&self) -> (ArrayViewD<T>, ArrayViewD<T>);
    fn test_data(&self) -> (ArrayViewD<T>, ArrayViewD<T>);

    fn num_samples(&self) -> usize;

    fn num_training_samples(&self) -> usize;

    fn shuffle_all(&mut self);

    fn shuffle_train(&mut self);

    fn num_test_samples(&self) -> usize {
        self.num_samples() - self.num_training_samples()
    }
}

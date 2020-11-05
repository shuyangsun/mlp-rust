use ndarray::ArrayViewD;

pub struct InputOutputData<'a, T> {
    pub input: ArrayViewD<'a, T>,
    pub output: ArrayViewD<'a, T>,
}

impl<'a, T> InputOutputData<'a, T> {
    pub fn new(input: ArrayViewD<'a, T>, output: ArrayViewD<'a, T>) -> Self {
        assert_eq!(input.ndim(), output.ndim());
        assert_eq!(input.shape()[0], output.shape()[0]);
        Self { input, output }
    }

    pub fn num_samples(&self) -> usize {
        self.input.shape()[0]
    }
}

pub trait DataSet<'a, T> {
    fn next_training_batch(&'a self) -> InputOutputData<'a, T>;

    fn test_data(&'a self) -> InputOutputData<'a, T>;

    fn num_samples(&self) -> usize;

    fn num_training_samples(&self) -> usize;

    fn num_test_samples(&self) -> usize {
        self.num_samples() - self.num_test_samples()
    }
}

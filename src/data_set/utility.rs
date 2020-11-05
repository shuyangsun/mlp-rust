use ndarray::{ArrayView, Dimension};

pub struct InputOutputData<'a, T, D> {
    pub input: ArrayView<'a, T, D>,
    pub output: ArrayView<'a, T, D>,
}

impl<'a, T, D> InputOutputData<'a, T, D>
where
    D: Dimension,
{
    pub fn new(input: ArrayView<'a, T, D>, output: ArrayView<'a, T, D>) -> Self {
        assert_eq!(input.ndim(), output.ndim());
        assert_eq!(input.shape()[0], output.shape()[0]);
        Self { input, output }
    }

    pub fn num_samples(&self) -> usize {
        self.input.shape()[0]
    }
}

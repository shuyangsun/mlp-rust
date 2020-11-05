use crate::MLPFloat;
use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis, Slice};
use rand::{thread_rng, Rng};

pub struct InputOutputData<'a, T, InputD, OutputD> {
    pub input: ArrayView<'a, T, InputD>,
    pub output: ArrayView<'a, T, OutputD>,
}

impl<'a, T, InputD, OutputD> InputOutputData<'a, T, InputD, OutputD>
where
    InputD: Dimension,
    OutputD: Dimension,
{
    pub fn new(input: ArrayView<'a, T, InputD>, output: ArrayView<'a, T, OutputD>) -> Self {
        assert_eq!(input.shape()[0], output.shape()[0]);
        Self { input, output }
    }

    pub fn num_samples(&self) -> usize {
        self.input.shape()[0]
    }
}

pub struct DataBatch<'data, T, InputD, OutputD> {
    cur_start_idx: usize,
    batch_size: usize,
    data_input: &'data Array<T, InputD>,
    data_output: &'data Array<T, OutputD>,
}

impl<'data, T, InputD, OutputD> DataBatch<'data, T, InputD, OutputD>
where
    InputD: Dimension,
    OutputD: Dimension,
{
    pub fn new(
        input: &'data Array<T, InputD>,
        output: &'data Array<T, OutputD>,
        batch_size: usize,
    ) -> Self {
        assert_eq!(
            input.shape()[0],
            output.shape()[0],
            "Input and output data have different sizes on axis 0."
        );
        Self {
            cur_start_idx: 0,
            batch_size,
            data_input: input,
            data_output: output,
        }
    }
}

impl<'data, T, InputD, OutputD> Iterator for DataBatch<'data, T, InputD, OutputD>
where
    InputD: Dimension,
    OutputD: Dimension,
{
    type Item = InputOutputData<'data, T, InputD, OutputD>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_samples = self.data_input.shape()[0];
        if self.cur_start_idx >= n_samples {
            return None;
        }
        let start_row_idx = self.cur_start_idx;
        let end_row_idx = std::cmp::min(self.cur_start_idx + self.batch_size, n_samples);
        let input: ArrayView<'data, T, InputD> = self
            .data_input
            .slice_axis(Axis(0), Slice::from(start_row_idx..end_row_idx));
        let output: ArrayView<'data, T, OutputD> = self
            .data_output
            .slice_axis(Axis(0), Slice::from(start_row_idx..end_row_idx));
        self.cur_start_idx = end_row_idx;
        Some(InputOutputData::<'data, T, InputD, OutputD>::new(
            input, output,
        ))
    }
}

pub fn generate_arg_shuffle_indices(range: std::ops::Range<usize>) -> (Vec<usize>, Vec<usize>) {
    let size = range.end - range.start;
    let mut rng = thread_rng();
    let a = (0..size)
        .map(|_| rng.gen_range(range.start, range.end))
        .collect();
    let b = (0..size)
        .map(|_| rng.gen_range(range.start, range.end))
        .collect();
    (a, b)
}

pub fn shuffle_array<T, D>(arr: &mut Array<T, D>, arg_shuffle_indices: (&Vec<usize>, &Vec<usize>))
where
    D: RemoveAxis,
    T: MLPFloat,
{
    for (idx_a, idx_b) in arg_shuffle_indices
        .0
        .iter()
        .zip(arg_shuffle_indices.1.iter())
    {
        let (idx_a, idx_b) = (idx_a.clone(), idx_b.clone());
        let first_clone = arr.index_axis(Axis(0), idx_a).into_owned();
        let second_clone = arr.index_axis(Axis(0), idx_b).into_owned();
        let mut first_row = arr.index_axis_mut(Axis(0), idx_a);
        first_row.assign(&second_clone);
        let mut second_row = arr.index_axis_mut(Axis(0), idx_b);
        second_row.assign(&first_clone);
    }
}

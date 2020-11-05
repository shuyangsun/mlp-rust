use self::super::utility::{generate_arg_shuffle_indices, shuffle_array, DataBatch};
use crate::{DataSet, InputOutputData, MLPFloat};
use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis, Slice};

pub struct DataSetInMemory<T, InputD, OutputD> {
    input_data: Array<T, InputD>,
    output_data: Array<T, OutputD>,
    training_sample_size: usize,
}

impl<T, InputD, OutputD> DataSetInMemory<T, InputD, OutputD>
where
    InputD: RemoveAxis,
    OutputD: RemoveAxis,
    T: MLPFloat,
{
    pub fn new(
        input_data: Array<T, InputD>,
        output_data: Array<T, OutputD>,
        test_data_ratio: f64,
        should_shuffle: bool,
    ) -> Self {
        assert_eq!(input_data.shape()[0], output_data.shape()[0]);
        assert!(test_data_ratio >= 0.);
        assert!(test_data_ratio <= 1.);
        let mut res = Self {
            input_data,
            output_data,
            training_sample_size: 0,
        };
        res.update_test_data_ratio(test_data_ratio);
        if should_shuffle {
            res.shuffle_all()
        }
        res
    }

    pub fn shuffle_all(&mut self) {
        let (a, b) = generate_arg_shuffle_indices(0..self.num_samples());
        shuffle_array(&mut self.input_data, (&a, &b));
        shuffle_array(&mut self.output_data, (&a, &b));
    }

    pub fn shuffle_train(&mut self) {
        let (a, b) = generate_arg_shuffle_indices(0..self.num_training_samples());
        shuffle_array(&mut self.input_data, (&a, &b));
        shuffle_array(&mut self.output_data, (&a, &b));
    }

    pub fn update_test_data_ratio(&mut self, test_data_ratio: f64) {
        self.training_sample_size =
            (self.input_data.shape()[0] as f64 * (1. - test_data_ratio)).floor() as usize;
    }
}

impl<'data, T, InputD, OutputD> DataSet<'data, T, InputD, OutputD>
    for DataSetInMemory<T, InputD, OutputD>
where
    T: 'static,
    InputD: Dimension,
    OutputD: Dimension,
{
    fn next_train_batch(&'data self, batch_size: usize) -> DataBatch<'data, T, InputD, OutputD> {
        DataBatch::new(&self.input_data, &self.output_data, batch_size)
    }

    fn train_data(&'data self) -> InputOutputData<'data, T, InputD, OutputD> {
        let input: ArrayView<'data, T, InputD> = self
            .input_data
            .slice_axis(Axis(0), Slice::from(..self.num_training_samples()));
        let output: ArrayView<'data, T, OutputD> = self
            .output_data
            .slice_axis(Axis(0), Slice::from(..self.num_training_samples()));
        InputOutputData::<'data, T, InputD, OutputD>::new(input, output)
    }

    fn test_data(&'data self) -> InputOutputData<'data, T, InputD, OutputD> {
        let input: ArrayView<'data, T, InputD> = self
            .input_data
            .slice_axis(Axis(0), Slice::from(self.num_training_samples()..));
        let output: ArrayView<'data, T, OutputD> = self
            .output_data
            .slice_axis(Axis(0), Slice::from(self.num_training_samples()..));
        InputOutputData::<'data, T, InputD, OutputD>::new(input, output)
    }

    fn num_samples(&self) -> usize {
        self.input_data.shape()[0]
    }

    fn num_training_samples(&self) -> usize {
        self.training_sample_size
    }
}

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    use crate::DataSet;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_data_set_1() {
        let num_samples = 997usize;
        let input_size = 8;
        let output_size = 2;
        let input_data = Array2::random((num_samples, input_size), Uniform::new(-1., 1.));
        let output_data = Array2::random((num_samples, output_size), Uniform::new(-1., 1.));
        let dataset = DataSetInMemory::new(input_data, output_data, 0.4, true);
        assert_eq!(dataset.num_samples(), num_samples);
        assert_eq!(dataset.num_training_samples(), 598);
        assert_eq!(dataset.num_test_samples(), 399);
        let test_data = dataset.test_data();
        assert_eq!(test_data.input.shape()[0], 399);
        assert_eq!(test_data.output.shape()[0], test_data.input.shape()[0]);

        let batch_size = 100usize;
        let mut total_sample = 0usize;
        let mut counter = 0usize;
        for batch in dataset.next_train_batch(batch_size) {
            if counter > 10 {
                break;
            }
            assert_eq!(batch.input.shape()[0], batch.output.shape()[0]);
            assert!(batch.input.shape()[0] <= batch_size);
            total_sample += batch.input.shape()[0];
            counter += 1;
        }
        assert_eq!(total_sample, num_samples);
    }

    #[test]
    fn test_data_set_shuffle() {
        let num_samples = 997usize;
        let input_size = 8;
        let output_size = 2;
        let input_data = Array2::random((num_samples, input_size), Uniform::new(-1., 1.));
        let output_data = Array2::random((num_samples, output_size), Uniform::new(-1., 1.));
        let mut dataset = DataSetInMemory::new(input_data, output_data, 0.4, true);
        let (train_1_input, train_1_output) = (
            dataset.train_data().input.into_owned(),
            dataset.train_data().output.into_owned(),
        );
        let (test_1_input, test_1_output) = (
            dataset.test_data().input.into_owned(),
            dataset.test_data().output.into_owned(),
        );
        dataset.shuffle_train();
        let (train_2_input, train_2_output) = (
            dataset.train_data().input.into_owned(),
            dataset.train_data().output.into_owned(),
        );
        let (test_2_input, test_2_output) = (
            dataset.test_data().input.into_owned(),
            dataset.test_data().output.into_owned(),
        );
        assert_ne!(train_1_input, train_2_input);
        assert_ne!(train_1_output, train_2_output);
        assert_eq!(test_1_input, test_2_input);
        assert_eq!(test_1_output, test_2_output);
        dataset.shuffle_all();
        let (train_3_input, train_3_output) = (
            dataset.train_data().input.into_owned(),
            dataset.train_data().output.into_owned(),
        );
        let (test_3_input, test_3_output) = (
            dataset.test_data().input.into_owned(),
            dataset.test_data().output.into_owned(),
        );
        assert_ne!(train_3_input, train_2_input);
        assert_ne!(train_3_output, train_2_output);
        assert_ne!(test_3_input, test_2_input);
        assert_ne!(test_3_output, test_2_output);
    }
}

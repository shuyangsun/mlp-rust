use crate::utility::math::{shuffle_array, shuffle_array_within_range};
use crate::{DataSet, MLPFloat};
use ndarray::{s, Array, ArrayD, ArrayViewD, Dimension, RemoveAxis};
use std::marker::PhantomData;

pub struct DataSetInMemory<T, D> {
    data: ArrayD<T>,
    output_size: usize,
    training_sample_size: usize,
    _phantom: PhantomData<*const D>,
}

impl<T, D> DataSetInMemory<T, D>
where
    D: RemoveAxis,
    T: MLPFloat,
{
    pub fn new(
        data: Array<T, D>,
        output_size: usize,
        test_data_ratio: f64,
        should_shuffle: bool,
    ) -> Self {
        assert!(test_data_ratio >= 0.);
        assert!(test_data_ratio <= 1.);
        let mut data = data;
        if should_shuffle {
            shuffle_array(&mut data);
        }
        let mut res = Self {
            data: data.into_dyn(),
            output_size,
            training_sample_size: 0,
            _phantom: PhantomData,
        };
        res.update_test_data_ratio(test_data_ratio);
        res
    }

    pub fn update_test_data_ratio(&mut self, test_data_ratio: f64) {
        self.training_sample_size =
            (self.data.shape()[0] as f64 * (1. - test_data_ratio)).floor() as usize;
    }
}

impl<T, D> DataSet<T, D> for DataSetInMemory<T, D>
where
    T: MLPFloat,
    D: Dimension,
{
    fn next_train_batch(&self, batch_size: usize) -> Vec<(ArrayViewD<T>, ArrayViewD<T>)> {
        let mut res = Vec::new();
        let (n_samples, train_cols) = (
            self.train_data().0.shape()[0],
            self.train_data().0.shape()[1],
        );
        let mut cur_start_idx = 0usize;
        while cur_start_idx < n_samples {
            let start_row_idx = cur_start_idx;
            let end_row_idx = std::cmp::min(cur_start_idx + batch_size, n_samples);
            let input = self
                .data
                .slice(s![start_row_idx..end_row_idx, ..train_cols])
                .into_dyn();
            let output = self
                .data
                .slice(s![start_row_idx..end_row_idx, train_cols..])
                .into_dyn();
            res.push((input, output));
            cur_start_idx = end_row_idx;
        }
        res
    }

    fn train_data(&self) -> (ArrayViewD<T>, ArrayViewD<T>) {
        let input = self
            .data
            .slice(s![
                ..self.num_training_samples(),
                ..self.data.shape()[1] - self.output_size
            ])
            .into_dyn();
        let output = self
            .data
            .slice(s![
                ..self.num_training_samples(),
                self.data.shape()[1] - self.output_size..
            ])
            .into_dyn();
        (input, output)
    }

    fn test_data(&self) -> (ArrayViewD<T>, ArrayViewD<T>) {
        let input = self
            .data
            .slice(s![
                self.num_training_samples()..self.num_samples(),
                ..self.data.shape()[1] - self.output_size
            ])
            .into_dyn();
        let output = self
            .data
            .slice(s![
                self.num_training_samples()..self.num_samples(),
                self.data.shape()[1] - self.output_size..
            ])
            .into_dyn();
        (input, output)
    }

    fn num_samples(&self) -> usize {
        self.data.shape()[0]
    }

    fn num_training_samples(&self) -> usize {
        self.training_sample_size
    }

    fn shuffle_all(&mut self) {
        shuffle_array(&mut self.data)
    }

    fn shuffle_train(&mut self) {
        shuffle_array_within_range(&mut self.data, 0..self.training_sample_size)
    }
}

use crate::utility::math::shuffle_array;
use crate::{DataSet, InputOutputData, MLPFloat};
use ndarray::prelude::*;

pub struct DataSet2D<T> {
    data: ArrayD<T>,
    output_size: usize,
    num_train_sample: usize,
}

impl<T> DataSet2D<T>
where
    T: MLPFloat,
{
    pub fn new(
        data: ArrayD<T>,
        output_size: usize,
        test_data_ratio: f32,
        should_shuffle: bool,
    ) -> Self {
        let mut data = data;
        if should_shuffle {
            shuffle_array(&mut data);
        }
        let mut res = Self {
            data,
            output_size,
            num_train_sample: 0,
        };
        res.update_test_data_ratio(test_data_ratio);
        res
    }

    pub fn shuffle(&mut self) {
        shuffle_array(&mut self.data)
    }

    pub fn update_test_data_ratio(&mut self, test_data_ratio: f32) {
        self.num_train_sample = ((self.data.shape()[0] as f32) * test_data_ratio).floor() as usize
    }
}

impl<'a, T> DataSet<'a, T> for DataSet2D<T> {
    fn next_training_batch(&'a self) -> InputOutputData<'a, T> {
        unimplemented!()
    }

    fn test_data(&'a self) -> InputOutputData<'a, T> {
        let input: ArrayViewD<'a, T> = self
            .data
            .slice(s![
                self.num_train_sample..self.num_samples(),
                ..self.data.shape()[1] - self.output_size
            ])
            .into_dyn();
        let output: ArrayViewD<'a, T> = self
            .data
            .slice(s![
                self.num_train_sample..self.num_samples(),
                self.data.shape()[1] - self.output_size..
            ])
            .into_dyn();
        InputOutputData::<'a, T>::new(input, output)
    }

    fn num_samples(&self) -> usize {
        self.data.shape()[0]
    }

    fn num_training_samples(&self) -> usize {
        self.num_train_sample
    }
}

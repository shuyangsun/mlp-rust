use crate::utility::math::shuffle_array;
use crate::{DataSet, InputOutputData, MLPFloat};
use ndarray::{s, Array, ArrayD, ArrayView, Dimension, RemoveAxis, Slice, SliceInfo, SliceOrIndex};
use std::marker::PhantomData;

pub struct DataSetInMemory<T, D> {
    data: ArrayD<T>,
    output_size: usize,
    num_train_sample: usize,
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
        test_data_ratio: f32,
        should_shuffle: bool,
    ) -> Self {
        let mut data = data;
        if should_shuffle {
            shuffle_array(&mut data);
        }
        let mut res = Self {
            data: data.into_dyn(),
            output_size,
            num_train_sample: 0,
            _phantom: PhantomData,
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

impl<'a, T, D> DataSet<'a, T, D> for DataSetInMemory<T, D>
where
    D: Dimension + RemoveAxis,
{
    fn next_training_batch(&'a self) -> InputOutputData<'a, T, D> {
        unimplemented!()
    }

    fn test_data(&'a self) -> InputOutputData<'a, T, D> {
        let input: ArrayView<'a, T, D> = self
            .data
            .slice(s![
                self.num_train_sample..self.num_samples(),
                ..self.data.shape()[1] - self.output_size
            ])
            .into_dimensionality::<D>()
            .unwrap();
        let output: ArrayView<'a, T, D> = self
            .data
            .slice(s![
                self.num_train_sample..self.num_samples(),
                self.data.shape()[1] - self.output_size..
            ])
            .into_dimensionality::<D>()
            .unwrap();
        InputOutputData::<'a, T, D>::new(input, output)
    }

    fn num_samples(&self) -> usize {
        self.data.shape()[0]
    }

    fn num_training_samples(&self) -> usize {
        self.num_train_sample
    }
}

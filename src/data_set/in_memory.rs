use crate::data_set::utility::DataBatch;
use crate::utility::math::shuffle_array;
use crate::{DataSet, InputOutputData, MLPFloat};
use ndarray::{s, Array, ArrayD, ArrayView, Dimension, RemoveAxis};
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

    pub fn shuffle(&mut self) {
        shuffle_array(&mut self.data)
    }

    pub fn update_test_data_ratio(&mut self, test_data_ratio: f64) {
        self.training_sample_size =
            (self.data.shape()[0] as f64 * (1. - test_data_ratio)).floor() as usize;
    }
}

impl<'data, T, D> DataSet<'data, T, D> for DataSetInMemory<T, D>
where
    T: 'static,
    D: Dimension,
{
    fn next_training_batch(&'data self, batch_size: usize) -> DataBatch<'data, T, D> {
        DataBatch::new(&self.data, batch_size, self.output_size)
    }

    fn test_data(&'data self) -> InputOutputData<'data, T, D> {
        let input: ArrayView<'data, T, D> = self
            .data
            .slice(s![
                self.num_training_samples()..self.num_samples(),
                ..self.data.shape()[1] - self.output_size
            ])
            .into_dimensionality::<D>()
            .unwrap();
        let output: ArrayView<'data, T, D> = self
            .data
            .slice(s![
                self.num_training_samples()..self.num_samples(),
                self.data.shape()[1] - self.output_size..
            ])
            .into_dimensionality::<D>()
            .unwrap();
        InputOutputData::<'data, T, D>::new(input, output)
    }

    fn num_samples(&self) -> usize {
        self.data.shape()[0]
    }

    fn num_training_samples(&self) -> usize {
        self.training_sample_size
    }
}

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    extern crate ndarray;

    use crate::DataSet;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_data_set_1() {
        let shape = [997, 10];
        let input_data = Array::random(shape, Uniform::new(-1., 1.)).into_dyn();
        let dataset = DataSetInMemory::new(input_data, 2, 0.4, true);
        assert_eq!(dataset.num_samples(), shape[0]);
        assert_eq!(dataset.num_training_samples(), 598);
        assert_eq!(dataset.num_test_samples(), 399);
        let test_data = dataset.test_data();
        assert_eq!(test_data.input.shape()[0], 399);
        assert_eq!(test_data.output.shape()[0], test_data.input.shape()[0]);

        let batch_size = 100usize;
        let mut total_sample = 0usize;
        let mut counter = 0usize;
        for batch in dataset.next_training_batch(batch_size) {
            if counter > 10 {
                break;
            }
            assert_eq!(batch.input.shape()[0], batch.output.shape()[0]);
            assert!(batch.input.shape()[0] <= batch_size);
            total_sample += batch.input.shape()[0];
            counter += 1;
        }
        assert_eq!(total_sample, shape[0]);
    }
}

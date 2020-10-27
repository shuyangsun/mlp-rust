extern crate ndarray;
use super::super::traits::numerical_traits::MLPFloat;
use super::super::traits::tensor_traits::{Tensor, TensorSampleIndependent};
use ndarray::{prelude::*, Axis};
use ndarray_stats;
use ndarray_stats::QuantileExt;

pub enum Loss {
    MSE,
    SoftmaxCrossEntropy,
}

impl<T> Tensor<T> for Loss
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        assert_eq!(input.ndim(), 2);
        let res: ArrayD<T> = match self {
            Self::MSE => input.clone().into_owned(),
            Self::SoftmaxCrossEntropy => {
                // Subtract max value in the row from the original values, to make it numerically more stable.
                let row_max = input
                    .map_axis(Axis(1), |row: ArrayView1<T>| row.max().unwrap().clone())
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let num_samples = row_max.len();
                let row_max = row_max.into_shape((num_samples, 1)).unwrap();
                let shifted = &input - &row_max;
                let exp = shifted.mapv(|ele| T::one().exp().powf(ele));
                let axis_sum = exp.sum_axis(Axis(1));
                exp / axis_sum
            }
        };
        assert_eq!(res.shape(), input.shape());
        res
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        let output_shape = output.shape();
        let shape_after_diff_mean = &output_shape[1..];
        Array::ones(shape_after_diff_mean)
    }
}

impl<T> TensorSampleIndependent<T> for Loss where T: MLPFloat {}

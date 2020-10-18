extern crate ndarray;
use super::super::custom_types::custom_traits::MLPFloat;
use ndarray::{prelude::*, Axis};
use ndarray_stats;
use ndarray_stats::QuantileExt;

pub enum LossLayer {
    MSE,
    SoftmaxCrossEntropy,
}

impl LossLayer {
    fn forward<T>(&self, input: &Array2<T>) -> Array2<T>
    where
        T: MLPFloat,
    {
        let res: Array2<T> = match self {
            Self::MSE => input.clone(),
            Self::SoftmaxCrossEntropy => {
                // Subtract max value in the row from the original values, to make it numerically more stable.
                let row_max: Array1<T> = input.map_axis(Axis(1), |row| row.max().unwrap().clone());
                let num_samples = row_max.len();
                let row_max = row_max.into_shape((num_samples, 1)).unwrap();
                let unboxed_shifted = input - &row_max;
                let exp = unboxed_shifted.mapv(|ele| T::one().exp().powf(ele));
                let axis_sum = exp.sum_axis(Axis(1));
                exp / axis_sum
            }
        };
        debug_assert_eq!(res.shape(), input.shape());
        res
    }

    fn backward<T>(&self, output: &Array2<T>, actual: &Array2<T>) -> Array1<T>
    where
        T: MLPFloat,
    {
        let res: Array2<T> = output - actual;
        debug_assert_eq!(res.shape(), output.shape());
        res.mean_axis(Axis(0)).unwrap()
    }
}

extern crate ndarray;

use super::type_def::MLPFloat;
use ndarray::{prelude::*, Axis};
use ndarray_stats;
use ndarray_stats::QuantileExt;
use std::ops::Mul;

pub enum ActivationFunction {
    Sigmoid,
    Softmax,
    TanH,
    ReLu,
    LeakyReLu,
}

impl ActivationFunction {
    /// Forward propagation of activation functions. Takes 2-D array as argument and returns 2-D array as result.
    /// Data samples are row-based.
    /// ```rust
    /// extern crate ndarray;
    /// use ndarray::prelude::*;
    /// use mlp_rust::ActivationFunction;
    ///
    /// let rand_arr = arr2(&[[1., 2.]]);
    /// let forward_res = ActivationFunction::Softmax.forward(&rand_arr);
    ///
    /// assert_eq!(forward_res, arr2(&[[0.2689414213699951, 0.7310585786300049]]));
    /// ```
    pub fn forward<T>(&self, input: &Array2<T>) -> Array2<T>
    where
        T: MLPFloat,
    {
        let res: Array2<T> = match self {
            Self::Sigmoid => {
                let exp_neg = input.mapv(|ele| T::one().exp().powf(ele.neg()));
                exp_neg.mapv(|ele| T::one().div(T::one().add(ele)))
            }
            Self::Softmax => {
                // Subtract max value in the row from the original values, to make it numerically more stable.
                let row_max: Array1<T> = input.map_axis(Axis(1), |row| row.max().unwrap().clone());
                let num_samples = row_max.len();
                let row_max = row_max.into_shape((num_samples, 1)).unwrap();
                let input_shifted = input - &row_max;
                let exp = input_shifted.mapv(|ele| T::one().exp().powf(ele));
                let axis_sum = exp.sum_axis(Axis(1));
                exp / axis_sum
            }
            Self::TanH => input.mapv(|ele| ele.sinh().div(ele.cosh())),
            Self::ReLu => input.mapv(|ele| ele.max(T::zero())),
            Self::LeakyReLu => input.mapv(|ele| {
                if ele > T::zero() {
                    ele
                } else {
                    ele.div(T::from_f32(10f32).unwrap())
                }
            }),
        };
        debug_assert_eq!(res.shape(), input.shape());
        res
    }

    pub fn derivative<T>(&self, input: &Array2<T>) -> Array2<T>
    where
        T: MLPFloat,
    {
        match self {
            Self::Sigmoid => input.mul(&self.forward(&input).mapv(|ele| T::one().sub(ele))),
            Self::Softmax => {
                // TODO
                let row_max: Array1<T> = input.map_axis(Axis(1), |row| row.max().unwrap().clone());
                let num_samples = row_max.len();
                let row_max = row_max.into_shape((num_samples, 1)).unwrap();
                let input_shifted = input - &row_max;
                let exp = input_shifted.mapv(|ele| T::one().exp().powf(ele));
                let axis_sum = exp.sum_axis(Axis(1));
                exp / axis_sum
            }
            Self::TanH => input.mapv(|ele| ele.sinh().powi(2)),
            Self::ReLu => input.mapv(|ele| if ele > T::zero() { T::one() } else { T::zero() }),
            Self::LeakyReLu => input.mapv(|ele| {
                if ele > T::zero() {
                    T::one()
                } else {
                    T::one().div(T::from_f32(10f32).unwrap())
                }
            }),
        }
    }
}

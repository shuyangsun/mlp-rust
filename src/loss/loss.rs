extern crate ndarray;
use crate::traits::numerical_traits::MLPFloat;
use crate::utility::math::to_2d_view;
use ndarray::prelude::*;
use ndarray_stats;
use ndarray_stats::QuantileExt;

pub enum Loss {
    MSE,
    SoftmaxCrossEntropy,
}

impl Loss {
    pub fn calculate_loss<T>(
        &self,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
        should_be_parallel: bool,
    ) -> T
    where
        T: MLPFloat,
    {
        assert_eq!(input.ndim(), 2);
        assert_eq!(expected_output.ndim(), 2);
        let input = to_2d_view(input);
        let expected_output = to_2d_view(expected_output);
        match self {
            Self::MSE => {
                let mut diff = &expected_output - &input;
                let two = T::one() + T::one();
                if should_be_parallel {
                    diff.par_mapv_inplace(|ele| ele.powi(2).div(two));
                } else {
                    diff.mapv_inplace(|ele| ele.powi(2).div(two));
                }
                diff.mean().unwrap()
            }
            Self::SoftmaxCrossEntropy => {
                let mut softmax_res = self.predict(input.view().into_dyn(), should_be_parallel);
                if should_be_parallel {
                    softmax_res.par_mapv_inplace(|ele| ele.ln());
                } else {
                    softmax_res.mapv_inplace(|ele| ele.ln());
                }
                let loss = (&expected_output * &softmax_res).sum_axis(Axis(1));
                -loss.mean().unwrap()
            }
        }
    }

    pub fn predict<T>(&self, input: ArrayViewD<T>, should_be_parallel: bool) -> ArrayD<T>
    where
        T: MLPFloat,
    {
        assert_eq!(input.ndim(), 2);
        let input = to_2d_view(input);
        match self {
            Self::MSE => input.into_owned(),
            Self::SoftmaxCrossEntropy => {
                // Subtract max value in the row from the original values, to make it numerically more stable.
                let row_max =
                    input.map_axis(Axis(1), |row: ArrayView1<T>| row.max().unwrap().clone());
                let num_samples = row_max.len();
                let row_max = row_max.into_shape((num_samples, 1)).unwrap();
                let mut shifted = &input - &row_max;
                if should_be_parallel {
                    shifted.par_mapv_inplace(|ele| T::one().exp().powf(ele));
                } else {
                    shifted.mapv_inplace(|ele| T::one().exp().powf(ele));
                }
                let axis_sum = shifted
                    .sum_axis(Axis(1))
                    .into_shape([shifted.shape()[0], 1])
                    .unwrap();
                shifted / axis_sum
            }
        }
        .into_dyn()
    }

    pub fn backward_with_respect_to_input<T>(
        &self,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
        should_be_parallel: bool,
    ) -> ArrayD<T>
    where
        T: MLPFloat,
    {
        assert_eq!(input.ndim(), 2);
        assert_eq!(expected_output.ndim(), 2);
        let input = to_2d_view(input);
        let expected_output = to_2d_view(expected_output);
        match self {
            Self::MSE => (&input - &expected_output).into_dyn(),
            Self::SoftmaxCrossEntropy => {
                let softmax_res = self.predict(input.view().into_dyn(), should_be_parallel);
                (&softmax_res - &expected_output).into_dyn()
            }
        }
    }
}

#[macro_export]
macro_rules! mse {
    () => {{
        crate::loss::loss::Loss::MSE
    }};
}

#[macro_export]
macro_rules! softmax_cross_entropy {
    () => {{
        crate::loss::loss::Loss::SoftmaxCrossEntropy
    }};
}

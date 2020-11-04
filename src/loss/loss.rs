extern crate ndarray;
use crate::regularization::weight_regularization::WeightRegularization;
use crate::traits::numerical_traits::MLPFloat;
use crate::utility::math::to_2d_view;
use ndarray::prelude::*;
use ndarray_stats;
use ndarray_stats::QuantileExt;

pub enum LossKind {
    MSE,
    SoftmaxCrossEntropy,
}

pub struct Loss<'a, T> {
    kind: LossKind,
    regularization: Option<WeightRegularization<'a, T>>,
}

impl<'a, T> Loss<'a, T> {
    pub fn new(kind: LossKind) -> Self {
        Self {
            kind,
            regularization: None,
        }
    }

    pub fn new_with_regularization(
        kind: LossKind,
        regularization: WeightRegularization<'a, T>,
    ) -> Self {
        Self {
            kind,
            regularization: Some(regularization),
        }
    }
}

impl<'a, T> Loss<'a, T>
where
    T: MLPFloat,
{
    pub fn calculate_loss(
        &self,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
        should_be_parallel: bool,
    ) -> T {
        assert_eq!(input.ndim(), 2);
        assert_eq!(expected_output.ndim(), 2);
        let input = to_2d_view(input);
        let expected_output = to_2d_view(expected_output);
        let mut res = match self.kind {
            LossKind::MSE => {
                let mut diff = &expected_output - &input;
                let two = T::one() + T::one();
                if should_be_parallel {
                    diff.par_mapv_inplace(|ele| ele * ele.div(two));
                } else {
                    diff.mapv_inplace(|ele| ele * ele.div(two));
                }
                diff.mean().unwrap()
            }
            LossKind::SoftmaxCrossEntropy => {
                let mut softmax_res = self.predict(input.view().into_dyn(), should_be_parallel);
                if should_be_parallel {
                    softmax_res.par_mapv_inplace(|ele| ele.ln());
                } else {
                    softmax_res.mapv_inplace(|ele| ele.ln());
                }
                let loss = (&expected_output * &softmax_res).sum_axis(Axis(1));
                -loss.mean().unwrap()
            }
        };
        if self.regularization.is_some() {
            res = res
                + self
                    .regularization
                    .as_ref()
                    .unwrap()
                    .calculate_loss(should_be_parallel);
        }
        res
    }

    pub fn predict(&self, input: ArrayViewD<T>, should_be_parallel: bool) -> ArrayD<T> {
        assert_eq!(input.ndim(), 2);
        let input = to_2d_view(input);
        match self.kind {
            LossKind::MSE => input.into_owned(),
            LossKind::SoftmaxCrossEntropy => {
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

    pub fn backward_with_respect_to_input(
        &self,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
        should_be_parallel: bool,
    ) -> ArrayD<T> {
        assert_eq!(input.ndim(), 2);
        assert_eq!(expected_output.ndim(), 2);
        let input = to_2d_view(input);
        let expected_output = to_2d_view(expected_output);
        match self.kind {
            LossKind::MSE => (&input - &expected_output).into_dyn(),
            LossKind::SoftmaxCrossEntropy => {
                let softmax_res = self.predict(input.view().into_dyn(), should_be_parallel);
                (&softmax_res - &expected_output).into_dyn()
            }
        }
    }
}

#[macro_export]
macro_rules! mse {
    () => {{
        crate::loss::loss::LossKind::MSE
    }};
}

#[macro_export]
macro_rules! softmax_cross_entropy {
    () => {{
        crate::loss::loss::LossKind::SoftmaxCrossEntropy
    }};
}

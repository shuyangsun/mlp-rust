extern crate ndarray;
use super::super::traits::numerical_traits::MLPFloat;
use super::super::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::prelude::*;
use ndarray_stats;
use ndarray_stats::QuantileExt;

pub enum Loss {
    Softmax,
}

impl<T> Tensor<T> for Loss
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        assert_eq!(input.ndim(), 2);
        let input = input.into_dimensionality::<Ix2>().unwrap();
        let res: ArrayD<T> = match self {
            Self::Softmax => {
                // Subtract max value in the row from the original values, to make it numerically more stable.
                let row_max =
                    input.map_axis(Axis(1), |row: ArrayView1<T>| row.max().unwrap().clone());
                let num_samples = row_max.len();
                let row_max = row_max.into_shape((num_samples, 1)).unwrap();
                let shifted = &input - &row_max;
                let exp = shifted.mapv(|ele| T::one().exp().powf(ele));
                let axis_sum = exp
                    .sum_axis(Axis(1))
                    .into_shape([exp.shape()[0], 1])
                    .unwrap();
                (exp / axis_sum).into_dyn()
            }
        };
        assert_eq!(res.shape(), input.shape());
        res
    }

    fn backward_respect_to_input(
        &self,
        _: ArrayViewD<T>,
        layer_output: ArrayViewD<T>,
    ) -> ArrayD<T> {
        let mut res = layer_output.into_owned();
        res
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(0)
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        match self {
            Self::Softmax => CounterEst::None, // Cannot determine because it depends on the size of last layers
        }
    }
}

#[macro_export]
macro_rules! softmax {
    () => {{
        Box::new(Loss::Softmax)
    }};
}

extern crate ndarray;
use super::{tensor::Tensor, type_def::MLPFloat};
use ndarray::prelude::*;

pub enum ActivationFunction {
    TanH,
    ReLu,
    LeakyReLu,
}

impl<T> Tensor<T> for ActivationFunction
where
    T: MLPFloat,
{
    /// Forward propagation of activation functions. Takes 2-D array as argument and returns 2-D array as result.
    /// Data samples are row-based.
    /// ```rust
    /// extern crate ndarray;
    /// use ndarray::prelude::*;
    /// use mlp_rust::{ActivationFunction, Tensor};
    ///
    /// let rand_arr = arr2(&[[-1., 2.]]);
    /// let forward_res = ActivationFunction::ReLu.forward(&rand_arr);
    ///
    /// assert_eq!(forward_res, arr2(&[[0., 2.]]));
    /// ```
    fn forward(&self, input: &Array2<T>) -> Array2<T>
    where
        T: MLPFloat,
    {
        let res: Array2<T> = match self {
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

    fn backward_batch(&self, output: &Array2<T>) -> Array2<T>
    where
        T: MLPFloat,
    {
        let res = match self {
            Self::TanH => output.mapv(|ele| T::one() - ele.powi(2)),
            Self::ReLu => output.mapv(|ele| if ele > T::zero() { T::one() } else { T::zero() }),
            Self::LeakyReLu => output.mapv(|ele| {
                if ele > T::zero() {
                    T::one()
                } else {
                    T::one().div(T::from_f32(10f32).unwrap())
                }
            }),
        };
        debug_assert_eq!(res.shape(), output.shape());
        res
    }
}

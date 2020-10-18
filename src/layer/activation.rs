extern crate ndarray;
use super::super::custom_types::custom_traits::{MLPFloat, Tensor};
use ndarray::prelude::*;

pub enum ActivationLayer {
    TanH,
    ReLu,
    LeakyReLu,
}

impl<T> Tensor<T> for ActivationLayer
where
    T: MLPFloat,
{
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

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::super::super::custom_types::custom_traits::Tensor;
    use super::ActivationLayer;
    use ndarray::prelude::*;

    #[test]
    fn test_relu_forward() {
        let rand_arr = arr2(&[[-1., 2.]]);
        let forward_res = ActivationLayer::ReLu.forward(&rand_arr);
        assert_eq!(forward_res, arr2(&[[0., 2.]]));
    }

    #[test]
    fn test_relu_backward() {
        let rand_arr = arr2(&[[-1., 2.]]);
        let forward_res = ActivationLayer::ReLu.forward(&rand_arr);
        let backward_res = ActivationLayer::ReLu.backward(&forward_res);
        assert_eq!(backward_res, arr1(&[0., 1.]));
    }

    #[test]
    fn test_leaky_relu_forward() {
        let rand_arr = arr2(&[[-1., 2.]]);
        let forward_res = ActivationLayer::LeakyReLu.forward(&rand_arr);
        assert_eq!(forward_res, arr2(&[[-0.1, 2.]]));
    }

    #[test]
    fn test_leaky_relu_backward() {
        let rand_arr = arr2(&[[-1., 2.]]);
        let forward_res = ActivationLayer::LeakyReLu.forward(&rand_arr);
        let backward_res = ActivationLayer::LeakyReLu.backward(&forward_res);
        assert_eq!(backward_res, arr1(&[0.1, 1.]));
    }
}

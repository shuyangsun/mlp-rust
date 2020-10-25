extern crate ndarray;
use super::super::custom_types::numerical_traits::MLPFloat;
use super::super::custom_types::tensor_traits::TensorComputable;
use ndarray::prelude::*;

pub enum ActivationLayer {
    TanH,
    ReLu,
    LeakyReLu,
}

impl<T> TensorComputable<T> for ActivationLayer
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T>
    where
        T: MLPFloat,
    {
        let res = match self {
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
        assert_eq!(res.shape(), input.shape());
        res
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T>
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
        assert_eq!(res.shape(), output.shape());
        res
    }
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::super::super::custom_types::tensor_traits::TensorComputable;
    use super::ActivationLayer;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_relu_forward() {
        let rand_arr = arr2(&[[-1., 2.]]);
        let forward_res = ActivationLayer::ReLu.forward(rand_arr.into_dyn().view());
        assert_eq!(forward_res, arr2(&[[0., 2.]]).into_dyn());
    }

    #[test]
    fn test_relu_forward_random_arr() {
        let shape = [2, 5];
        let rand_arr = Array::random(shape, Uniform::new(0., 10.));
        let forward_res = ActivationLayer::ReLu.forward(rand_arr.into_dyn().view());
        assert_eq!(forward_res.shape(), &shape);
    }

    #[test]
    fn test_relu_backward() {
        let rand_arr = arr2(&[[-1., 2.]]);
        let forward_res = ActivationLayer::ReLu.forward(rand_arr.into_dyn().view());
        let backward_res = ActivationLayer::ReLu.backward(forward_res.view());
        assert_eq!(backward_res, arr1(&[0., 1.]).into_dyn());
    }

    #[test]
    fn test_leaky_relu_forward() {
        let rand_arr = Box::new(arr2(&[[-1., 2.]]));
        let forward_res = ActivationLayer::LeakyReLu.forward(rand_arr.into_dyn().view());
        assert_eq!(forward_res, arr2(&[[-0.1, 2.]]).into_dyn());
    }

    #[test]
    fn test_leaky_relu_backward() {
        let rand_arr = Box::new(arr2(&[[-1., 2.]]));
        let forward_res = ActivationLayer::LeakyReLu.forward(rand_arr.into_dyn().view());
        let backward_res = ActivationLayer::LeakyReLu.backward(forward_res.view());
        assert_eq!(backward_res, arr1(&[0.1, 1.]).into_dyn());
    }
}

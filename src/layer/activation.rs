extern crate ndarray;
use super::super::traits::numerical_traits::MLPFloat;
use super::super::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::prelude::*;

pub enum Activation {
    TanH,
    ReLu,
    LeakyReLu,
    // Swish, // TODO: implementation
}

impl Activation {
    fn forward_helper<T>(&self, input: ArrayViewD<T>, is_parallel: bool) -> ArrayD<T>
    where
        T: MLPFloat,
    {
        let mut res: ArrayD<T> = input.into_owned();
        let closure_fn: fn(T) -> T = match self {
            Self::TanH => |ele: T| ele.sinh().div(ele.cosh()),
            Self::ReLu => |ele: T| ele.max(T::zero()),
            Self::LeakyReLu => |ele: T| {
                if ele > T::zero() {
                    ele
                } else {
                    ele.div(T::from_f32(10f32).unwrap())
                }
            },
        };
        if is_parallel {
            res.par_mapv_inplace(closure_fn)
        } else {
            res.mapv_inplace(closure_fn)
        }
        res
    }
}

impl<T> Tensor<T> for Activation
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T>
    where
        T: MLPFloat,
    {
        self.forward_helper(input, false)
    }

    fn backward(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        let mut res: ArrayD<T> = output.into_owned();
        match self {
            Self::TanH => res.par_mapv_inplace(|ele| T::one() - ele.powi(2)),
            Self::ReLu => {
                res.par_mapv_inplace(|ele| if ele > T::zero() { T::one() } else { T::zero() })
            }
            Self::LeakyReLu => res.par_mapv_inplace(|ele| {
                if ele > T::zero() {
                    T::one()
                } else {
                    T::one().div(T::from_f32(10f32).unwrap())
                }
            }),
        };
        res
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T>
    where
        T: MLPFloat,
    {
        self.forward_helper(input, true)
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(0)
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(match self {
            Self::TanH => 2,
            Self::ReLu => 1,
            Self::LeakyReLu => 1,
        })
    }
}

#[macro_export]
macro_rules! tanh {
    () => {{
        crate::traits::tensor_traits::TensorTraitObjWrapper::ForwardParallel(Box::new(
            Activation::TanH,
        ))
    }};
}

#[macro_export]
macro_rules! relu {
    () => {{
        crate::traits::tensor_traits::TensorTraitObjWrapper::ForwardParallel(Box::new(
            Activation::ReLu,
        ))
    }};
}

#[macro_export]
macro_rules! leaky_relu {
    () => {{
        crate::traits::tensor_traits::TensorTraitObjWrapper::ForwardParallel(Box::new(
            Activation::LeakyReLu,
        ))
    }};
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::super::super::traits::tensor_traits::Tensor;
    use super::Activation;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_relu_forward() {
        let rand_arr = arr2(&[[-1., 2.]]).into_dyn();
        let forward_res = Activation::ReLu.forward(rand_arr.view());
        let par_forward_res = Activation::ReLu.forward(rand_arr.view());
        assert_eq!(forward_res, arr2(&[[0., 2.]]).into_dyn());
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_relu_forward_random_arr() {
        let shape = [2, 5];
        let rand_arr = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let forward_res = Activation::ReLu.forward(rand_arr.view());
        let par_forward_res = Activation::ReLu.forward(rand_arr.view());
        assert_eq!(forward_res.shape(), &shape);
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_relu_backward() {
        let rand_arr = arr2(&[[-1., 2.]]).into_dyn();
        let forward_res = Activation::ReLu.forward(rand_arr.view());
        let par_forward_res = Activation::ReLu.forward(rand_arr.view());
        let backward_res = Activation::ReLu.backward(forward_res.view());
        assert_eq!(backward_res, arr2(&[[0., 1.]]).into_dyn());
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let rand_arr = Box::new(arr2(&[[-1., 2.]])).into_dyn();
        let forward_res = Activation::LeakyReLu.forward(rand_arr.view());
        let par_forward_res = Activation::LeakyReLu.forward(rand_arr.view());
        assert_eq!(forward_res, arr2(&[[-0.1, 2.]]).into_dyn());
        assert_eq!(forward_res, par_forward_res);
    }

    #[test]
    fn test_leaky_relu_backward() {
        let rand_arr = Box::new(arr2(&[[-1., 2.]])).into_dyn();
        let forward_res = Activation::LeakyReLu.forward(rand_arr.view());
        let par_forward_res = Activation::LeakyReLu.forward(rand_arr.view());
        let backward_res = Activation::LeakyReLu.backward(forward_res.view());
        assert_eq!(backward_res, arr2(&[[0.1, 1.]]).into_dyn());
        assert_eq!(forward_res, par_forward_res);
    }
}

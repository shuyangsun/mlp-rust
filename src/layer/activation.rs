use crate::utility::{counter::CounterEst, math::tanh_safe};
use crate::{MLPFloat, Tensor};
use ndarray::{ArrayD, ArrayViewD};

#[derive(Clone)]
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
            Self::TanH => |ele: T| tanh_safe(&ele),
            Self::ReLu => |ele: T| ele.max(T::zero()),
            Self::LeakyReLu => |ele: T| {
                if ele > T::zero() {
                    ele
                } else {
                    ele.div(T::from_u32(10).unwrap())
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

    fn backward_respect_to_input(
        &self,
        _: ArrayViewD<T>,
        layer_output: ArrayViewD<T>,
    ) -> ArrayD<T> {
        let mut res: ArrayD<T> = layer_output.into_owned();
        match self {
            Self::TanH => {
                res.par_mapv_inplace(|ele| T::one() - ele.powi(2));
            }
            Self::ReLu => {
                res.par_mapv_inplace(|ele| if ele > T::zero() { ele } else { T::zero() });
            }
            Self::LeakyReLu => {
                res.par_mapv_inplace(|ele| {
                    if ele > T::zero() {
                        ele
                    } else {
                        T::one().div(T::from_u32(10).unwrap())
                    }
                });
            }
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
        Box::new(Activation::TanH)
    }};
}

#[macro_export]
macro_rules! relu {
    () => {{
        Box::new(Activation::ReLu)
    }};
}

#[macro_export]
macro_rules! leaky_relu {
    () => {{
        Box::new(Activation::LeakyReLu)
    }};
}

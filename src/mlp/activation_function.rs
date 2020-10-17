extern crate ndarray;

use ndarray::prelude::*;
use num_traits;

pub enum ActivationFunction {
    Sigmoid,
}

impl ActivationFunction {
    pub fn forward<T>(&self, x: &Array2<T>) -> Array2<T>
    where
        T: num_traits::Float,
    {
        match self {
            Self::Sigmoid => {
                let e_to_the_neg = x.mapv(|ele| T::one().exp().powf(ele.neg()));
                e_to_the_neg.mapv(|ele| T::one().div(T::one().add(ele)))
            }
        }
    }
}

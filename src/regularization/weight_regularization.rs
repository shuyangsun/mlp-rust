extern crate ndarray;
use crate::traits::numerical_traits::MLPFloat;
use ndarray::ArrayViewD;
use rayon::prelude::*;

pub enum WeightRegularizationKind {
    Ridge,
    Lasso,
}

pub struct WeightRegularization<'a, T> {
    lambda: T,
    kind: WeightRegularizationKind,
    weights_refs: Vec<ArrayViewD<'a, T>>,
}

impl<'a, T> WeightRegularization<'a, T> {
    pub fn new(lambda: T, kind: WeightRegularizationKind) -> Self {
        Self {
            lambda,
            kind,
            weights_refs: Vec::new(),
        }
    }
}

impl<'a, T> WeightRegularization<'a, T>
where
    T: MLPFloat,
{
    pub fn calculate_loss(&self, should_be_parallel: bool) -> T {
        let map_func = |mat: &ArrayViewD<'a, T>| {
            mat.mapv(|ele| match self.kind {
                WeightRegularizationKind::Lasso => ele.abs(),
                WeightRegularizationKind::Ridge => ele * ele,
            })
            .sum()
        };
        let all_sums: Vec<T> = if should_be_parallel {
            self.weights_refs.par_iter().map(map_func).collect()
        } else {
            self.weights_refs.iter().map(map_func).collect()
        };
        let mut res = T::zero();
        for val in all_sums {
            res = res + val;
        }
        res
    }
}

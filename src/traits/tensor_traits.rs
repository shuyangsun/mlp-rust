extern crate ndarray;
use super::numerical_traits::MLPFloat;
use crate::traits::optimizer_traits::Optimizer;
use crate::utility::counter::CounterEst;
use crate::utility::linalg::{split_arr_view_into_chunks_by_axis0, stack_arr_views};
use ndarray::prelude::*;
use rayon::prelude::*;

pub trait Tensor<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T>;
    fn backward_respect_to_input(
        &self,
        layer_input: ArrayViewD<T>,
        layer_output: ArrayViewD<T>,
    ) -> ArrayD<T>;

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward(input)
    }

    fn is_frozen(&self) -> bool {
        true
    }

    fn backward_update_check_frozen(
        &mut self,
        input: ArrayViewD<T>,
        output_gradient: ArrayViewD<T>,
        optimizer: &Box<dyn Optimizer<T>>,
    ) {
        if self.is_frozen() {
            return;
        }
        self.backward_update(input, output_gradient, optimizer);
    }

    fn backward_update(
        &mut self,
        input: ArrayViewD<T>,
        output_gradient: ArrayViewD<T>,
        optimizer: &Box<dyn Optimizer<T>>,
    ) {
        unimplemented!()
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::None
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::None
    }
}

pub trait TensorForwardParallelable<T>: Tensor<T>
where
    T: MLPFloat,
{
    fn par_batch_forward(&self, inputs: &Vec<ArrayViewD<T>>) -> Vec<ArrayD<T>> {
        let stacked = stack_arr_views(inputs);
        let stacked_view = stacked.view();
        let res_stacked = self.par_forward(stacked_view);
        let res_stacked_view = res_stacked.view();
        let res_views = split_arr_view_into_chunks_by_axis0(&res_stacked_view, inputs.len());
        res_views
            .iter()
            .map(|arr_view| arr_view.clone().into_owned())
            .collect()
    }
}

impl<T, U> TensorForwardParallelable<T> for U
where
    T: MLPFloat,
    U: Tensor<T> + Sync,
{
    fn par_batch_forward(&self, inputs: &Vec<ArrayViewD<T>>) -> Vec<ArrayD<T>> {
        inputs
            .par_iter()
            .map(|input: &ArrayViewD<T>| self.forward(input.clone()))
            .collect()
    }
}

pub enum TensorTraitObjWrapper<T>
where
    T: MLPFloat,
{
    Basic(Box<dyn Tensor<T>>),
    ForwardParallel(Box<dyn TensorForwardParallelable<T>>),
}

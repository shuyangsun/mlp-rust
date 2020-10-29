extern crate ndarray;
use super::numerical_traits::MLPFloat;
use crate::utility::counter::CounterEst;
use crate::utility::linalg::{split_arr_view_into_chunks_by_axis0, stack_arr_views};
use ndarray::prelude::*;
use rayon::prelude::*;

pub trait Tensor<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T>;
    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T>;

    fn updatable_mat(&mut self) -> ArrayViewMutD<T> {
        unimplemented!()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward(input)
    }

    fn backward(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        self.backward_batch(output).mean_axis(Axis(0)).unwrap()
    }

    fn is_frozen(&self) -> bool {
        true
    }

    fn update(&mut self, gradient_last_layer: ArrayViewD<T>) {
        if self.is_frozen() {
            return;
        }
        let mut original_mat = self.updatable_mat();
        let update_res: ArrayD<T> = &original_mat - &gradient_last_layer;
        assert_eq!(update_res.shape(), original_mat.shape());
        original_mat.assign(&update_res);
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::None
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::None
    }
}

pub trait TensorSampleIndependent<T>: Tensor<T>
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

impl<T, U> TensorSampleIndependent<T> for U
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
    SampleIndependent(Box<dyn TensorSampleIndependent<T>>),
}

#[macro_export]
macro_rules! tensor {
    ($x:expr) => {{
        TensorTraitObjWrapper::Basic(Box::new($x))
    }};
}

#[macro_export]
macro_rules! par_tensor {
    ($x:expr) => {{
        TensorTraitObjWrapper::SampleIndependent(Box::new($x))
    }};
}

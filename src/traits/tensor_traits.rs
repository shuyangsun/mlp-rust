extern crate ndarray;
use crate::traits::optimizer_traits::Optimizer;
use crate::utility::counter::CounterEst;
use downcast_rs::{impl_downcast, Downcast};
use ndarray::prelude::*;

pub trait Tensor<T>: Downcast {
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
        _input: ArrayViewD<T>,
        _output_gradient: ArrayViewD<T>,
        _optimizer: &Box<dyn Optimizer<T>>,
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

impl_downcast!(Tensor<T>);

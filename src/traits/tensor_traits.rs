use crate::traits::optimizer_traits::Optimizer;
use crate::utility::counter::CounterEst;
use downcast_rs::{impl_downcast, Downcast};
use ndarray::{Array, ArrayView};

pub trait Tensor<T, InputD, OutputD>: Downcast {
    fn is_frozen(&self) -> bool {
        true
    }

    fn forward(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD>;

    fn par_forward(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD> {
        self.forward(input)
    }

    fn backward_respect_to_input(
        &self,
        layer_input: ArrayView<T, InputD>,
        layer_output: ArrayView<T, OutputD>,
    ) -> Array<T, InputD>;

    fn backward_update(
        &mut self,
        _input: ArrayView<T, InputD>,
        _output_gradient: ArrayView<T, OutputD>,
        _optimizer: &Box<dyn Optimizer<T, InputD>>,
    ) {
        unimplemented!()
    }

    fn backward_update_check_frozen(
        &mut self,
        input: ArrayView<T, InputD>,
        output_gradient: ArrayView<T, OutputD>,
        optimizer: &Box<dyn Optimizer<T, InputD>>,
    ) {
        if self.is_frozen() {
            return;
        }
        self.backward_update(input, output_gradient, optimizer);
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::None
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::None
    }
}

impl_downcast!(Tensor<T, InputD, OutputD>);

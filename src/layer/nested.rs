use crate::utility::counter::CounterEst;
use crate::{Optimizer, Tensor};
use ndarray::{Array, ArrayView};
use std::cell::RefCell;

pub struct NestedTensor<'input, T, InputD, IntermediateD, OutputD> {
    first: Box<dyn Tensor<T, InputD, IntermediateD>>,
    second: Option<NestedTensor<'input, T, IntermediateD, IntermediateD, OutputD>>,
    input: RefCell<Option<ArrayView<'input, T, InputD>>>,
    intermediate_output: RefCell<Option<Array<T, IntermediateD>>>,
    output: RefCell<Option<Array<T, OutputD>>>,
}

impl<'input, T, InputD, IntermediateD, OutputD> Tensor<T, InputD, OutputD>
    for NestedTensor<'input, T, InputD, IntermediateD, OutputD>
{
    fn is_frozen(&self) -> bool {
        self.first.is_frozen() && self.second.is_frozen()
    }

    fn forward(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD> {
        let output_1 = self.first.forward(input.clone());
        let output_2 = self.second.forward(output_1.view());
        *self.input.borrow_mut() = Some(input); // TODO: who owns input who owns output?
    }

    fn par_forward(&self, input: ArrayView<'_, T, InputD>) -> Array<T, OutputD> {
        unimplemented!()
    }

    fn backward_respect_to_input(
        &self,
        layer_input: ArrayView<'_, T, InputD>,
        layer_output: ArrayView<'_, T, OutputD>,
    ) -> Array<T, InputD> {
        unimplemented!()
    }

    fn backward_update(
        &mut self,
        _input: ArrayView<'_, T, InputD>,
        _output_gradient: ArrayView<'_, T, OutputD>,
        _optimizer: &Box<dyn Optimizer<T>>,
    ) {
        unimplemented!()
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        unimplemented!()
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        unimplemented!()
    }
}

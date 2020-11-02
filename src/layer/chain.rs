use crate::traits::numerical_traits::MLPFloat;
use crate::traits::tensor_traits::{Tensor, TensorTraitObjWrapper};
use crate::utility::counter::CounterEst;
use crate::utility::linalg::{split_arr_view_into_chunks_by_axis0, stack_arr_views};
use ndarray::{ArrayD, ArrayViewD};
use num_cpus;
use std::cell::RefCell;

#[derive(Clone)]
enum LayerOutput<T> {
    Single(ArrayD<T>),
    Multiple(Vec<ArrayD<T>>),
}

pub struct LayerChain<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    layers: Vec<TensorTraitObjWrapper<T>>,
    layer_outputs: RefCell<Vec<LayerOutput<T>>>,
}

impl<T> LayerChain<T>
where
    T: MLPFloat,
{
    pub fn new() -> Self {
        Self::new_from_sublayers(Vec::new())
    }

    pub fn new_from_sublayers<I: IntoIterator<Item = TensorTraitObjWrapper<T>>>(layers: I) -> Self {
        Self {
            is_frozen: false,
            layers: layers.into_iter().collect(),
            layer_outputs: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&mut self, layer: TensorTraitObjWrapper<T>) {
        self.layers.push(layer)
    }

    pub fn push_all<I: IntoIterator<Item = TensorTraitObjWrapper<T>>>(&mut self, layer: I) {
        self.layers.extend(layer)
    }

    pub fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false, false)
    }

    pub fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false, true)
    }

    pub fn output_diff(&self, expected: ArrayViewD<T>, actual: ArrayViewD<T>) -> ArrayD<T> {
        &expected - &actual
    }

    fn forward_helper(
        &self,
        input: ArrayViewD<T>,
        should_cache_layer_outputs: bool,
        is_parallel: bool,
    ) -> ArrayD<T> {
        self.layer_outputs.borrow_mut().clear();
        if self.layers.is_empty() {
            panic!("Cannot calculate feed forward propagation with no layer specified.");
        }
        let first_res = match self.layers.first().unwrap() {
            TensorTraitObjWrapper::Basic(layer) => {
                if is_parallel {
                    LayerOutput::Single(layer.par_forward(input))
                } else {
                    LayerOutput::Single(layer.forward(input))
                }
            }
            TensorTraitObjWrapper::ForwardParallel(layer) => {
                if is_parallel {
                    let thread_count = num_cpus::get();
                    let view_sliced = split_arr_view_into_chunks_by_axis0(&input, thread_count);
                    LayerOutput::Multiple(layer.par_batch_forward(&view_sliced))
                } else {
                    LayerOutput::Single(layer.forward(input))
                }
            }
        };
        let mut outputs = self.layer_outputs.borrow_mut();
        outputs.push(first_res);
        for layer in &self.layers[1..] {
            let cur_input = outputs.last().unwrap();
            let next = layer_forward_helper(layer, cur_input, is_parallel);
            if !should_cache_layer_outputs {
                outputs.clear();
            }
            outputs.push(next);
        }
        let last_output: LayerOutput<T> = if should_cache_layer_outputs {
            outputs.last().unwrap().clone()
        } else {
            outputs.pop().unwrap()
        };
        match last_output {
            LayerOutput::Single(arr) => arr,
            LayerOutput::Multiple(arr_vec) => {
                let arr_view: Vec<ArrayViewD<T>> = arr_vec.iter().map(|ele| ele.view()).collect();
                stack_arr_views(&arr_view)
            }
        }
    }
}

impl<T> Tensor<T> for LayerChain<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, true, false)
    }

    fn backward_batch(&self, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, true, true)
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        let mut res = CounterEst::Accurate(0);
        for layer in &self.layers {
            res += match layer {
                TensorTraitObjWrapper::Basic(tensor) => tensor.num_parameters(),
                TensorTraitObjWrapper::ForwardParallel(tensor) => tensor.num_parameters(),
            };
        }
        res
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        let mut res = CounterEst::Accurate(0);
        for layer in &self.layers {
            res += match layer {
                TensorTraitObjWrapper::Basic(tensor) => tensor.num_operations_per_forward(),
                TensorTraitObjWrapper::ForwardParallel(tensor) => {
                    tensor.num_operations_per_forward()
                }
            };
        }
        res
    }
}

fn layer_forward_helper<T>(
    layer: &TensorTraitObjWrapper<T>,
    input: &LayerOutput<T>,
    is_parallel: bool,
) -> LayerOutput<T>
where
    T: MLPFloat,
{
    if is_parallel {
        match layer {
            TensorTraitObjWrapper::Basic(layer) => match input {
                LayerOutput::Single(input_arr) => {
                    LayerOutput::Single(layer.par_forward(input_arr.view()))
                }
                LayerOutput::Multiple(input_vec) => {
                    let input_view: Vec<ArrayViewD<T>> =
                        input_vec.iter().map(|ele| ele.view()).collect();
                    let stacked = stack_arr_views(&input_view);
                    LayerOutput::Single(layer.par_forward(stacked.view()))
                }
            },
            TensorTraitObjWrapper::ForwardParallel(layer) => match input {
                LayerOutput::Single(input_arr) => {
                    let thread_count = num_cpus::get();
                    let input_view = input_arr.view();
                    let view_sliced =
                        split_arr_view_into_chunks_by_axis0(&input_view, thread_count);
                    LayerOutput::Multiple(layer.par_batch_forward(&view_sliced))
                }
                LayerOutput::Multiple(input_vec) => {
                    let input_view: Vec<ArrayViewD<T>> =
                        input_vec.iter().map(|ele| ele.view()).collect();
                    LayerOutput::Multiple(layer.par_batch_forward(&input_view))
                }
            },
        }
    } else {
        match input {
            LayerOutput::Single(input_arr) => LayerOutput::Single(match layer {
                TensorTraitObjWrapper::Basic(tensor) => tensor.forward(input_arr.view()),
                TensorTraitObjWrapper::ForwardParallel(tensor) => tensor.forward(input_arr.view()),
            }),
            LayerOutput::Multiple(input_vec) => {
                let input_view: Vec<ArrayViewD<T>> =
                    input_vec.iter().map(|ele| ele.view()).collect();
                let stacked = stack_arr_views(&input_view);
                LayerOutput::Single(match layer {
                    TensorTraitObjWrapper::Basic(tensor) => tensor.forward(stacked.view()),
                    TensorTraitObjWrapper::ForwardParallel(tensor) => {
                        tensor.forward(stacked.view())
                    }
                })
            }
        }
    }
}

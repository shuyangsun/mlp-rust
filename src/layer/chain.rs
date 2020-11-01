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

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    extern crate ndarray;
    use super::LayerChain;
    use crate::layer::{
        activation::Activation, bias::Bias, dense::Dense, normalization::BatchNormalization,
        output_and_loss::Loss,
    };
    use crate::traits::tensor_traits::{Tensor, TensorTraitObjWrapper};
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn generate_stress_dnn_classifier(input_size: usize, output_size: usize) -> LayerChain<f32> {
        LayerChain::new_from_sublayers(vec![
            dense!(input_size, 4096),
            bias!(4096),
            leaky_relu!(),
            dense!(4096, 2048),
            bias!(2048),
            relu!(),
            dense!(2048, 1024),
            bias!(1024),
            tanh!(),
            dense!(1024, 500),
            bias!(500),
            relu!(),
            dense!(500, output_size),
            bias!(output_size),
            softmax!(),
        ])
    }

    #[test]
    fn test_chained_forward() {
        let shape = [3, 10];
        let input_data = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let simple_dnn = LayerChain::new_from_sublayers(vec![
            dense!(10, 128),
            bias!(128),
            tanh!(),
            dense!(128, 64),
            tanh!(),
            batch_norm!(64),
            dense!(64, 1),
            bias!(1),
            mse!(),
        ]);
        let prediction = simple_dnn.predict(input_data.view());
        let par_prediction = simple_dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[3usize, 1usize]);
        assert_eq!(prediction, par_prediction);
    }

    #[test]
    fn test_chained_predict_stress() {
        let shape = &[100usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_dnn_classifier(shape[1], output_size);
        let prediction = dnn.predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_chained_par_predict_stress() {
        let shape = &[1000usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_dnn_classifier(shape[1], output_size);
        let prediction = dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }
}

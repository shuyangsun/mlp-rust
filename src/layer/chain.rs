use crate::traits::numerical_traits::MLPFloat;
use crate::traits::tensor_traits::{Tensor, TensorTraitObjWrapper};
use crate::utility::linalg::{split_arr_view_into_chunks_by_axis0, stack_arr_views};
use ndarray::{ArrayD, ArrayViewD};
use num_cpus;
use std::cell::RefCell;

#[derive(Clone)]
enum LayerOutput<T> {
    Basic(ArrayD<T>),
    SampleIndependent(Vec<ArrayD<T>>),
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

    pub fn new_from_sublayers(layers: Vec<TensorTraitObjWrapper<T>>) -> Self {
        Self {
            is_frozen: false,
            layers,
            layer_outputs: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&mut self, layer: TensorTraitObjWrapper<T>) {
        self.layers.push(layer)
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
                    LayerOutput::Basic(layer.par_forward(input))
                } else {
                    LayerOutput::Basic(layer.forward(input))
                }
            }
            TensorTraitObjWrapper::SampleIndependent(layer) => {
                if is_parallel {
                    let thread_count = num_cpus::get();
                    let view_sliced = split_arr_view_into_chunks_by_axis0(&input, thread_count);
                    LayerOutput::SampleIndependent(layer.par_batch_forward(&view_sliced))
                } else {
                    LayerOutput::Basic(layer.forward(input))
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
            LayerOutput::Basic(arr) => arr,
            LayerOutput::SampleIndependent(arr_vec) => {
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

    fn num_param(&self) -> Option<usize> {
        let mut res = 0usize;
        for layer in &self.layers {
            let cur_num_param = match layer {
                TensorTraitObjWrapper::Basic(tensor) => tensor.num_param(),
                TensorTraitObjWrapper::SampleIndependent(tensor) => tensor.num_param(),
            };
            if cur_num_param.is_none() {
                return None;
            }
            res += cur_num_param.unwrap();
        }
        Some(res)
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
                LayerOutput::Basic(input_arr) => {
                    LayerOutput::Basic(layer.par_forward(input_arr.view()))
                }
                LayerOutput::SampleIndependent(input_vec) => {
                    let input_view: Vec<ArrayViewD<T>> =
                        input_vec.iter().map(|ele| ele.view()).collect();
                    let stacked = stack_arr_views(&input_view);
                    LayerOutput::Basic(layer.par_forward(stacked.view()))
                }
            },
            TensorTraitObjWrapper::SampleIndependent(layer) => match input {
                LayerOutput::Basic(input_arr) => {
                    let thread_count = num_cpus::get();
                    let input_view = input_arr.view();
                    let view_sliced =
                        split_arr_view_into_chunks_by_axis0(&input_view, thread_count);
                    LayerOutput::SampleIndependent(layer.par_batch_forward(&view_sliced))
                }
                LayerOutput::SampleIndependent(input_vec) => {
                    let input_view: Vec<ArrayViewD<T>> =
                        input_vec.iter().map(|ele| ele.view()).collect();
                    LayerOutput::SampleIndependent(layer.par_batch_forward(&input_view))
                }
            },
        }
    } else {
        match input {
            LayerOutput::Basic(input_arr) => LayerOutput::Basic(match layer {
                TensorTraitObjWrapper::Basic(tensor) => tensor.forward(input_arr.view()),
                TensorTraitObjWrapper::SampleIndependent(tensor) => {
                    tensor.forward(input_arr.view())
                }
            }),
            LayerOutput::SampleIndependent(input_vec) => {
                let input_view: Vec<ArrayViewD<T>> =
                    input_vec.iter().map(|ele| ele.view()).collect();
                let stacked = stack_arr_views(&input_view);
                LayerOutput::Basic(match layer {
                    TensorTraitObjWrapper::Basic(tensor) => tensor.forward(stacked.view()),
                    TensorTraitObjWrapper::SampleIndependent(tensor) => {
                        tensor.forward(stacked.view())
                    }
                })
            }
        }
    }
}

#[cfg(test)]
mod unit_test {
    #[macro_use]
    use crate::{tensor, par_tensor};
    extern crate ndarray;
    use super::LayerChain;
    use crate::layer::{
        activation::Activation, bias::Bias, normalization::BatchNormalization,
        output_and_loss::Loss, weight::Weight,
    };
    use crate::traits::tensor_traits::{Tensor, TensorTraitObjWrapper};
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn generate_stress_dnn_classifier(input_size: usize, output_size: usize) -> LayerChain<f32> {
        LayerChain::new_from_sublayers(vec![
            par_tensor!(Weight::new_random_uniform(input_size, 4096)),
            par_tensor!(Bias::new(4096)),
            par_tensor!(Activation::LeakyReLu),
            par_tensor!(Weight::new_random_uniform(4096, 2048)),
            par_tensor!(Bias::new(2048)),
            par_tensor!(Activation::ReLu),
            par_tensor!(Weight::new_random_uniform(2048, 1024)),
            par_tensor!(Bias::new(1024)),
            par_tensor!(Activation::TanH),
            par_tensor!(Weight::new_random_uniform(1024, 500)),
            par_tensor!(Bias::new(500)),
            par_tensor!(Activation::ReLu),
            par_tensor!(Weight::new_random_uniform(500, output_size)),
            par_tensor!(Bias::new(output_size)),
            par_tensor!(Loss::SoftmaxCrossEntropy),
        ])
    }

    #[test]
    fn test_chained_forward() {
        let shape = [3, 10];
        let input_data = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let simple_dnn = LayerChain::new_from_sublayers(vec![
            par_tensor!(Weight::new_random_uniform(10, 128)),
            par_tensor!(Bias::new(128)),
            par_tensor!(Activation::TanH),
            par_tensor!(Weight::new_random_uniform(128, 64)),
            par_tensor!(Activation::TanH),
            tensor!(BatchNormalization::new(64)),
            par_tensor!(Weight::new_random_uniform(64, 1)),
            par_tensor!(Bias::new(1)),
            par_tensor!(Loss::MSE),
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
        let shape = &[100usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_dnn_classifier(shape[1], output_size);
        println!("Num param: {:#?}", dnn.num_param());
        let prediction = dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }
}

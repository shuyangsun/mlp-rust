use crate::traits::numerical_traits::MLPFloat;
use crate::traits::tensor_traits::Tensor;
use ndarray::{ArrayD, ArrayViewD};
use std::cell::RefCell;

pub struct LayerChain<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    layers: Vec<Box<dyn Tensor<T>>>,
    layer_outputs: RefCell<Vec<ArrayD<T>>>,
}

impl<T> LayerChain<T>
where
    T: MLPFloat,
{
    pub fn new() -> Self {
        Self::new_from_sublayers(Vec::new())
    }

    pub fn new_from_sublayers(layers: Vec<Box<dyn Tensor<T>>>) -> Self {
        Self {
            is_frozen: false,
            layers,
            layer_outputs: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&mut self, layer: Box<dyn Tensor<T>>) {
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
        let first_res = if is_parallel {
            self.layers.first().unwrap().par_forward(input)
        } else {
            self.layers.first().unwrap().forward(input)
        };
        let mut outputs = self.layer_outputs.borrow_mut();
        outputs.push(first_res);
        for layer in &self.layers[1..] {
            let cur_input = outputs.last().unwrap().view();
            let next = if is_parallel {
                layer.par_forward(cur_input)
            } else {
                layer.forward(cur_input)
            };
            if !should_cache_layer_outputs {
                outputs.clear();
            }
            outputs.push(next);
        }
        if should_cache_layer_outputs {
            outputs.last().unwrap().clone()
        } else {
            outputs.pop().unwrap()
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
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::LayerChain;
    use crate::layer::{
        activation::Activation, bias::Bias, normalization::BatchNormalization,
        output_and_loss::Loss, weight::Weight,
    };
    use crate::traits::tensor_traits::Tensor;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_propagation_manager_forward() {
        let shape = [3, 10];
        let input_data = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let simple_dnn = LayerChain::new_from_sublayers(vec![
            Box::new(Weight::new_random_uniform(10, 128)),
            Box::new(Bias::new(128)),
            Box::new(Activation::TanH),
            Box::new(Weight::new_random_uniform(128, 64)),
            Box::new(Activation::TanH),
            Box::new(BatchNormalization::new(64)),
            Box::new(Weight::new_random_uniform(64, 1)),
            Box::new(Bias::new(1)),
            Box::new(Loss::MSE),
        ]);
        let prediction = simple_dnn.predict(input_data.view());
        let par_prediction = simple_dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[3usize, 1usize]);
        assert_eq!(prediction, par_prediction);
    }
}

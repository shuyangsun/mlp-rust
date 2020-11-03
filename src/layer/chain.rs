use crate::traits::numerical_traits::MLPFloat;
use crate::traits::optimizer_traits::Optimizer;
use crate::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::prelude::*;
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

    pub fn new_from_sublayers<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(layers: I) -> Self {
        Self {
            is_frozen: false,
            layers: layers.into_iter().collect(),
            layer_outputs: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&mut self, layer: Box<dyn Tensor<T>>) {
        self.layers.push(layer)
    }

    pub fn push_all<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(&mut self, layer: I) {
        self.layers.extend(layer)
    }

    pub fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false, false)
    }

    pub fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false, true)
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
            let cur_input = outputs.last().unwrap();
            let next = layer_forward_helper(layer, cur_input.view(), is_parallel);
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

    fn backward_respect_to_input(&self, _: ArrayViewD<T>, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, true, true)
    }

    fn backward_update(
        &mut self,
        input: ArrayViewD<T>,
        output_gradient: ArrayViewD<T>,
        optimizer: &Box<dyn Optimizer<T>>,
    ) {
        if self.is_frozen {
            return;
        }
        let input_owned = input.into_owned();
        let mut cur_gradient = output_gradient.into_owned();
        let num_layers = self.layers.len();
        for layer_idx in (0..num_layers).rev() {
            // Chain rule to multiply current layer output.
            let mut shape_after_mean_samples = Vec::from(cur_gradient.shape());
            shape_after_mean_samples[0] = 1;
            let cur_layer_output = self.layer_outputs.borrow_mut().pop().unwrap();
            let gradient_mul_output = (cur_layer_output * &cur_gradient)
                .mean_axis(Axis(0))
                .unwrap()
                .into_shape(shape_after_mean_samples)
                .unwrap();
            // Calculate next gradient before updating layer values.
            let layer_input = if layer_idx <= 0 {
                input_owned.clone() // TODO: bad performance
            } else {
                self.layer_outputs.borrow()[layer_idx - 1].clone()
            };
            let next_gradient = self.layers[layer_idx]
                .backward_respect_to_input(layer_input.view(), gradient_mul_output.view());
            // Update matrix with current gradient.
            self.layers[layer_idx].backward_update_check_frozen(
                layer_input.view(),
                gradient_mul_output.view(),
                optimizer,
            );
            // Update current gradient with next gradient.
            cur_gradient = next_gradient;
        }
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        let mut res = CounterEst::Accurate(0);
        for layer in &self.layers {
            res += layer.num_parameters();
        }
        res
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        let mut res = CounterEst::Accurate(0);
        for layer in &self.layers {
            res += layer.num_operations_per_forward();
        }
        res
    }
}

fn layer_forward_helper<T>(
    layer: &Box<dyn Tensor<T>>,
    input: ArrayViewD<T>,
    is_parallel: bool,
) -> ArrayD<T>
where
    T: MLPFloat,
{
    if is_parallel {
        layer.par_forward(input.view())
    } else {
        layer.forward(input.view())
    }
}

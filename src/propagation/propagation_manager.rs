use super::super::custom_types::numerical_traits::MLPFloat;
use super::super::custom_types::tensor_traits::TensorComputable;
use ndarray::{ArrayD, ArrayViewD};
use std::cell::RefCell;

pub struct PropagationManager<T>
where
    T: MLPFloat,
{
    layers: Vec<Box<dyn TensorComputable<T>>>,
    layer_outputs: RefCell<Vec<ArrayD<T>>>,
}

impl<T> PropagationManager<T>
where
    T: MLPFloat,
{
    pub fn new() -> Self {
        Self::new_from_layers(Vec::new())
    }

    pub fn new_from_layers(layers: Vec<Box<dyn TensorComputable<T>>>) -> Self {
        Self {
            layers,
            layer_outputs: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&mut self, layer: Box<dyn TensorComputable<T>>) {
        self.layers.push(layer)
    }
}

impl<T> TensorComputable<T> for PropagationManager<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        if self.layers.is_empty() {
            panic!("Cannot calculate feed forward propagation with no layer specified.");
        }
        let first_res = self.layers.first().unwrap().forward(input);
        self.layer_outputs.borrow_mut().push(first_res);
        for layer in &self.layers[1..] {
            let next = layer.forward(self.layer_outputs.borrow().last().unwrap().view());
            self.layer_outputs.borrow_mut().push(next);
        }
        self.layer_outputs.borrow().last().unwrap().clone()
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }
}

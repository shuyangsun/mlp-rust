use crate::custom_types::numerical_traits::MLPFloat;
use crate::custom_types::tensor_traits::TensorComputable;
use ndarray::{ArrayD, ArrayViewD};
use std::cell::RefCell;

pub struct LayerChain<T>
where
    T: MLPFloat,
{
    layers: Vec<Box<dyn TensorComputable<T>>>,
    layer_outputs: RefCell<Vec<ArrayD<T>>>,
}

impl<T> LayerChain<T>
where
    T: MLPFloat,
{
    pub fn new() -> Self {
        Self::new_from_sublayers(Vec::new())
    }

    pub fn new_from_sublayers(layers: Vec<Box<dyn TensorComputable<T>>>) -> Self {
        Self {
            layers,
            layer_outputs: RefCell::new(Vec::new()),
        }
    }

    pub fn push(&mut self, layer: Box<dyn TensorComputable<T>>) {
        self.layers.push(layer)
    }

    pub fn forward_no_cache(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        let res = self.forward(input);
        self.layer_outputs.borrow_mut().clear();
        res
    }
}

impl<T> TensorComputable<T> for LayerChain<T>
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

    fn backward_batch(&self, _: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::LayerChain;
    use crate::custom_types::tensor_traits::TensorComputable;
    use crate::layer::{
        activation::Activation, bias::Bias, normalization::BatchNormalization,
        output_and_loss::Loss, weight::Weight,
    };
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
        let forward_res = simple_dnn.forward(input_data.view());
        assert_eq!(forward_res.shape(), &[3usize, 1usize]);
    }
}

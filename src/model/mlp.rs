use crate::{
    batch_norm, bias, dense, input_norm, Activation, BatchNormalization, Bias, Dense,
    InputNormalization, Loss, MLPFLoatRandSampling, MLPFloat, Model, Optimizer, Serial, Tensor,
};
use ndarray::{ArrayD, ArrayViewD};

pub struct MLP<T>
where
    T: MLPFloat,
{
    serial_model: Serial<T>,
}

impl<T> MLP<T>
where
    T: MLPFLoatRandSampling,
{
    fn new<I: IntoIterator<Item = usize>>(
        input_size: usize,
        output_size: usize,
        hidden_layer_sizes: I,
        activation_function: Activation,
        should_normalize_input: bool,
        should_use_batch_norm: bool, // TODO: change to enums
        loss: Loss,
    ) -> Self {
        let mut layers: Vec<Box<dyn Tensor<T>>> = Vec::new();
        if should_normalize_input {
            layers.push(input_norm!(input_size));
        }
        let mut last_layer_size = input_size;
        for layer_size in hidden_layer_sizes {
            layers.push(dense!(last_layer_size, layer_size));
            layers.push(bias!(layer_size));
            layers.push(Box::new(activation_function.clone()));
            if should_use_batch_norm {
                layers.push(batch_norm!(layer_size));
            }
            last_layer_size = layer_size;
        }
        layers.push(dense!(last_layer_size, output_size));
        layers.push(bias!(output_size));
        let serial_model = Serial::new_from_layers(layers, loss);
        Self { serial_model }
    }

    fn new_classifier<I: IntoIterator<Item = usize>>(
        input_size: usize,
        output_size: usize,
        hidden_layer_sizes: I,
        activation_function: Activation,
        should_normalize_input: bool,
        should_use_batch_norm: bool, // TODO: change to enums
    ) -> Self {
        Self::new(
            input_size,
            output_size,
            hidden_layer_sizes,
            activation_function,
            should_normalize_input,
            should_use_batch_norm,
            Loss::SoftmaxCrossEntropy,
        )
    }

    fn new_regressor<I: IntoIterator<Item = usize>>(
        input_size: usize,
        output_size: usize,
        hidden_layer_sizes: I,
        activation_function: Activation,
        should_normalize_input: bool,
        should_use_batch_norm: bool, // TODO: change to enums
    ) -> Self {
        Self::new(
            input_size,
            output_size,
            hidden_layer_sizes,
            activation_function,
            should_normalize_input,
            should_use_batch_norm,
            Loss::MSE,
        )
    }
}

impl<T> Model<T> for MLP<T>
where
    T: MLPFloat,
{
    fn train(
        &mut self,
        max_num_iter: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
    ) {
        self.serial_model
            .train(max_num_iter, optimizer, input, expected_output)
    }

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.serial_model.predict(input)
    }

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.serial_model.par_predict(input)
    }
}

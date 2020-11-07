use crate::utility::counter::CounterEst;
use crate::{
    batch_norm, bias, dense, input_norm, Activation, BatchNormalization, Bias, DataSet, Dense,
    InputNormalization, Loss, MLPFLoatRandSampling, MLPFloat, Model, Optimizer, Serial, Tensor,
};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

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

    pub fn new_classifier<I: IntoIterator<Item = usize>>(
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

    pub fn new_regressor<I: IntoIterator<Item = usize>>(
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

    pub fn num_param(&self) -> CounterEst<usize> {
        self.serial_model.num_param()
    }

    pub fn num_operations_per_forward(&self) -> CounterEst<usize> {
        self.serial_model.num_operations_per_forward()
    }
}

impl<T> Model<T> for MLP<T>
where
    T: MLPFloat,
{
    fn train(
        &mut self,
        data: &mut Box<dyn DataSet<T, IxDyn>>,
        batch_size: usize,
        max_num_epoch: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        should_print: bool,
    ) {
        self.serial_model
            .train(data, batch_size, max_num_epoch, optimizer, should_print)
    }

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.serial_model.predict(input)
    }

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.serial_model.par_predict(input)
    }
}

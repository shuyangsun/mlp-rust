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

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    extern crate ndarray;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn generate_stress_mlp_classifier(input_size: usize, output_size: usize) -> MLP<f32> {
        MLP::new_classifier(
            input_size,
            output_size,
            vec![4096, 2048, 1024, 512, 256, 128, 10],
            Activation::TanH,
            true,
            false,
        )
    }

    fn generate_simple_mlp_regressor(input_size: usize) -> MLP<f32> {
        // let layers: Vec<Box<dyn Tensor<f32>>> =
        //     vec![dense!(input_size, output_size), bias!(output_size), tanh!()];
        MLP::new_regressor(
            input_size,
            1,
            vec![64, 32, 8],
            Activation::ReLu,
            true,
            false,
        )
    }

    #[test]
    fn test_mlp_forward() {
        let shape = [5, 10];
        let input_data = Array::random(shape, Uniform::new(-1., 1.)).into_dyn();
        let simple_dnn = generate_simple_mlp_regressor(shape[1]);
        let prediction = simple_dnn.predict(input_data.view());
        let par_prediction = simple_dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], 1usize]);
        assert_eq!(prediction, par_prediction);
    }

    #[test]
    fn test_mlp_forward_no_hidden_layer() {
        let shape = [3, 10];
        let input_data = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let simple_dnn = MLP::new_regressor(shape[1], 1, vec![], Activation::ReLu, false, false);
        let prediction = simple_dnn.predict(input_data.view());
        let par_prediction = simple_dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], 1usize]);
        assert_eq!(prediction, par_prediction);
    }

    #[test]
    fn test_model_predict_stress() {
        let shape = &[100usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_mlp_classifier(shape[1], output_size);
        let prediction = dnn.predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_model_par_predict_stress() {
        let shape = &[1000usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_mlp_classifier(shape[1], output_size);
        let prediction = dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_model_train() {
        let input_data = arr2(&vec![
            [0.5f32, 0.05f32],
            [0.0f32, 0.0f32],
            [-0.5f32, -0.5f32],
        ])
        .into_dyn();
        let output_data = arr2(&vec![[1.0f32], [0.0f32], [-1.0f32]]).into_dyn();
        let mut simple_dnn = MLP::new_regressor(2, 1, vec![], Activation::ReLu, false, false);
        // let mut simple_dnn = generate_simple_dnn(2, 2);
        println!(
            "Before train prediction: {:#?}",
            simple_dnn.predict(input_data.view())
        );
        simple_dnn.train(
            100,
            &gradient_descent!(0.01f32),
            input_data.view(),
            output_data.view(),
        );
        println!(
            "After train prediction: {:#?}",
            simple_dnn.predict(input_data.view())
        );
    }
}

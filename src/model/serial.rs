use crate::layer::chain::LayerChain;
use crate::loss::loss::Loss;
use crate::traits::model_traits::Model;
use crate::traits::numerical_traits::{MLPFLoatRandSampling, MLPFloat};
use crate::traits::optimizer_traits::Optimizer;
use crate::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::{ArrayD, ArrayViewD};

pub struct Serial<T>
where
    T: MLPFloat,
{
    layer_chain: LayerChain<T>,
    loss_function: Loss,
}

impl<T> Serial<T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new(loss_function: Loss) -> Self {
        Self {
            layer_chain: LayerChain::new(),
            loss_function,
        }
    }

    pub fn new_from_layers<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(
        layers: I,
        loss_function: Loss,
    ) -> Self {
        Self {
            layer_chain: LayerChain::new_from_sublayers(layers),
            loss_function,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Tensor<T>>) {
        self.layer_chain.push(layer);
    }

    pub fn add_all<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(&mut self, layers: I) {
        self.layer_chain.push_all(layers)
    }

    pub fn num_param(&self) -> CounterEst<usize> {
        self.layer_chain.num_parameters()
    }

    pub fn num_operations_per_forward(&self) -> CounterEst<usize> {
        self.layer_chain.num_operations_per_forward()
    }
}

impl<T> Model<T> for Serial<T>
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
        // TODD: clones below are temp var for testing.
        let input_clone = input.clone();
        let expected_output_clone = expected_output.clone();
        for i in 0..max_num_iter {
            let forward_res = self.layer_chain.forward(input_clone.view());
            assert_eq!(forward_res.shape(), expected_output_clone.shape());
            let gradient = self.loss_function.backward_with_respect_to_input(
                forward_res.view(),
                expected_output_clone.view(),
                true,
            );
            println!(
                "Iter {}: loss={}",
                i,
                self.loss_function.calculate_loss(
                    forward_res.view(),
                    expected_output_clone.view(),
                    true
                )
            );
            self.layer_chain.backward_update_check_frozen(
                input_clone.view(),
                gradient.view(),
                optimizer,
            );
        }
    }

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.loss_function
            .predict(self.layer_chain.predict(input).view(), false)
    }

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.loss_function
            .predict(self.layer_chain.par_predict(input.into_dyn()).view(), true)
    }
}

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    extern crate ndarray;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn generate_stress_dnn_classifier(input_size: usize, output_size: usize) -> Serial<f32> {
        let layers: Vec<Box<dyn Tensor<f32>>> = vec![
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
        ];
        Serial::new_from_layers(layers, softmax_cross_entropy!())
    }

    fn generate_simple_dnn(input_size: usize, output_size: usize) -> Serial<f32> {
        // let layers: Vec<Box<dyn Tensor<f32>>> =
        //     vec![dense!(input_size, output_size), bias!(output_size), tanh!()];
        let layers: Vec<Box<dyn Tensor<f32>>> = vec![
            dense!(input_size, 64),
            bias!(64),
            relu!(),
            dense!(64, 32),
            bias!(32),
            relu!(),
            dense!(32, 8),
            bias!(8),
            relu!(),
            dense!(8, output_size),
            bias!(output_size),
            relu!(),
        ];
        Serial::new_from_layers(layers, softmax_cross_entropy!())
    }

    #[test]
    fn test_model_forward() {
        let shape = [3, 10];
        let input_data = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let layers: Vec<Box<dyn Tensor<f32>>> = vec![
            dense!(10, 128),
            bias!(128),
            tanh!(),
            dense!(128, 64),
            tanh!(),
            batch_norm!(64),
            dense!(64, 1),
            bias!(1),
        ];
        let simple_dnn = Serial::new_from_layers(layers, mse!());
        let prediction = simple_dnn.predict(input_data.view());
        let par_prediction = simple_dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[3usize, 1usize]);
        assert_eq!(prediction, par_prediction);
    }

    #[test]
    fn test_model_predict_stress() {
        let shape = &[100usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_dnn_classifier(shape[1], output_size);
        let prediction = dnn.predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_model_par_predict_stress() {
        let shape = &[1000usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_dnn_classifier(shape[1], output_size);
        let prediction = dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_model_train() {
        // let input_data =
        //     arr2(&vec![[0.1f32, 0.5f32], [0.7f32, 0.2f32], [5.0f32, -0.1f32]]).into_dyn();
        let input_data = arr2(&vec![
            [0.5f32, 0.05f32],
            [0.0f32, 0.0f32],
            [-0.5f32, -0.5f32],
        ])
        .into_dyn();
        // let output_data = arr2(&vec![[0f32, 1f32], [1f32, 0f32], [1f32, 0f32]]).into_dyn();
        let output_data = arr2(&vec![[1.0f32], [0.0f32], [-1.0f32]]).into_dyn();
        let layers: Vec<Box<dyn Tensor<f32>>> = vec![dense!(2, 1), bias!(1)];
        let mut simple_dnn = Serial::new_from_layers(layers, mse!());
        // let mut simple_dnn = generate_simple_dnn(2, 2);
        println!(
            "Before train prediction: {:#?}",
            simple_dnn.predict(input_data.view())
        );
        simple_dnn.train(
            1000,
            &gradient_descent!(0.0001f32),
            input_data.view(),
            output_data.view(),
        );
        println!(
            "After train prediction: {:#?}",
            simple_dnn.predict(input_data.view())
        );
    }
}

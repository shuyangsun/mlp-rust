use crate::layer::chain::LayerChain;
use crate::traits::numerical_traits::MLPFLoatRandSampling;
use crate::traits::optimizer_traits::Optimizer;
use crate::traits::tensor_traits::Tensor;
use crate::utility::counter::CounterEst;
use ndarray::{ArrayD, ArrayViewD};

pub struct Model<T>
where
    T: MLPFLoatRandSampling,
{
    layer_chain: LayerChain<T>,
}

impl<T> Model<T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new() -> Self {
        Self {
            layer_chain: LayerChain::new(),
        }
    }

    pub fn new_from_layers<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(layers: I) -> Self {
        Self {
            layer_chain: LayerChain::new_from_sublayers(layers),
        }
    }

    pub fn train(
        &mut self,
        max_num_iter: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        input: ArrayViewD<T>,
        expected_output: ArrayViewD<T>,
    ) {
        let asdf = input.clone();
        for _ in 0..max_num_iter {
            let forward_res = self.layer_chain.forward(asdf.view());
            assert_eq!(forward_res.shape(), expected_output.shape());
            let gradient = forward_res - &expected_output;
            self.layer_chain
                .backward_update_check_frozen(asdf.view(), gradient.view(), optimizer);
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

    pub fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.layer_chain.predict(input)
    }

    pub fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.layer_chain.par_predict(input.into_dyn())
    }
}

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    use crate::Model;
    extern crate ndarray;
    use crate::optimizer::gradient_descent::GradientDescent;
    use crate::traits::optimizer_traits::Optimizer;
    use crate::traits::tensor_traits::Tensor;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn generate_stress_dnn_classifier(input_size: usize, output_size: usize) -> Model<f32> {
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
            softmax!(),
        ];
        Model::new_from_layers(layers)
    }

    fn generate_simple_dnn(input_size: usize, output_size: usize) -> Model<f32> {
        let layers: Vec<Box<dyn Tensor<f32>>> = vec![
            dense!(input_size, 8),
            bias!(8),
            relu!(),
            dense!(8, output_size),
            bias!(output_size),
            relu!(),
            mse!(),
        ];
        Model::new_from_layers(layers)
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
            mse!(),
        ];
        let simple_dnn = Model::new_from_layers(layers);
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
        let shape = [3, 10];
        let input_data = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
        let output_data = Array::random((3, 1), Uniform::new(0., 10.)).into_dyn();
        let mut simple_dnn = generate_simple_dnn(shape[1], 1);
        println!(
            "Before train prediction: {:#?}",
            simple_dnn.predict(input_data.view())
        );
        let optimizer = Box::new(GradientDescent::new(0.01f32)) as Box<dyn Optimizer<f32>>;
        simple_dnn.train(2, &optimizer, input_data.view(), output_data.view());
        println!(
            "After train prediction: {:#?}",
            simple_dnn.predict(input_data.view())
        );
    }
}

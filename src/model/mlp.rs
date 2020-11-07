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
    fn train<'dset, 'dview, 'model>(
        &'model mut self,
        data: &'dset mut Box<dyn DataSet<'dset, 'dview, T, IxDyn>>,
        batch_size: usize,
        max_num_epoch: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        should_print: bool,
    ) where
        'dset: 'dview,
    {
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

#[cfg(test)]
mod unit_test {
    use crate::prelude::*;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn generate_stress_mlp_classifier(input_size: usize, output_size: usize) -> MLP<f32> {
        MLP::new_classifier(
            input_size,
            output_size,
            vec![512, 256, 128, 10],
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
    fn test_mlp_predict_stress() {
        let shape = &[100usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_mlp_classifier(shape[1], output_size);
        let prediction = dnn.predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_mlp_par_predict_stress() {
        let shape = &[1000usize, 1024];
        let output_size = 10;
        let input_data = Array::random(shape.clone(), Uniform::new(0., 10.)).into_dyn();
        let dnn = generate_stress_mlp_classifier(shape[1], output_size);
        let prediction = dnn.par_predict(input_data.view());
        assert_eq!(prediction.shape(), &[shape[0], output_size]);
    }

    #[test]
    fn test_mlp_train() {
        let data = arr2(&vec![[0.5f32, 0.05, 1.], [0.0, 0.0, 0.], [-0.5, -0.5, -1.]]).into_dyn();
        // let mut dataset =
        //     Box::new(DataSetInMemory::new(data, 1, 1., false)) as Box<dyn DataSet<f32, IxDyn>>;
        // let mut simple_dnn = MLP::new_regressor(2, 1, vec![25, 4], Activation::ReLu, false, false);
        // let train_input = dataset.train_data().input;
        // println!(
        //     "Before train prediction: {:#?}",
        //     simple_dnn.predict(train_input)
        // );
        // // let optimizer = gradient_descent!(0.1f32);
        // let optimizer = Box::new(GradientDescent::new(0.1f32)) as Box<dyn Optimizer<f32>>;
        // simple_dnn.train(&mut dataset, 2, 100, &optimizer, true);
        // println!(
        //     "After train prediction: {:#?}",
        //     simple_dnn.predict(dataset.train_data().input)
        // );
    }
}

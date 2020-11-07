use mlp_rust::prelude::*;
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
fn test_mlp_regressor_train() {
    let data = arr2(&vec![[0.5f32, 0.05, 1.], [0.0, 0.0, 0.], [-0.5, -0.5, -1.]]).into_dyn();
    let mut dataset =
        Box::new(DataSetInMemory::new(data, 1, 1., false)) as Box<dyn DataSet<f32, IxDyn>>;
    let mut simple_dnn = MLP::new_regressor(2, 1, vec![25, 4], Activation::ReLu, false, false);
    let optimizer = Box::new(GradientDescent::new(0.01f32)) as Box<dyn Optimizer<f32>>;
    println!(
        "Before train prediction: {:#?}",
        simple_dnn.predict(dataset.train_data().0)
    );
    simple_dnn.train(&mut dataset, 2, 100, &optimizer, false);
    println!(
        "After train prediction: {:#?}",
        simple_dnn.predict(dataset.train_data().0)
    );
}

#[test]
fn test_mlp_classifier_train() {
    let data = arr2(&vec![[0.5f32, 0.05, 0.], [0.0, 1.0, 0.], [-0.5, -0.5, 1.]]).into_dyn();
    let mut dataset =
        Box::new(DataSetInMemory::new(data, 1, 1., false)) as Box<dyn DataSet<f32, IxDyn>>;
    let mut simple_dnn = MLP::new_classifier(2, 1, vec![25, 4], Activation::ReLu, false, false);
    let optimizer = Box::new(GradientDescent::new(0.01f32)) as Box<dyn Optimizer<f32>>;
    println!(
        "Before train prediction: {:#?}",
        simple_dnn.predict(dataset.train_data().0)
    );
    simple_dnn.train(&mut dataset, 2, 100, &optimizer, false);
    println!(
        "After train prediction: {:#?}",
        simple_dnn.predict(dataset.train_data().0)
    );
}

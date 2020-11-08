use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    // Mimicking MNIST dataset size.
    let sample_size = 60_000usize;
    let feature_size = 28 * 28;
    let output_size = 1;
    let data = Array::random(
        (sample_size, feature_size + output_size),
        Uniform::new(-0f32, 1.),
    )
    .into_dyn();
    let mut dataset = Box::new(DataSetInMemory::new(data, output_size, 0.2, true))
        as Box<dyn DataSet<f32, IxDyn>>;
    let hidden_layer_sizes = vec![4096, 2048, 1024, 512, 256];
    let mut simple_dnn = MLP::new_regressor(
        feature_size,
        output_size,
        hidden_layer_sizes,
        Activation::ReLu,
        false,
        false,
    );
    let optimizer = Box::new(GradientDescent::new(0.00f32)) as Box<dyn Optimizer<f32>>;
    // println!(
    //     "Before train prediction: {:#?}",
    //     simple_dnn.par_predict(dataset.test_data().0)
    // );
    simple_dnn.train(&mut dataset, 4096, 5, &optimizer, true);
    // println!(
    //     "After train prediction: {:#?}",
    //     simple_dnn.par_predict(dataset.test_data().0)
    // );
}

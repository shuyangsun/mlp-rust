extern crate ndarray;

use mlp_rust::ActivationFunction;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    let rand_arr: Array2<f64> = Array::random((1, 5), Uniform::new(0., 1.));
    let res = ActivationFunction::Softmax.forward(&rand_arr);
    println!("Rand arr: {}", rand_arr);
    println!("New arr: {}", res);
}

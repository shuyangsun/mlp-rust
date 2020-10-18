extern crate ndarray;

use mlp_rust::{ActivationFunction, Tensor};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    let rand_arr: Box<Array2<f64>> = Box::new(Array::random((1, 5), Uniform::new(0., 1.)));
    let res = ActivationFunction::ReLu.forward(&rand_arr);
    println!("Rand arr: {}", rand_arr.as_ref());
    println!("New arr: {}", res);
}

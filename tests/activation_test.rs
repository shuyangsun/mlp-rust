use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[test]
fn test_relu_forward() {
    let rand_arr = arr2(&[[-1., 2.]]).into_dyn();
    let forward_res = Activation::ReLu.forward(rand_arr.view());
    let par_forward_res = Activation::ReLu.forward(rand_arr.view());
    assert_eq!(forward_res, arr2(&[[0., 2.]]).into_dyn());
    assert_eq!(forward_res, par_forward_res);
}

#[test]
fn test_relu_forward_random_arr() {
    let shape = [2, 5];
    let rand_arr = Array::random(shape, Uniform::new(0., 10.)).into_dyn();
    let forward_res = Activation::ReLu.forward(rand_arr.view());
    let par_forward_res = Activation::ReLu.forward(rand_arr.view());
    assert_eq!(forward_res.shape(), &shape);
    assert_eq!(forward_res, par_forward_res);
}

#[test]
fn test_relu_backward() {
    let rand_arr = arr2(&[[-1., 2.]]).into_dyn();
    let forward_res = Activation::ReLu.forward(rand_arr.view());
    let par_forward_res = Activation::ReLu.forward(rand_arr.view());
    let backward_res =
        Activation::ReLu.backward_respect_to_input(rand_arr.view(), forward_res.view());
    assert_eq!(backward_res, arr2(&[[0., 2.]]).into_dyn());
    assert_eq!(forward_res, par_forward_res);
}

#[test]
fn test_leaky_relu_forward() {
    let rand_arr = Box::new(arr2(&[[-1., 2.]])).into_dyn();
    let forward_res = Activation::LeakyReLu.forward(rand_arr.view());
    let par_forward_res = Activation::LeakyReLu.forward(rand_arr.view());
    assert_eq!(forward_res, arr2(&[[-0.1, 2.]]).into_dyn());
    assert_eq!(forward_res, par_forward_res);
}

#[test]
fn test_leaky_relu_backward() {
    let rand_arr = Box::new(arr2(&[[-1., 2.]])).into_dyn();
    let forward_res = Activation::LeakyReLu.forward(rand_arr.view());
    let par_forward_res = Activation::LeakyReLu.forward(rand_arr.view());
    let backward_res =
        Activation::LeakyReLu.backward_respect_to_input(rand_arr.view(), forward_res.view());
    assert_eq!(backward_res, arr2(&[[0.1, 2.]]).into_dyn());
    assert_eq!(forward_res, par_forward_res);
}

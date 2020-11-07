use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[test]
fn test_bias_forward() {
    let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
    let bias = Bias::new(2);
    let forward_res = bias.forward(arr.view());
    assert_eq!(forward_res.ndim(), 2usize);
    assert_eq!(&forward_res, arr);
}

#[test]
fn test_bias_forward_rand() {
    let shape = [1000, 100];
    let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
    let weights = Bias::new(100);
    let forward_res = weights.forward(rand_arr.view());
    assert_eq!(forward_res.ndim(), 2usize);
    assert_eq!(&forward_res, rand_arr);
}

use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[test]
fn test_weights_forward() {
    let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
    let weights = Dense::new_random_uniform(2, 5);
    let forward_res = weights.forward(arr.view());
    let par_forward_res = weights.par_forward(arr.view());
    assert_eq!(forward_res.ndim(), 2usize);
    assert_eq!(forward_res.shape(), &[3usize, 5usize]);
    assert_eq!(forward_res, par_forward_res);
}

#[test]
fn test_weights_forward_rand_1() {
    let shape = [997, 100]; // 997 is a prime number, testing parallel splitting.
    let output_size = 50;
    let rand_arr = &Array::random(shape, Uniform::new(0., 10.)).into_dyn();
    let weights = Dense::new_random_uniform(shape[1], output_size);
    let forward_res = weights.forward(rand_arr.view());
    let par_forward_res = weights.par_forward(rand_arr.view());
    assert_eq!(forward_res.ndim(), 2usize);
    assert_eq!(forward_res.shape(), &[shape[0], output_size]);
    assert_eq!(forward_res, par_forward_res);
}

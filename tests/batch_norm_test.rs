use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[test]
fn test_batch_norm_forward_random_arr() {
    let shape = [10, 5];
    let rand_arr = Array2::random(shape, Uniform::new(-10., 10.)).into_dyn();
    let batch_norm = BatchNormalization::new(5);
    let forward_res = batch_norm.forward(rand_arr.view());
    assert_eq!(forward_res.shape(), &shape);
}

#[test]
fn test_batch_norm_forward_consistency() {
    let arr_1 = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
    let arr_2 = &arr2(&[[1.1, -2.], [-1.3, 2.1], [100., 0.5]]).into_dyn();
    let batch_norm_1 = BatchNormalization::new(2);
    let forward_res_1 = batch_norm_1.forward(arr_1.view());
    let batch_norm_2 = BatchNormalization::new(2);
    let forward_res_2 = batch_norm_2.forward(arr_2.view());
    assert_eq!(
        forward_res_1.index_axis(Axis(1), 1), // forward_res_1[:, 1]
        forward_res_2.index_axis(Axis(1), 1)  // forward_res_2[:, 1]
    );
}

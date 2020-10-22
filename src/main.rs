extern crate ndarray;
use ndarray::prelude::*;

fn main() {
    let arr_1 = &arr2(&[[1., 2.], [3., 4.]]);
    let arr_2 = &arr2(&[[10., 20.]]);
    let res_1 = arr_1 + arr_2;
    // let _ = arr_2 + arr_1; // Error: thread 'main' panicked at 'ndarray: could not broadcast array from shape: [2, 2] to: [1, 2]'
    println!("arr_1 + arr_2  = {}", res_1);
}

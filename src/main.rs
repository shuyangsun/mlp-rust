extern crate ndarray;
use ndarray::prelude::*;

fn main() {
    let arr_1 = &arr2(&[[1., 2.], [3., 4.]]);
    let arr_2 = &arr2(&[[10., 20.]]);
    let res_1 = arr_1 + arr_2;
    // let _ = arr_2 + arr_1; // Error: thread 'main' panicked at 'ndarray: could not broadcast array from shape: [2, 2] to: [1, 2]'
    println!("arr_1 + arr_2  = {}", res_1);

    let mut asdf = arr2(&[[1., 2.], [3., 4.]]);
    asdf[[0, 0]] = 100.;
    let mut asdf_dyn = asdf.into_dyn();
    let view_1 = asdf_dyn.view_mut();
    println!("View 1: {}", view_1);
    let mut back: ArrayViewMut2<f64> = view_1.into_dimensionality().unwrap();
    back[[0, 1]] = 1000.;
    println!("Back: {}", back);
    println!("asdf_dyn: {}", asdf_dyn);
    let mut cde = asdf_dyn.view_mut();
    cde.assign(&arr2(&[[2., 2.], [2., 2.]]));
    println!("{}", asdf_dyn.view());
    let fdddsfsf: ArrayView2<f64> = asdf_dyn.view().into_dimensionality().unwrap();
    println!("{}", fdddsfsf);
}

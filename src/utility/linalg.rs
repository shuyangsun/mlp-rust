extern crate ndarray;
extern crate num_cpus;
use ndarray::prelude::*;
use ndarray::{stack, Slice};
use rayon::prelude::*;

use self::ndarray::RemoveAxis;
use super::super::traits::numerical_traits::MLPFloat;

pub fn mat_mul<T>(lhs: &ArrayView2<T>, rhs: &ArrayView2<T>) -> Array2<T>
where
    T: MLPFloat,
{
    assert_eq!(lhs.ndim(), 2);
    assert_eq!(rhs.ndim(), 2);
    assert_eq!(lhs.ncols(), rhs.nrows());
    lhs.dot(rhs).into_dimensionality().unwrap()
}

pub fn par_mat_mul<T>(lhs: &ArrayView2<T>, rhs: &ArrayView2<T>) -> Array2<T>
where
    T: MLPFloat,
{
    assert_eq!(lhs.ndim(), 2);
    assert_eq!(rhs.ndim(), 2);
    assert_eq!(lhs.ncols(), rhs.nrows());
    par_arr_operation(&lhs, |lhs_sub: &ArrayView2<T>| mat_mul(lhs_sub, &rhs))
}

fn par_arr_operation<'a, T, D, F>(arr_view: &'a ArrayView<T, D>, operation: F) -> Array<T, D>
where
    T: Copy + Send + Sync,
    D: Dimension + RemoveAxis,
    F: Fn(&ArrayView<'a, T, D>) -> Array<T, D> + Send + Sync,
{
    let thread_count = num_cpus::get();
    let view_sliced: Vec<ArrayView<'a, T, D>> =
        split_arr_view_into_chunks_by_axis0(&arr_view, thread_count);
    let par_res: Vec<Array<T, D>> = view_sliced.par_iter().map(operation).collect();
    stack_arr_views(&par_res.iter().map(|ele| ele.view()).collect())
}

pub fn split_arr_view_into_chunks_by_axis0<'a, T, D>(
    arr: &'a ArrayView<T, D>,
    num_chunks: usize,
) -> Vec<ArrayView<'a, T, D>>
where
    D: ndarray::Dimension,
{
    assert!(arr.ndim() > 0);
    let mut res: Vec<ArrayView<'a, T, D>> = Vec::new();
    let nrows = arr.shape()[0];
    let mut num_sample_per_chunk = std::cmp::max(nrows / num_chunks, 1);
    if nrows > num_chunks && nrows % num_chunks > 0 {
        num_sample_per_chunk += 1;
    }
    for i in 0..num_chunks {
        let lower_idx = i * num_sample_per_chunk;
        let upper_idx = std::cmp::min(lower_idx + num_sample_per_chunk, nrows);
        res.push(arr.slice_axis(Axis(0), Slice::from(lower_idx..upper_idx)));
        if lower_idx + num_sample_per_chunk >= nrows {
            break;
        }
    }
    res
}

pub fn stack_arr_views<T, D>(arr_vec: &Vec<ArrayView<T, D>>) -> Array<T, D>
where
    T: Copy,
    D: Dimension + RemoveAxis,
{
    stack(Axis(0), arr_vec.as_slice()).unwrap()
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;

    use super::{mat_mul, par_mat_mul};
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_mat_mul_1() {
        let lhs = arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]);
        let rhs = arr2(&[[1.5, -2., 1., 2.], [1.3, 2.1, 9., -8.2]]);
        let res = mat_mul(&lhs.view(), &rhs.view());
        let par_res = par_mat_mul(&lhs.view(), &rhs.view());
        assert_eq!(res.shape(), &[3, 4]);
        assert_eq!(res, par_res);
    }

    #[test]
    fn test_mat_mul_random_1() {
        let shape_1 = [100, 10];
        let shape_2 = [10, 5];
        let lhs = Array2::random(shape_1, Uniform::new(-1., 1.));
        let rhs = Array2::random(shape_2, Uniform::new(-1., 1.));
        let res = mat_mul(&lhs.view(), &rhs.view());
        let par_res = par_mat_mul(&lhs.view(), &rhs.view());
        assert_eq!(res.shape(), &[shape_1[0], shape_2[1]]);
        assert_eq!(res, par_res);
    }

    #[test]
    fn test_mat_mul_random_stress() {
        let shape_1 = [1000, 1000];
        let shape_2 = [1000, 500];
        let lhs = Array2::random(shape_1, Uniform::new(-1., 1.));
        let rhs = Array2::random(shape_2, Uniform::new(-1., 1.));
        let res = mat_mul(&lhs.view(), &rhs.view());
        assert_eq!(res.shape(), &[shape_1[0], shape_2[1]]);
    }

    #[test]
    fn test_par_mat_mul_random_stress() {
        let shape_1 = [1000, 1000];
        let shape_2 = [1000, 500];
        let lhs = Array2::random(shape_1, Uniform::new(-1., 1.));
        let rhs = Array2::random(shape_2, Uniform::new(-1., 1.));
        let par_res = par_mat_mul(&lhs.view(), &rhs.view());
        assert_eq!(par_res.shape(), &[shape_1[0], shape_2[1]]);
    }
}

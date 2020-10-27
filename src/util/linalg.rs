extern crate ndarray;
extern crate num_cpus;
use ndarray::prelude::*;
use ndarray::stack;
use rayon::prelude::*;

use super::super::custom_types::numerical_traits::MLPFloat;

pub fn mat_mul<T>(lhs: ArrayView2<T>, rhs: ArrayView2<T>) -> Array2<T>
where
    T: MLPFloat,
{
    assert_eq!(lhs.ndim(), 2);
    assert_eq!(rhs.ndim(), 2);
    assert_eq!(lhs.ncols(), rhs.nrows());
    let mat_mul_res = lhs.dot(&rhs).into_dimensionality().unwrap();
    assert_eq!(mat_mul_res.ndim(), 2);
    mat_mul_res
}

pub fn par_mat_mul<T>(lhs: ArrayView2<T>, rhs: ArrayView2<T>) -> Array2<T>
where
    T: MLPFloat,
{
    assert_eq!(lhs.ndim(), 2);
    assert_eq!(rhs.ndim(), 2);
    assert_eq!(lhs.ncols(), rhs.nrows());
    let mut lhs_sliced: Vec<ArrayView2<T>> = Vec::new();
    let thread_count = num_cpus::get();
    let mut num_sample_per_thread = std::cmp::max(lhs.nrows() / thread_count, 1);
    if lhs.nrows() > thread_count && lhs.nrows() % thread_count > 0 {
        num_sample_per_thread += 1;
    }
    for i in 0..thread_count {
        let lower_idx = i * num_sample_per_thread;
        let upper_idx = std::cmp::min(lower_idx + num_sample_per_thread, lhs.nrows());
        lhs_sliced.push(lhs.slice(s![lower_idx..upper_idx, ..]));
        if lower_idx + num_sample_per_thread >= lhs.nrows() {
            break;
        }
    }
    let mat_mul_res_par: Vec<Array2<T>> = lhs_sliced
        .par_iter()
        .map(|lhs_sub: &ArrayView2<T>| lhs_sub.dot(&rhs))
        .collect();
    let mat_mul_res_par_view: Vec<ArrayView2<T>> =
        mat_mul_res_par.iter().map(|ele| ele.view()).collect();
    let mat_mul_res = stack(Axis(0), mat_mul_res_par_view.as_slice()).unwrap();
    assert_eq!(mat_mul_res.ndim(), 2);
    mat_mul_res
}

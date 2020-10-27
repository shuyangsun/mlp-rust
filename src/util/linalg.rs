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
        let res = mat_mul(lhs.view(), rhs.view());
        let par_res = par_mat_mul(lhs.view(), rhs.view());
        assert_eq!(res.shape(), &[3, 4]);
        assert_eq!(res, par_res);
    }

    #[test]
    fn test_mat_mul_random_1() {
        let shape_1 = (100, 10);
        let shape_2 = (10, 5);
        let lhs = Array2::random(shape_1, Uniform::new(-1., 1.));
        let rhs = Array2::random(shape_2, Uniform::new(-1., 1.));
        let res = mat_mul(lhs.view(), rhs.view());
        let par_res = par_mat_mul(lhs.view(), rhs.view());
        assert_eq!(res.shape(), &[shape_1.0, shape_2.1]);
        assert_eq!(res, par_res);
    }

    #[test]
    fn test_mat_mul_random_stress() {
        let shape_1 = (1000, 1000);
        let shape_2 = (1000, 500);
        let lhs = Array2::random(shape_1, Uniform::new(-1., 1.));
        let rhs = Array2::random(shape_2, Uniform::new(-1., 1.));
        let res = mat_mul(lhs.view(), rhs.view());
        assert_eq!(res.shape(), &[shape_1.0, shape_2.1]);
    }

    #[test]
    fn test_par_mat_mul_random_stress() {
        let shape_1 = (10000, 1000);
        let shape_2 = (1000, 5000);
        let lhs = Array2::random(shape_1, Uniform::new(-1., 1.));
        let rhs = Array2::random(shape_2, Uniform::new(-1., 1.));
        let par_res = par_mat_mul(lhs.view(), rhs.view());
        assert_eq!(par_res.shape(), &[shape_1.0, shape_2.1]);
    }
}

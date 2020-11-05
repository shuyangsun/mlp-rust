use crate::MLPFloat;
use ndarray::{Array, ArrayView2, ArrayViewD, Axis, Dimension, Ix2, RemoveAxis};
use rand::Rng;

pub fn eps<T>() -> T
where
    T: MLPFloat,
{
    T::from_f32(std::f32::EPSILON).unwrap()
}

pub fn tanh_safe<T>(value: &T) -> T
where
    T: MLPFloat,
{
    // 8.32 is the input threshold for tanh(x) absolute value to be (1f32 - eps()).
    // Given eps() = std::f32::EPSILON = 0.00000012;
    let threshold = T::from_f32(8.32).unwrap();
    if value.abs() > threshold {
        if value > &T::zero() {
            T::one()
        } else {
            -T::one()
        }
    } else {
        value.sinh() / value.cosh()
    }
}

pub fn calculate_std_from_variance<T, D>(
    variance: &Array<T, D>,
    should_be_parallel: bool,
) -> Array<T, D>
where
    D: Dimension,
    T: MLPFloat,
{
    let mut std_stable = variance.clone();
    if should_be_parallel {
        std_stable.par_mapv_inplace(|ele| (ele + eps()).sqrt());
    } else {
        std_stable.mapv_inplace(|ele| (ele + eps()).sqrt());
    }
    std_stable
}

pub fn to_2d_view<T>(arr_view: ArrayViewD<T>) -> ArrayView2<T> {
    arr_view.into_dimensionality::<Ix2>().unwrap()
}

pub fn shuffle_array<T, D>(arr: &mut Array<T, D>)
where
    D: Dimension + RemoveAxis,
    T: MLPFloat,
{
    let mut rng = rand::thread_rng();
    let nrows = arr.shape()[0];
    for _ in 0..nrows {
        let idx_a = rng.gen::<usize>() % nrows;
        let idx_b: usize = rng.gen::<usize>() % nrows;
        let first_clone = arr.index_axis(Axis(0), idx_a).into_owned();
        let second_clone = arr.index_axis(Axis(0), idx_b).into_owned();
        let mut first_row = arr.index_axis_mut(Axis(0), idx_a);
        first_row.assign(&second_clone);
        let mut second_row = arr.index_axis_mut(Axis(0), idx_b);
        second_row.assign(&first_clone);
    }
}

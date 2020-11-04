use crate::MLPFloat;
use ndarray::ArrayD;

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

pub fn calculate_std_from_variance<T>(variance: &ArrayD<T>, should_be_parallel: bool) -> ArrayD<T>
where
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

extern crate ndarray;
use super::numerical_traits::MLPFloat;
use ndarray::prelude::*;
use ndarray::Zip;

pub trait Optimizer<T>
where
    T: MLPFloat,
{
    fn modify_inplace(&self, gradient: &mut ArrayViewMutD<T>);

    fn modify(&self, gradient: ArrayViewD<T>) -> ArrayD<T> {
        let mut res = gradient.into_owned();
        self.modify_inplace(&mut res.view_mut());
        res
    }

    fn change_values(&self, old_value: &mut ArrayViewMutD<T>, gradient: ArrayViewD<T>) {
        let gradient_mean = gradient.mean_axis(Axis(0)).unwrap();
        let diff = self.modify(gradient_mean.view());
        let diff_view = diff.broadcast(old_value.dim()).unwrap();
        let zip = Zip::from(old_value).and(&diff_view);
        zip.apply(|old, delta| {
            *old = *old - *delta;
        });
    }
}

extern crate ndarray;
use super::numerical_traits::MLPFloat;
use ndarray::{Array, ArrayView, ArrayViewMut, Axis, Zip};

pub trait Optimizer<T, D>
where
    T: MLPFloat,
{
    fn modify_inplace(&self, gradient: &mut ArrayViewMut<T, D>);

    fn modify(&self, gradient: ArrayView<T, D>) -> Array<T, D> {
        let mut res = gradient.into_owned();
        self.modify_inplace(&mut res.view_mut());
        res
    }

    fn change_values(&self, old_value: &mut ArrayViewMut<T, D>, gradient: ArrayView<T, D>) {
        let gradient_mean = gradient.mean_axis(Axis(0)).unwrap();
        let diff = self.modify(gradient_mean.view());
        let diff_view = diff.broadcast(old_value.dim()).unwrap();
        let zip = Zip::from(old_value).and(&diff_view);
        zip.apply(|old, delta| {
            *old = *old - *delta;
        });
    }
}

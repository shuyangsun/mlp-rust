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
        assert_eq!(old_value.shape(), gradient.shape());
        let diff = self.modify(gradient);
        let diff_view = diff.view();
        let zip = Zip::from(old_value).and(&diff_view);
        zip.apply(|old, delta| {
            *old = *old - *delta;
        });
    }
}

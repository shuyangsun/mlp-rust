extern crate ndarray;
use super::super::traits::numerical_traits::MLPFloat;
use super::super::traits::tensor_traits::Tensor;
use ndarray::prelude::*;
use std::cell::RefCell;

pub struct BatchNormalization<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    size: usize,
    mean: RefCell<Option<ArrayD<T>>>,
    std_stable: RefCell<Option<ArrayD<T>>>,
    gama: T,
    beta: T,
}

impl<T> BatchNormalization<T>
where
    T: MLPFloat,
{
    fn forward_helper(&self, input: ArrayViewD<T>, is_parallel: bool) -> ArrayD<T> {
        let eps = T::from_f32(1e-7f32).unwrap();
        if self.mean.borrow().is_none() {
            let mean: ArrayD<T> = input.mean_axis(Axis(0)).unwrap();

            let mut std_stable: ArrayD<T> = (&input - &mean).into_dimensionality().unwrap();
            if is_parallel {
                std_stable.par_mapv_inplace(|ele| ele.powi(2));
            } else {
                std_stable.mapv_inplace(|ele| ele.powi(2));
            }
            let mut std_stable = std_stable.mean_axis(Axis(0)).unwrap();
            if is_parallel {
                std_stable.par_mapv_inplace(|ele| (ele + eps).sqrt());
            } else {
                std_stable.mapv_inplace(|ele| (ele + eps).sqrt());
            }
            self.update_mean(mean);
            self.update_std(std_stable);
        };
        let mut input_normalized = (&input - &self.mean.borrow().as_ref().unwrap().view())
            / &self.std_stable.borrow().as_ref().unwrap().view();
        let gama_clone = self.gama.clone();
        let beta_clone = self.beta.clone();
        if is_parallel {
            input_normalized.par_mapv_inplace(|ele| ele * gama_clone + beta_clone);
        } else {
            input_normalized.mapv_inplace(|ele| ele * gama_clone + beta_clone);
        }
        input_normalized
    }
}

impl<T> Tensor<T> for BatchNormalization<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, false)
    }

    fn backward_batch(&self, output: ArrayViewD<T>) -> ArrayD<T> {
        unimplemented!()
    }

    fn updatable_mat(&mut self) -> ArrayViewMutD<'_, T> {
        unimplemented!()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        self.forward_helper(input, true)
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn num_param(&self) -> Option<usize> {
        Some((self.size + 1) * 2)
    }
}

impl<T> BatchNormalization<T>
where
    T: MLPFloat,
{
    pub fn new(size: usize) -> Self {
        Self {
            is_frozen: false,
            size,
            mean: RefCell::new(None),
            std_stable: RefCell::new(None),
            gama: T::one(),
            beta: T::zero(),
        }
    }

    pub fn new_frozen(size: usize) -> Self {
        let mut res = Self::new(size);
        res.is_frozen = true;
        res
    }

    fn update_mean(&self, mean: ArrayD<T>) {
        assert_eq!(mean.len(), self.size);
        self.mean.borrow_mut().get_or_insert(mean);
    }

    fn update_std(&self, std: ArrayD<T>) {
        assert_eq!(std.len(), self.size);
        self.std_stable.borrow_mut().get_or_insert(std);
    }
}

#[cfg(test)]
mod unit_test {
    extern crate ndarray;
    use super::super::super::traits::tensor_traits::Tensor;
    use super::BatchNormalization;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_batch_norm_forward() {
        let arr = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let batch_norm = BatchNormalization::new(2);
        let forward_res = batch_norm.forward(arr.view());
        assert_eq!(
            forward_res,
            arr2(&[
                [1.224742575001387, -1.303930263591683],
                [0.0, 1.1261215912837261],
                [-1.224742575001387, 0.17780867230795672]
            ])
            .into_dyn()
        );
    }

    #[test]
    fn test_batch_norm_forward_random_arr() {
        let shape = [10, 5];
        let rand_arr = Array2::random(shape, Uniform::new(-10., 10.)).into_dyn();
        let batch_norm = BatchNormalization::new(5);
        let forward_res = batch_norm.forward(rand_arr.view());
        assert_eq!(forward_res.shape(), &shape);
    }

    #[test]
    fn test_batch_norm_forward_consistency() {
        let arr_1 = &arr2(&[[1.5, -2.], [1.3, 2.1], [1.1, 0.5]]).into_dyn();
        let arr_2 = &arr2(&[[1.1, -2.], [-1.3, 2.1], [100., 0.5]]).into_dyn();
        let batch_norm_1 = BatchNormalization::new(2);
        let forward_res_1 = batch_norm_1.forward(arr_1.view());
        let batch_norm_2 = BatchNormalization::new(2);
        let forward_res_2 = batch_norm_2.forward(arr_2.view());
        assert_eq!(
            forward_res_1.index_axis(Axis(1), 1), // forward_res_1[:, 1]
            forward_res_2.index_axis(Axis(1), 1)  // forward_res_2[:, 1]
        );
    }
}

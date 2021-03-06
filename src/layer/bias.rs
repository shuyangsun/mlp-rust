use crate::utility::counter::CounterEst;
use crate::{MLPFLoatRandSampling, MLPFloat, Optimizer, Tensor};
use ndarray::{Array2, ArrayD, ArrayViewD, Zip};

pub struct Bias<T>
where
    T: MLPFloat,
{
    is_frozen: bool,
    bias_arr: ArrayD<T>, // 2D array with shape 1 x N, N = number of neurons
}

impl<T> Tensor<T> for Bias<T>
where
    T: MLPFloat,
{
    fn forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        let bias_arr_broadcasted_view = self.bias_arr.broadcast(input.dim()).unwrap();
        &input + &bias_arr_broadcasted_view
    }

    fn backward_respect_to_input(
        &self,
        _: ArrayViewD<T>,
        layer_output: ArrayViewD<T>,
    ) -> ArrayD<T> {
        layer_output.into_owned()
    }

    fn par_forward(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        let bias_arr_broadcasted_view = self.bias_arr.broadcast(input.dim()).unwrap();
        let mut res = input.into_owned();
        Zip::from(&mut res)
            .and(&bias_arr_broadcasted_view)
            .apply(|a, &b| *a = *a + b);
        res
    }

    fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    fn backward_update(
        &mut self,
        _: ArrayViewD<T>,
        output_gradient: ArrayViewD<T>,
        optimizer: &Box<dyn Optimizer<T>>,
    ) {
        optimizer.change_values(&mut self.bias_arr.view_mut(), output_gradient);
    }

    fn num_parameters(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.bias_arr.len())
    }

    fn num_operations_per_forward(&self) -> CounterEst<usize> {
        CounterEst::Accurate(self.bias_arr.len())
    }

    fn to_frozen(&self) -> Box<dyn Tensor<T> + Sync> {
        Box::new(Self {
            is_frozen: true,
            bias_arr: self.bias_arr.clone(),
        })
    }
}

impl<T> Bias<T>
where
    T: MLPFloat + MLPFLoatRandSampling,
{
    pub fn new(size: usize) -> Self {
        Self {
            is_frozen: false,
            bias_arr: Array2::zeros((1, size)).into_dyn(),
        }
    }

    pub fn new_frozen(size: usize) -> Self {
        let mut res = Self::new(size);
        res.is_frozen = true;
        res
    }
}

#[macro_export]
macro_rules! bias {
    ($x:expr) => {{
        Box::new(Bias::new($x))
    }};
}

use crate::utility::{counter::CounterEst, linalg::par_arr_operation};
use crate::{DataSet, LayerChain, Loss, MLPFLoatRandSampling, MLPFloat, Model, Optimizer, Tensor};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub struct Serial<T>
where
    T: MLPFloat,
{
    layer_chain: LayerChain<T>,
    layer_chain_frozen: Option<Box<dyn Tensor<T> + Sync>>,
    loss_function: Loss,
}

impl<T> Serial<T>
where
    T: MLPFloat,
{
    pub fn add(&mut self, layer: Box<dyn Tensor<T>>) {
        self.layer_chain.push(layer);
    }

    pub fn add_all<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(&mut self, layers: I) {
        self.layer_chain.push_all(layers)
    }

    pub fn num_param(&self) -> CounterEst<usize> {
        self.layer_chain.num_parameters()
    }

    pub fn num_operations_per_forward(&self) -> CounterEst<usize> {
        self.layer_chain.num_operations_per_forward()
    }

    fn update_frozen(&mut self) {
        self.layer_chain_frozen = Some(self.layer_chain.to_frozen());
    }
}

impl<T> Serial<T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new(loss_function: Loss) -> Self {
        Self::new_from_layers(vec![], loss_function)
    }

    pub fn new_from_layers<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(
        layers: I,
        loss_function: Loss,
    ) -> Self {
        let mut res = Self {
            layer_chain: LayerChain::new_from_sublayers(layers),
            layer_chain_frozen: None,
            loss_function,
        };
        res.update_frozen();
        res
    }
}

impl<T> Model<T> for Serial<T>
where
    T: MLPFloat,
{
    fn train(
        &mut self,
        data: &mut Box<dyn DataSet<T, IxDyn>>,
        batch_size: usize,
        max_num_epoch: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        should_print: bool,
    ) {
        let mut cur_iter = 0usize;
        for cur_epoch in 0..max_num_epoch {
            for batch in data.next_train_batch(batch_size) {
                let forward_res = self.layer_chain.par_forward(batch.0.clone());
                assert_eq!(forward_res.shape(), batch.1.shape());
                let l2_reg_cost = T::from_f32(0.0).unwrap() * self.layer_chain.dense_l2_sum();
                let mut gradient = self.loss_function.backward_with_respect_to_input(
                    forward_res.view(),
                    batch.1.clone(),
                    true,
                );
                gradient.par_mapv_inplace(|ele| ele + l2_reg_cost);
                if should_print {
                    let cur_loss =
                        self.loss_function
                            .calculate_loss(forward_res.view(), batch.1, true)
                            + l2_reg_cost;
                    println!("epoch {}, iter {}, loss={}", cur_epoch, cur_iter, cur_loss);
                }
                self.layer_chain
                    .backward_update_check_frozen(batch.0, gradient.view(), optimizer);
                cur_iter += 1;
            }
            data.shuffle_train();
        }
        self.update_frozen();
    }

    fn predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        let layers = self.layer_chain_frozen.as_ref().unwrap();
        self.loss_function
            .predict(layers.forward(input).view(), false)
    }

    fn par_predict(&self, input: ArrayViewD<T>) -> ArrayD<T> {
        let loss = self.loss_function.clone();
        let layers = self.layer_chain_frozen.as_ref().unwrap();
        par_arr_operation(&input, |arr: &ArrayViewD<T>| {
            loss.predict(layers.forward(arr.clone()).view(), false)
        })
    }
}

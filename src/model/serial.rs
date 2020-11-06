use crate::utility::counter::CounterEst;
use crate::{DataSet, LayerChain, Loss, MLPFLoatRandSampling, MLPFloat, Model, Optimizer, Tensor};
use ndarray::{Array, ArrayView, Dimension};

pub struct Serial<T, D>
where
    T: MLPFloat,
{
    layer_chain: LayerChain<T, D>,
    loss_function: Loss,
}

impl<T> Serial<T>
where
    T: MLPFLoatRandSampling,
{
    pub fn new(loss_function: Loss) -> Self {
        Self {
            layer_chain: LayerChain::new(),
            loss_function,
        }
    }

    pub fn new_from_layers<I: IntoIterator<Item = Box<dyn Tensor<T>>>>(
        layers: I,
        loss_function: Loss,
    ) -> Self {
        Self {
            layer_chain: LayerChain::new_from_sublayers(layers),
            loss_function,
        }
    }

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
}

impl<T, InputD, OutputD> Model<T, InputD, OutputD> for Serial<T>
where
    InputD: Dimension,
    OutputD: Dimension,
    T: MLPFloat,
{
    fn train<'data, 'model>(
        &'model mut self,
        data: &'data mut Box<dyn DataSet<'data, T, InputD, OutputD>>,
        max_num_epoch: usize,
        batch_size: usize,
        optimizer: &Box<dyn Optimizer<T>>,
        should_print_loss: bool,
    ) where
        'data: 'model,
    {
        let mut iter_idx = 0usize;
        for epoch_idx in 0..max_num_epoch {
            for batch in data.next_train_batch(batch_size) {
                let forward_res = self.layer_chain.forward(batch.input.into_dyn());
                assert_eq!(forward_res.shape(), batch.output.shape());
                // TODO: super hacky
                let l2_reg_cost = T::from_f32(0.0).unwrap() * self.layer_chain.dense_l2_sum();
                let mut gradient = self.loss_function.backward_with_respect_to_input(
                    forward_res.view(),
                    batch.output.into_dyn(),
                    true,
                );
                gradient.par_mapv_inplace(|ele| ele + l2_reg_cost);
                if should_print_loss {
                    let cur_loss = self.loss_function.calculate_loss(
                        forward_res.into_dyn().view(),
                        batch.output.into_dyn(),
                        true,
                    ) + l2_reg_cost;
                    println!("Epoch={}, iter={}, loss={}", epoch_idx, iter_idx, cur_loss);
                }
                self.layer_chain.backward_update_check_frozen(
                    batch.input.into_dyn(),
                    gradient.view(),
                    optimizer,
                );
                iter_idx += 1;
            }
        }
    }

    fn predict(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD> {
        self.loss_function
            .predict(self.layer_chain.predict(input.into_dyn()).view(), false)
            .into_dimensionality::<OutputD>()
            .unwrap()
    }

    fn par_predict(&self, input: ArrayView<T, InputD>) -> Array<T, OutputD> {
        self.loss_function
            .predict(
                self.layer_chain
                    .par_predict(input.into_dyn().into_dyn())
                    .view(),
                true,
            )
            .into_dimensionality::<OutputD>()
            .unwrap()
    }
}

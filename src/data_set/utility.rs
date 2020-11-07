// use crate::MLPFloat;
// use ndarray::{s, ArrayD, ArrayView, ArrayViewD, Dimension};
// use std::marker::PhantomData;
//
// pub struct InputOutputData<'a, T, D> {
//     pub input: ArrayView<'a, T, D>,
//     pub output: ArrayView<'a, T, D>,
// }
//
// impl<'a, T, D> InputOutputData<'a, T, D>
// where
//     D: Dimension,
// {
//     pub fn new(input: ArrayView<'a, T, D>, output: ArrayView<'a, T, D>) -> Self {
//         assert_eq!(input.ndim(), output.ndim());
//         assert_eq!(input.shape()[0], output.shape()[0]);
//         Self { input, output }
//     }
//
//     pub fn num_samples(&self) -> usize {
//         self.input.shape()[0]
//     }
// }
//
// pub struct DataBatch<'source_view, 'batch_view, T, D>
// where
//     'source_view: 'batch_view,
// {
//     cur_start_idx: usize,
//     batch_size: usize,
//     output_size: usize,
//     data_set: ArrayViewD<'source_view, T>,
//     _phantom: PhantomData<&'batch_view D>,
// }
//
// impl<'source_view, 'batch_view, T, D> DataBatch<'source_view, 'batch_view, T, D>
// where
//     'source_view: 'batch_view,
//     D: Dimension,
// {
//     pub fn new(
//         data_set: ArrayViewD<'source_view, T>,
//         batch_size: usize,
//         output_size: usize,
//     ) -> Self {
//         Self {
//             cur_start_idx: 0,
//             batch_size,
//             output_size,
//             data_set,
//             _phantom: PhantomData,
//         }
//     }
// }
//
// impl<'source_view, 'batch_view, T, D> Iterator for DataBatch<'source_view, 'batch_view, T, D>
// where
//     T: MLPFloat,
//     D: Dimension,
//     'source_view: 'batch_view,
// {
//     type Item = (ArrayViewD<'batch_view, T>, ArrayViewD<'batch_view, T>);
//
//     fn next(&mut self) -> Option<Self::Item> {
//         let (n_samples, n_cols) = (self.data_set.shape()[0], self.data_set.shape()[1]);
//         if self.cur_start_idx >= n_samples {
//             return None;
//         }
//         let start_row_idx = self.cur_start_idx;
//         let end_row_idx = std::cmp::min(self.cur_start_idx + self.batch_size, n_samples);
//         let output_col_idx = n_cols - self.output_size;
//         let input = self
//             .data_set
//             .clone()
//             .slice(s![start_row_idx..end_row_idx, ..output_col_idx])
//             .into_dyn();
//         let output = self
//             .data_set
//             .clone()
//             .slice(s![start_row_idx..end_row_idx, output_col_idx..])
//             .into_dyn();
//         self.cur_start_idx = end_row_idx;
//         Some((input, output))
//     }
// }

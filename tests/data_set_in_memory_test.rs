use mlp_rust::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[test]
fn test_data_set_1() {
    let shape = [997, 10];
    let input_data = Array::random(shape, Uniform::new(-1., 1.)).into_dyn();
    let mut dataset = DataSetInMemory::new(input_data, 2, 0.4, true);
    assert_eq!(dataset.num_samples(), shape[0]);
    assert_eq!(dataset.num_training_samples(), 598);
    assert_eq!(dataset.num_test_samples(), 399);
    let test_data = dataset.test_data();
    assert_eq!(test_data.0.shape()[0], 399);
    assert_eq!(test_data.1.shape()[0], test_data.0.shape()[0]);
    dataset.update_test_data_ratio(0.);

    let batch_size = 100usize;
    let mut total_sample = 0usize;
    let mut counter = 0usize;
    for batch in dataset.next_train_batch(batch_size) {
        if counter > 10 {
            break;
        }
        assert_eq!(batch.0.shape()[0], batch.1.shape()[0]);
        assert!(batch.0.shape()[0] <= batch_size);
        total_sample += batch.0.shape()[0];
        counter += 1;
    }
    assert_eq!(total_sample, shape[0]);
}

#[test]
fn test_data_set_shuffle() {
    let shape = [997, 10];
    let input_data = Array::random(shape, Uniform::new(-1., 1.)).into_dyn();
    let mut dataset = DataSetInMemory::new(input_data, 2, 0.4, true);
    let (train_1_input, train_1_output) = (
        dataset.train_data().0.into_owned(),
        dataset.train_data().1.into_owned(),
    );
    let (test_1_input, test_1_output) = (
        dataset.test_data().0.into_owned(),
        dataset.test_data().1.into_owned(),
    );
    dataset.shuffle_train();
    let (train_2_input, train_2_output) = (
        dataset.train_data().0.into_owned(),
        dataset.train_data().1.into_owned(),
    );
    let (test_2_input, test_2_output) = (
        dataset.test_data().0.into_owned(),
        dataset.test_data().1.into_owned(),
    );
    assert_ne!(train_1_input, train_2_input);
    assert_ne!(train_1_output, train_2_output);
    assert_eq!(test_1_input, test_2_input);
    assert_eq!(test_1_output, test_2_output);
    dataset.shuffle_all();
    let (train_3_input, train_3_output) = (
        dataset.train_data().0.into_owned(),
        dataset.train_data().1.into_owned(),
    );
    let (test_3_input, test_3_output) = (
        dataset.test_data().0.into_owned(),
        dataset.test_data().1.into_owned(),
    );
    assert_ne!(train_3_input, train_2_input);
    assert_ne!(train_3_output, train_2_output);
    assert_ne!(test_3_input, test_2_input);
    assert_ne!(test_3_output, test_2_output);
}

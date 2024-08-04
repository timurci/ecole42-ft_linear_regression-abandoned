extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::ReaderBuilder;
use ndarray::{array, s, stack, Array, Array1, Array2, Axis};
use ndarray_csv::{Array2Reader, ReadError};
use std::fs::File;

use std::env::args;
use std::path::Path;

fn read_csv(path: &str, has_headers: bool) -> Result<Array2<f32>, ReadError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) => panic!("Cannot open file \"{path}\": {err}"),
    };
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file);
    let array_read: Array2<f32> = reader.deserialize_array2_dynamic()?;

    Ok(array_read)
}

fn linear_estimate(estimators: &Array2<f32>, weights: &Array1<f32>) -> Array1<f32> {
    // Calculates linear estimates across all observations.

    let nrows = estimators.nrows();
    let mut estimates: Array1<f32> = Array::zeros((nrows,));

    for i in 0..nrows {
        estimates[i] = estimators.index_axis(Axis(0), i).dot(weights);
    }

    estimates
}

fn mean_sqsum_loss_gradient(
    estimators: &Array2<f32>,
    observations: &Array1<f32>,
    weights: &Array1<f32>,
) -> Array1<f32> {
    // loss = sum ( (x_i[] * theta_i[] - y_i)^2 ) where i is the number of observations
    // grad(loss(...)) = 1/m * sum ( 2 * (sum_col(x_i[] * theta_i[]) - y_i) * x_i[] )

    let diff = linear_estimate(&estimators, &weights) - observations;

    let mut tmp_weights = weights.clone();

    tmp_weights[0] = diff.sum();

    for j in 1..tmp_weights.len() {
        tmp_weights[j] = estimators.index_axis(Axis(1), j).dot(&diff);
    }

    tmp_weights / (observations.len() as f32)
}

fn gradient_descent2(data: &Array2<f32>, lr: f32, min_change: f32) -> Array1<f32> {
    const ITER_MAX: usize = 10000;
    let mut weights: Array1<f32> = array![0., 0.];
    let estimators = stack!(
        // Prepend 1 for x_0
        Axis(1),
        Array::ones((data.dim().0,)),
        data.index_axis(Axis(1), 0),
    );

    let price = data.index_axis(Axis(1), 1).to_owned();

    println!("Learning rate: {lr}");
    for _i in 0..ITER_MAX {
        let change = lr * mean_sqsum_loss_gradient(&estimators, &price, &weights);
        let change_abs: Array1<f32> = change.iter().map(|x| x.abs()).collect();
        let max = change_abs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if max < min_change {
            break;
        }

        weights = weights - change;
    }

    weights
}

fn print_help(prog_name: &str) {
    let file_name = Path::new(prog_name).file_name().unwrap().to_str().unwrap();
    println!("{file_name} <file path> <learning rate> [min. change threshold]");
}

fn main() {
    let path: String = args().nth(1).expect("no path provided");

    match path.as_str() {
        "-h" => {
            print_help(&args().nth(0).unwrap());
            return;
        }
        "--help" => {
            print_help(&args().nth(0).unwrap());
            return;
        }
        _ => {}
    }

    let learn_rate = args()
        .nth(2)
        .expect("no learning rate provided")
        .parse::<f32>()
        .expect("cannot parse learning rate");
    let min_change = match args().nth(3) {
        Some(val) => val.parse::<f32>().expect("cannot parse min_change"),
        None => 1e-5,
    };

    println!("path {path} lr {learn_rate} min {min_change}");

    let data = read_csv(&path, true).unwrap();

    println!(
        "File at {} loaded with dims {}, {}",
        path,
        data.dim().0,
        data.dim().1
    );

    println!("km    [0..5]: {:6}", data.slice(s![0..5, 0]));
    println!("price [0..5]: {:6}", data.slice(s![0..5, 1]));

    let weights = gradient_descent2(&data, learn_rate, min_change);

    println!("Trained weights : {:.5},{:.5}", weights[0], weights[1]);
}

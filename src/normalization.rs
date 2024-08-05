extern crate ndarray;

use ndarray::{Array1, Array2};

fn scalar_subtraction2(array: &mut Array2<f64>, col_index: usize, value: f64) {
    for i in 0..array.nrows() {
        array[[i, col_index]] -= value;
    }
}

fn scalar_division2(array: &mut Array2<f64>, col_index: usize, value: f64) {
    for i in 0..array.nrows() {
        array[[i, col_index]] /= value;
    }
}

pub fn std_scaler2(array: &Array2<f64>, mean: Array1<f64>, std: Array1<f64>) -> Array2<f64> {
    // Apply standard scaling along columns
    let mut scaled = array.clone();
    for j in 0..scaled.ncols() {
        scalar_subtraction2(&mut scaled, j, mean[j]);
        scalar_division2(&mut scaled, j, std[j]);
    }
    scaled    
}

#[cfg(test)]
mod tests {

    use super::*;

    use ndarray::{array, Axis};
    
    fn mean_std2(array: &Array2<f64>, axis: Axis) -> (Array1<f64>, Array1<f64>) {
        let mean = array.mean_axis(axis.clone()).unwrap();
        let std = array.std_axis(axis, 1.);
        (mean, std)
    }
    
    fn expand_to(n: f64, decimal_places: u8) -> f64 {
        let f = 10.0f64.powi(decimal_places as i32);
        (n * f).trunc()
    }

    #[test]
    fn std_scaler2_test() {
        let a1 = array![[4.5, 3.2, 5.6], [6.7, -45.1, 4.89], [1.98, 9.10, 10.22]];
        let (a1_mean, a1_std) = mean_std2(&a1, Axis(0));
        let a1_sum = a1.sum_axis(Axis(0));

        println!("a1 \n{a1}");
        println!("a1_mean \n{a1_mean}\n a1_std \n{a1_std}\n a1_sum \n{a1_sum}");

        assert_eq!(
            std_scaler2(&a1, a1_mean, a1_std).map(|x| expand_to(*x, 4)),
            array![
                [0.0451, 0.4752, -0.4503],
                [0.9766, -1.1490, -0.6956],
                [-1.0218, 0.6737, 1.1459]
            ].map(|x| expand_to(*x, 4))
        );
    }
}

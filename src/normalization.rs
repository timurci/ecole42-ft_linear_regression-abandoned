extern crate ndarray;

use ndarray::{Array1, Array2, Axis};

pub struct StdParams<T> {
    mean: Array1<T>,
    std: Array1<T>,
}

impl<T> StdParams<T> {
    pub fn mean<'a>(&'a self) -> &'a Array1<T> {
        &self.mean
    }

    pub fn std<'a>(&'a self) -> &'a Array1<T> {
        &self.std
    }
}

macro_rules! declare_scalar_row_ops {
    ($fn_name:ident, $op:tt) => {
        fn $fn_name(array: &mut Array2<f64>, col_index: usize, value: f64) {
            for i in 0..array.nrows() {
                array[[i, col_index]] $op value;
            }
        }
    }
}

declare_scalar_row_ops!(scalar_add2, +=);
declare_scalar_row_ops!(scalar_sub2, -=);
declare_scalar_row_ops!(scalar_mul2, *=);
declare_scalar_row_ops!(scalar_div2, /=);

pub fn std_scaler2(array: &Array2<f64>, params: &StdParams<f64>) -> Array2<f64> {
    // Apply standard scaling along columns
    let mut scaled = array.clone();
    for j in 0..scaled.ncols() {
        scalar_sub2(&mut scaled, j, params.mean[j]);
        scalar_div2(&mut scaled, j, params.std[j]);
    }
    scaled
}

pub fn inv_std_scaler2(array: &Array2<f64>, params: &StdParams<f64>) -> Array2<f64> {
    // Apply standard scaling along columns
    let mut scaled = array.clone();
    for j in 0..scaled.ncols() {
        scalar_mul2(&mut scaled, j, params.std[j]);
        scalar_add2(&mut scaled, j, params.mean[j]);
    }
    scaled
}

pub fn mean_std2(array: &Array2<f64>, axis: Axis) -> StdParams<f64> {
    let mean = array.mean_axis(axis.clone()).unwrap();
    let std = array.std_axis(axis, 1.);

    StdParams { mean, std }
}

#[cfg(test)]
mod tests {

    use super::*;

    use ndarray::array;

    fn expand_to(n: f64, decimal_places: u8) -> f64 {
        let f = 10.0f64.powi(decimal_places as i32);
        (n * f).trunc()
    }

    #[test]
    fn std_scaler2_test() {
        let a1 = array![[4.5, 3.2, 5.6], [6.7, -45.1, 4.89], [1.98, 9.10, 10.22]];
        let a1_params = mean_std2(&a1, Axis(0));

        assert_eq!(
            std_scaler2(&a1, &a1_params).map(|x| expand_to(*x, 4)),
            array![
                [0.0451, 0.4752, -0.4503],
                [0.9766, -1.1490, -0.6956],
                [-1.0218, 0.6737, 1.1459]
            ]
            .map(|x| expand_to(*x, 4))
        );
    }

    #[test]
    fn inv_std_scaler2_test() {
        let a1 = array![[4.5, 3.2, 5.6], [6.7, -45.1, 4.89], [1.98, 9.10, 10.22]];
        let a1_params = mean_std2(&a1, Axis(0));
        let a2 = std_scaler2(&a1, &a1_params);

        assert_eq!(
            inv_std_scaler2(&a2, &a1_params).map(|x| expand_to(*x, 4)),
            array![[4.5, 3.2, 5.6], [6.7, -45.1, 4.89], [1.98, 9.10, 10.22]]
                .map(|x| expand_to(*x, 4))
        )
    }
}

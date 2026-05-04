//! NVR (noise variance reduction) threshold.
//!
//! Mirrors lines 1022–1031 of `nordic/denoise.py`: average the top singular
//! value of `n_iters` random `randn(prod(kernel), n_vols)` matrices.

use ndarray::{Array2, Zip};
use num_complex::Complex32;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::svd::svd_thin;

pub fn nvr_threshold(
    n_rows: usize,
    n_cols: usize,
    rng: &mut impl Rng,
    n_iters: usize,
) -> f32 {
    let normal = Normal::new(0.0_f32, 1.0).expect("Normal::new(0,1) is always valid");
    let mut sum_top = 0.0_f32;
    for _ in 0..n_iters {
        let mut m = Array2::<Complex32>::zeros((n_rows, n_cols));
        Zip::from(&mut m).for_each(|c| {
            // Python uses real-valued randn for the threshold calculation
            // (`np.linalg.svd(random_matrices, compute_uv=False)` is run on a
            // real array). We mirror by leaving the imaginary part at zero.
            *c = Complex32::new(normal.sample(rng), 0.0);
        });
        let svd = svd_thin(m.view());
        if let Some(&top) = svd.s.first() {
            sum_top += top;
        }
    }
    sum_top / n_iters as f32
}

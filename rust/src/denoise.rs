//! NORDIC / MP-PCA denoising pass.
//!
//! Mirrors `denoise_data` in `nordic/denoise.py`.

use ndarray::{Array3, Array4};
use num_complex::Complex32;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

use crate::config::Algorithm;
use crate::patch::kept_x_patch_indices;
use crate::subpatch::{process_x_patch, Selector};
use crate::threshold::nvr_threshold;

pub struct DenoiseOutput {
    pub denoised: Array4<Complex32>,
    pub total_weights: Array3<u32>,
    /// Sum-over-patches of the per-patch `sigmasq_2`. Only populated for the
    /// MP-PCA selector; NORDIC's hard-threshold selector leaves this at zero.
    pub noise: Array3<f32>,
    pub component_threshold: Array3<f32>,
    pub energy_removed: Array3<f32>,
    pub snr_weight: Array3<f32>,
}

pub fn auto_kernel_size(n_vols: usize, n_slices: usize) -> [usize; 3] {
    // Matches Python: `int(round(cbrt(n_vols * 11)))` repeated three ways.
    let cubic = ((n_vols as f64) * 11.0).cbrt().round() as usize;
    let mut k = [cubic, cubic, cubic];
    if n_slices <= k[2] {
        // Number of slices is less than cubic kernel — fall back to a square
        // 2D kernel `sqrt(n_vols * 11 / n_slices)` and clamp z to n_slices.
        let sq = ((n_vols as f64 * 11.0) / (n_slices as f64))
            .sqrt()
            .round() as usize;
        k = [sq, sq, n_slices];
    }
    k
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_data(
    data: &Array4<Complex32>,
    kernel_size: Option<[usize; 3]>,
    patch_overlap: usize,
    measured_noise: f32,
    factor_error: f32,
    has_complex: bool,
    algorithm: Algorithm,
    soft_thrs: Option<f32>,
    llr_scale: f32,
    scale_patches: bool,
    seed: Option<u64>,
) -> DenoiseOutput {
    let (n_x, n_y, n_z, n_vols) = data.dim();
    let auto = auto_kernel_size(n_vols, n_z);
    let kernel_size = kernel_size.unwrap_or(auto);
    let kx = kernel_size[0];

    // Build NVR threshold from the mean top singular value of `n_iters`
    // random Gaussian matrices.
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xC0FFEE));
    let n_iters = 10;
    let mut nvr = nvr_threshold(kx * kernel_size[1] * kernel_size[2], n_vols, &mut rng, n_iters);
    nvr *= measured_noise * factor_error;
    if has_complex {
        nvr *= std::f32::consts::SQRT_2;
    }
    tracing::debug!(nvr_threshold = nvr, "NVR threshold built for {:?}", algorithm);

    // Build the per-x-patch lambda threshold and pick the singular-value
    // selector. Python's `soft_thrs` is `None` for NORDIC (hard threshold) and
    // `10` for MPPCA / gfactor+mppca (MP-PCA cutoff).
    let lambda_thresh = llr_scale * nvr;
    let selector = match soft_thrs {
        None => Selector::NordicHard { lambda_thresh },
        Some(v) if (v - 10.0).abs() < f32::EPSILON => Selector::Mppca,
        Some(_) => {
            // The "other-value" branch raises NotImplementedError in Python.
            // We do the same here.
            panic!("soft_thrs values other than None or 10 are unimplemented");
        }
    };

    let kept = kept_x_patch_indices(n_x, kx, patch_overlap);
    let mut updates: Vec<_> = kept
        .par_iter()
        .map(|&n1| {
            process_x_patch(
                data.view(),
                n1,
                kernel_size,
                selector,
                patch_overlap,
                scale_patches,
            )
        })
        .collect();
    updates.sort_by_key(|u| u.x_start);

    let mut denoised = Array4::<Complex32>::zeros((n_x, n_y, n_z, n_vols));
    let mut weights = Array3::<u32>::zeros((n_x, n_y, n_z));
    let mut noise_acc = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut component_threshold = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut energy_removed = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut snr_weight = Array3::<f32>::zeros((n_x, n_y, n_z));

    for u in updates {
        let kx_u = u.weights.shape()[0];
        for ix in 0..kx_u {
            let gx = u.x_start + ix;
            for iy in 0..n_y {
                for iz in 0..n_z {
                    weights[(gx, iy, iz)] += u.weights[(ix, iy, iz)];
                    noise_acc[(gx, iy, iz)] += u.noise[(ix, iy, iz)];
                    component_threshold[(gx, iy, iz)] += u.component_threshold[(ix, iy, iz)];
                    energy_removed[(gx, iy, iz)] += u.energy_removed[(ix, iy, iz)];
                    snr_weight[(gx, iy, iz)] += u.snr_weight[(ix, iy, iz)];
                    for it in 0..n_vols {
                        denoised[(gx, iy, iz, it)] += u.denoised[(ix, iy, iz, it)];
                    }
                }
            }
        }
    }

    // Masked divide so the uncovered high-x edge stays at exact 0 (matches
    // Python's `np.divide(..., where=cov4d, out=zeros)`). Note: the diagnostic
    // accumulators below are NOT divided here — Python divides them only inside
    // the `if debug:` block, and the values written to disk for the diagnostic
    // maps are the means. We do the divide here as well so the `noise`,
    // `component_threshold`, etc. fields of `DenoiseOutput` are means, ready to
    // serialise straight to NIfTI without any further per-voxel work.
    for ix in 0..n_x {
        for iy in 0..n_y {
            for iz in 0..n_z {
                let w = weights[(ix, iy, iz)];
                if w == 0 {
                    continue;
                }
                let inv = 1.0 / w as f32;
                for it in 0..n_vols {
                    denoised[(ix, iy, iz, it)] = denoised[(ix, iy, iz, it)] * inv;
                }
                noise_acc[(ix, iy, iz)] *= inv;
                component_threshold[(ix, iy, iz)] *= inv;
                energy_removed[(ix, iy, iz)] *= inv;
                snr_weight[(ix, iy, iz)] *= inv;
            }
        }
    }

    DenoiseOutput {
        denoised,
        total_weights: weights,
        noise: noise_acc,
        component_threshold,
        energy_removed,
        snr_weight,
    }
}

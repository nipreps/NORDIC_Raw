//! G-factor estimation pass.
//!
//! Mirrors `estimate_gfactor` in `nordic/denoise.py`: patch-wise SVD with the
//! MP-PCA selector (`soft_thrs=10`, `llr_scale=0`, no NVR threshold). The
//! per-voxel "noise" accumulator is the per-patch `sigmasq_2` value; the
//! returned g-factor is `sqrt(noise / weights)`.

use ndarray::{Array3, Array4};
use num_complex::Complex32;
use rayon::prelude::*;

use crate::patch::kept_x_patch_indices;
use crate::subpatch::{process_x_patch, Selector};

/// Outputs of the g-factor estimation pass.
///
/// `gfactor` is the only field needed for the main pipeline; the others are
/// summed-over-patch diagnostics that mirror the per-stage `*.nii.gz` files
/// Python writes when `debug=True`.
pub struct GfactorOutput {
    pub gfactor: Array3<f32>,
    pub total_weights: Array3<u32>,
    pub component_threshold: Array3<f32>,
    pub energy_removed: Array3<f32>,
    pub snr_weight: Array3<f32>,
}

/// Default kernel size: `[14, 14, 1]` (matches Python).
pub fn default_kernel_size() -> [usize; 3] {
    [14, 14, 1]
}

pub fn estimate_gfactor(
    data: &Array4<Complex32>,
    kernel_size: Option<[usize; 3]>,
    patch_overlap: usize,
) -> GfactorOutput {
    let kernel_size = kernel_size.unwrap_or_else(default_kernel_size);
    let (n_x, n_y, n_z, _n_vols) = data.dim();

    let kept = kept_x_patch_indices(n_x, kernel_size[0], patch_overlap);

    // Each x-patch produces a fully independent XPatchUpdate with no shared
    // mutable state. rayon's `into_par_iter().map(...).collect()` is the
    // canonical lock-free pattern: workers materialise their results in
    // parallel, then we reduce sequentially in stable x-start order so the
    // accumulator sums are deterministic regardless of thread count (matches
    // the Python `_run_x_patches_parallel` ordering trick).
    let mut updates: Vec<_> = kept
        .par_iter()
        .map(|&n1| {
            process_x_patch(
                data.view(),
                n1,
                kernel_size,
                Selector::Mppca,
                patch_overlap,
                false,
            )
        })
        .collect();
    updates.sort_by_key(|u| u.x_start);

    let mut total_weights = Array3::<u32>::zeros((n_x, n_y, n_z));
    let mut noise_acc = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut component_threshold = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut energy_removed = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut snr_weight = Array3::<f32>::zeros((n_x, n_y, n_z));

    for u in updates {
        let kx = u.weights.shape()[0];
        for ix in 0..kx {
            let gx = u.x_start + ix;
            for iy in 0..n_y {
                for iz in 0..n_z {
                    total_weights[(gx, iy, iz)] += u.weights[(ix, iy, iz)];
                    noise_acc[(gx, iy, iz)] += u.noise[(ix, iy, iz)];
                    component_threshold[(gx, iy, iz)] += u.component_threshold[(ix, iy, iz)];
                    energy_removed[(gx, iy, iz)] += u.energy_removed[(ix, iy, iz)];
                    snr_weight[(gx, iy, iz)] += u.snr_weight[(ix, iy, iz)];
                }
            }
        }
    }

    // Masked divide for the per-voxel mean: voxels not covered by any kept
    // x-patch (the high-x edge) stay at exactly 0, matching Python's
    // `np.divide(..., where=cov3d, out=zeros)`.
    let mut gfactor = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut ct_mean = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut er_mean = Array3::<f32>::zeros((n_x, n_y, n_z));
    let mut sw_mean = Array3::<f32>::zeros((n_x, n_y, n_z));
    for ix in 0..n_x {
        for iy in 0..n_y {
            for iz in 0..n_z {
                let w = total_weights[(ix, iy, iz)];
                if w == 0 {
                    continue;
                }
                let inv = 1.0 / w as f32;
                // Clamp tiny negatives so sqrt does not produce NaN — Python
                // does `np.maximum(gfactor_var.real, 0.0)` for the same reason.
                gfactor[(ix, iy, iz)] = (noise_acc[(ix, iy, iz)] * inv).max(0.0).sqrt();
                ct_mean[(ix, iy, iz)] = component_threshold[(ix, iy, iz)] * inv;
                er_mean[(ix, iy, iz)] = energy_removed[(ix, iy, iz)] * inv;
                sw_mean[(ix, iy, iz)] = snr_weight[(ix, iy, iz)] * inv;
            }
        }
    }

    GfactorOutput {
        gfactor,
        total_weights,
        component_threshold: ct_mean,
        energy_removed: er_mean,
        snr_weight: sw_mean,
    }
}

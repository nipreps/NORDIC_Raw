//! Inner Y/Z sub-patch loop.
//!
//! Mirrors `subfunction_loop_for_nvr_avg_update` in `nordic/denoise.py`. Runs
//! every (y, z) sub-patch inside one x-patch, builds the Casorati matrix,
//! does an SVD, applies either the NORDIC hard threshold or the MP-PCA cutoff
//! to the singular values, reconstructs, and additively merges into the
//! per-x-patch accumulators.
//!
//! This function is *pure* — it only reads its arguments and writes into the
//! accumulators it allocates locally — so the calling code can run many
//! x-patches concurrently with rayon.

use ndarray::{s, Array2, Array3, Array4, ArrayView4};
use num_complex::Complex32;

use crate::mppca::mp_select;
use crate::patch::{yz_patch_starts, XPatchUpdate};
use crate::svd::{reconstruct_into, svd_thin};

/// What the singular values get put through.
#[derive(Debug, Clone, Copy)]
pub enum Selector {
    /// NORDIC: zero every `s[i] < lambda_thresh`. The Python `soft_thrs is None`
    /// branch.
    NordicHard { lambda_thresh: f32 },
    /// MP-PCA cutoff (Python's `soft_thrs == 10`). Used for both g-factor
    /// estimation and the MPPCA / gfactor+mppca pipelines' denoising step.
    Mppca,
}

#[allow(clippy::too_many_arguments)]
pub fn process_x_patch(
    data: ArrayView4<Complex32>,
    x_start: usize,
    kernel_size: [usize; 3],
    selector: Selector,
    patch_average_sub: usize,
    scale_patches: bool,
) -> XPatchUpdate {
    let kx = kernel_size[0];
    let ky = kernel_size[1];
    let kz = kernel_size[2];
    let (_, n_y, n_z, n_vols) = data.dim();

    // Slice out the x-patch we own. This is a view (no copy).
    let x_patch = data.slice(s![x_start..x_start + kx, .., .., ..]);

    let mut denoised = Array4::<Complex32>::zeros((kx, n_y, n_z, n_vols));
    let mut weights = Array3::<u32>::zeros((kx, n_y, n_z));
    let mut noise = Array3::<f32>::zeros((kx, n_y, n_z));
    let mut component_threshold = Array3::<f32>::zeros((kx, n_y, n_z));
    let mut energy_removed = Array3::<f32>::zeros((kx, n_y, n_z));
    let mut snr_weight = Array3::<f32>::zeros((kx, n_y, n_z));

    // Y/Z patch starts, with the deliberate duplicate at the end (MATLAB-bug
    // preservation). `dedupe=false` keeps Python parity.
    let y_starts = yz_patch_starts(n_y, ky, patch_average_sub, false);
    let z_starts = yz_patch_starts(n_z, kz, patch_average_sub, false);

    let n_rows_full = kx * ky * kz;

    // Reusable scratch buffers, sized once.
    let mut casorati = Array2::<Complex32>::zeros((n_rows_full, n_vols));
    let mut reconstructed = Array2::<Complex32>::zeros((n_rows_full, n_vols));

    for &y in &y_starts {
        for &z in &z_starts {
            // Pack the (kx, ky, kz, n_vols) sub-patch into a (kx*ky*kz, n_vols)
            // Casorati matrix. NumPy's reshape and our manual loop iterate in
            // the same row-major order, so the matrix matches Python's
            // np.reshape exactly.
            let sub = x_patch.slice(s![.., y..y + ky, z..z + kz, ..]);
            let mut row = 0usize;
            for ix in 0..kx {
                for iy in 0..ky {
                    for iz in 0..kz {
                        for it in 0..n_vols {
                            casorati[(row, it)] = sub[(ix, iy, iz, it)];
                        }
                        row += 1;
                    }
                }
            }

            // SVD.
            let svd = svd_thin(casorati.view());
            let s_orig = svd.s.clone();
            let mut s_kept = s_orig.clone();
            let n_total = s_kept.len();

            // Apply the selected cutoff.
            let (n_removed, energy_scrub, sigmasq_2_opt, first_removed) = match selector {
                Selector::NordicHard { lambda_thresh } => {
                    let n_below = s_kept.iter().filter(|&&v| v < lambda_thresh).count();
                    let sum_below: f32 = s_kept
                        .iter()
                        .filter(|&&v| v < lambda_thresh)
                        .map(|&v| v)
                        .sum();
                    let sum_all: f32 = s_kept.iter().sum();
                    let energy = if sum_all > 0.0 {
                        sum_below.sqrt() / sum_all.sqrt()
                    } else {
                        0.0
                    };
                    for v in s_kept.iter_mut() {
                        if *v < lambda_thresh {
                            *v = 0.0;
                        }
                    }
                    let first_removed = n_total - n_below;
                    (n_below, energy, None, first_removed)
                }
                Selector::Mppca => {
                    // Count the number of all-zero rows in the Casorati matrix.
                    let mut n_zero = 0usize;
                    for r in 0..n_rows_full {
                        let row_sum: Complex32 =
                            (0..n_vols).map(|c| casorati[(r, c)]).sum();
                        if row_sum.norm_sqr() == 0.0 {
                            n_zero += 1;
                        }
                    }
                    let n_nonzero = n_rows_full - n_zero;
                    if n_nonzero == 0 {
                        // All-zero patch: nothing to keep, no noise contribution.
                        for v in s_kept.iter_mut() {
                            *v = 0.0;
                        }
                        (n_total, 0.0, None, 0usize)
                    } else if let Some(mp) = mp_select(&s_orig, n_nonzero, n_vols) {
                        let n_below = n_total - mp.first_removed;
                        let kept_sum: f32 =
                            s_orig.iter().take(mp.first_removed).sum();
                        let removed_sum: f32 = s_orig.iter().skip(mp.first_removed).sum();
                        let total = kept_sum + removed_sum;
                        let energy = if total > 0.0 {
                            removed_sum.sqrt() / total.sqrt()
                        } else {
                            0.0
                        };
                        for (i, v) in s_kept.iter_mut().enumerate() {
                            if i >= mp.first_removed {
                                *v = 0.0;
                            }
                        }
                        (n_below, energy, Some(mp.sigmasq_2), mp.first_removed)
                    } else {
                        for v in s_kept.iter_mut() {
                            *v = 0.0;
                        }
                        (n_total, 0.0, None, 0usize)
                    }
                }
            };

            // Reconstruct the denoised Casorati matrix.
            reconstruct_into(&svd, &s_kept, &mut reconstructed);

            let patch_scale = if scale_patches {
                // Python: patch_scale = S.shape[0] - n_removed_components.
                // The else branch raises NotImplementedError, but if we ever
                // turn this on we want the right value.
                (n_total - n_removed) as f32
            } else {
                1.0
            };

            // SNR ratio of the largest kept singular value to the singular
            // value just below the cutoff. The Python special-cases all-zero
            // patches; we do the same.
            let denom_idx = first_removed.saturating_sub(2);
            let snr_ratio = if s_orig.get(denom_idx).copied().unwrap_or(0.0) > 0.0 {
                s_orig[0] / s_orig[denom_idx]
            } else {
                0.0
            };

            // Scatter the reconstructed (kx*ky*kz, n_vols) matrix back into the
            // (kx, ky, kz, n_vols) sub-patch position in `denoised`.
            let mut row = 0usize;
            for ix in 0..kx {
                for iy in 0..ky {
                    for iz in 0..kz {
                        let dy = y + iy;
                        let dz = z + iz;
                        for it in 0..n_vols {
                            denoised[(ix, dy, dz, it)] +=
                                reconstructed[(row, it)] * patch_scale;
                        }
                        weights[(ix, dy, dz)] += patch_scale as u32;
                        component_threshold[(ix, dy, dz)] += n_removed as f32;
                        energy_removed[(ix, dy, dz)] += energy_scrub;
                        snr_weight[(ix, dy, dz)] += snr_ratio;
                        if let Some(sig2) = sigmasq_2_opt {
                            noise[(ix, dy, dz)] += sig2;
                        }
                        row += 1;
                    }
                }
            }
        }
    }

    XPatchUpdate {
        x_start,
        denoised,
        weights,
        noise,
        component_threshold,
        energy_removed,
        snr_weight,
    }
}

//! Patch indexing.
//!
//! Encodes the "patch_statuses" trick from `nordic/denoise.py` (lines 815–818,
//! 1043–1047) as an explicit `Vec<usize>` of x-patch start indices. Also
//! produces the deliberate-duplicate-trailing-patch y/z lists from the
//! MATLAB-bug-preservation comment in `subfunction_loop_for_nvr_avg_update`.

use ndarray::Array4;
use num_complex::Complex32;

/// Per-x-patch additive contribution to the global accumulators.
///
/// Each field's leading axis has length `kernel_size[0]`. The reducer adds
/// these into the corresponding global arrays starting at row `x_start`.
/// Building these per-patch deltas — instead of read-modify-writing a shared
/// global array — is what lets the patch loop run on multiple threads.
#[derive(Debug)]
pub struct XPatchUpdate {
    pub x_start: usize,
    pub denoised: Array4<Complex32>,            // (kx, ny, nz, nvols)
    pub weights: ndarray::Array3<u32>,          // (kx, ny, nz)
    pub noise: ndarray::Array3<f32>,            // (kx, ny, nz)  — variance accumulator
    pub component_threshold: ndarray::Array3<f32>, // (kx, ny, nz)
    pub energy_removed: ndarray::Array3<f32>,   // (kx, ny, nz)
    pub snr_weight: ndarray::Array3<f32>,       // (kx, ny, nz)
}

/// Returns the list of x-patch start indices that get processed.
///
/// The Python pattern is:
///     val = max(1, kernel_size[0] // patch_overlap)
///     patch_statuses[1::val] = 2          # mark for skip
///     patch_statuses[-1] = 0              # but keep the very last
/// then process every index whose status != 2.
///
/// Rephrased: keep `0`, then keep every `val`-stride index after that, but
/// also force-keep the final index `n_x_patches - 1`.
pub fn kept_x_patch_indices(n_x: usize, kx: usize, patch_overlap: usize) -> Vec<usize> {
    if n_x <= kx {
        return vec![];
    }
    let n_x_patches = n_x - kx;
    if n_x_patches == 0 {
        return vec![];
    }
    let val = (kx / patch_overlap.max(1)).max(1);

    // Build patch_statuses faithfully to the Python.
    let mut status = vec![0u8; n_x_patches];
    // `for nw1 in range(1, val): patch_statuses[nw1::val] = 2`
    for nw1 in 1..val {
        let mut i = nw1;
        while i < n_x_patches {
            status[i] = 2;
            i += val;
        }
    }
    // `patch_statuses[-1] = 0` — force the final index to be kept.
    if let Some(last) = status.last_mut() {
        *last = 0;
    }

    (0..n_x_patches).filter(|&i| status[i] != 2).collect()
}

/// Y/Z patch starts inside one x-patch, with the deliberate duplicate at the end.
///
/// Python (`subfunction_loop_for_nvr_avg_update`):
///     spacing = max(1, kernel_size_y // patch_average_sub)
///     last = ny - kernel_size_y + 1
///     y_patches = list(range(0, last, spacing))
///     y_patches.append(ny - kernel_size_y)   # MATLAB-bug preservation
///
/// The `dedupe` flag is an opt-in fix: future callers who don't want the
/// duplicate run can pass `true`. The default (`false`) matches Python.
pub fn yz_patch_starts(
    full_len: usize,
    kernel_size: usize,
    patch_average_sub: usize,
    dedupe: bool,
) -> Vec<usize> {
    let spacing = (kernel_size / patch_average_sub.max(1)).max(1);
    let last = full_len.saturating_sub(kernel_size).saturating_add(1);

    let mut starts: Vec<usize> = (0..last).step_by(spacing).collect();
    let trailing = full_len.saturating_sub(kernel_size);
    starts.push(trailing);
    if dedupe {
        starts.sort_unstable();
        starts.dedup();
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_x_patches_matches_python_basic() {
        // n_x_patches = 10, val = 3.
        // status start: [0,0,0,0,0,0,0,0,0,0]
        // nw1=1: indices 1,4,7  -> 2
        // nw1=2: indices 2,5,8  -> 2
        // patch_statuses[-1]=0 (already 0 since 9 not touched)
        // kept: 0,3,6,9
        let kept = kept_x_patch_indices(10 + 14, 14, 14 / 3);
        // kx / patch_overlap = 14/(14/3)=14/4=3 (since patch_overlap=14/3=4)
        // Easier: just call directly with kx, val computed from inputs.
        let kept2 = kept_x_patch_indices_with_val(10, 3);
        assert_eq!(kept, kept2);
    }

    fn kept_x_patch_indices_with_val(n_patches: usize, val: usize) -> Vec<usize> {
        let mut status = vec![0u8; n_patches];
        for nw1 in 1..val {
            let mut i = nw1;
            while i < n_patches {
                status[i] = 2;
                i += val;
            }
        }
        if let Some(last) = status.last_mut() {
            *last = 0;
        }
        (0..n_patches).filter(|&i| status[i] != 2).collect()
    }

    #[test]
    fn yz_patch_duplicate_trailing_default() {
        // full_len=10, kernel=4, sub=2 -> spacing=2, last=7, starts=[0,2,4,6]
        // append duplicate trailing 6.
        let s = yz_patch_starts(10, 4, 2, false);
        assert_eq!(s, vec![0, 2, 4, 6, 6]);
    }

    #[test]
    fn yz_patch_dedupe_opt_in() {
        let s = yz_patch_starts(10, 4, 2, true);
        assert_eq!(s, vec![0, 2, 4, 6]);
    }
}

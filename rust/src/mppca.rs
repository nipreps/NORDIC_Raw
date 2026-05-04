//! Marchenko-Pastur singular-value cutoff (Veraart 2016 / NORDIC g-factor).
//!
//! Mirrors the `soft_thrs == 10` branch of
//! `subfunction_loop_for_nvr_avg_update` in `nordic/denoise.py` (≈ lines
//! 1543–1586).

/// Result of one MP-PCA cutoff selection on a Casorati patch.
pub struct MpResult {
    /// Index of the first singular value that gets zeroed (the "noise floor").
    /// Equal to `S.size - n_removed_components` in the Python.
    pub first_removed: usize,
    /// Estimated per-voxel noise variance contributed by this patch (the value
    /// `sigmasq_2[first_removed_component]` in Python).
    pub sigmasq_2: f32,
}

/// Run the MP-PCA selector on a sorted-descending vector of singular values.
///
/// Returns `None` for an entirely zero patch (the Python `else` branch sets
/// `n_removed_components = S.size`, `sigmasq_2 = None`).
///
/// `n_voxels_nonzero` is the count of *non-zero* rows of the Casorati matrix
/// (Python: `n_nonzero_voxels_in_patch`). `n_volumes` is the matrix's column
/// count.
pub fn mp_select(s: &[f32], n_voxels_nonzero: usize, n_volumes: usize) -> Option<MpResult> {
    if n_voxels_nonzero == 0 || s.is_empty() || n_volumes == 0 {
        return None;
    }
    let r = n_voxels_nonzero.min(n_volumes);
    if r == 0 {
        return None;
    }

    let nv = n_volumes as f32;
    let mp_vox = n_voxels_nonzero as f32;

    // vals[i] = s[i]^2 / n_volumes
    let vals: Vec<f32> = s.iter().take(r).map(|v| (v * v) / nv).collect();

    // First sigma^2 estimate (Eq. 1 from the ISMRM presentation).
    // csum[k] = sum_{j=R-1-k..R-1} vals[j]   (cumulative sum from the tail)
    let mut csum_rev = vec![0.0_f32; r];
    let mut acc = 0.0_f32;
    for i in 0..r {
        acc += vals[r - 1 - i];
        csum_rev[i] = acc;
    }
    // cmean[i] = csum_rev[r-1-i] / (r - i)  -- Python: csum[::-1] / arange(1,R+1)[::-1]
    let mut cmean = vec![0.0_f32; r];
    for i in 0..r {
        cmean[i] = csum_rev[r - 1 - i] / ((r - i) as f32);
    }
    let scaling: Vec<f32> = (0..r).map(|i| (mp_vox.max(nv) - i as f32) / nv).collect();
    let sigmasq_1: Vec<f32> = (0..r).map(|i| cmean[i] / scaling[i]).collect();

    // Second sigma^2 estimate (Eq. 2).
    // gamma[i] = (n_voxels_nonzero - i) / n_volumes
    // rangeMP[i] = 4 * sqrt(gamma[i])
    // rangeData[i] = vals[i] - vals[R-1]
    let last_val = vals[r - 1];
    let sigmasq_2: Vec<f32> = (0..r)
        .map(|i| {
            let gamma = (mp_vox - i as f32) / nv;
            let range_mp = 4.0 * gamma.sqrt();
            let range_data = vals[i] - last_val;
            if range_mp == 0.0 {
                f32::INFINITY
            } else {
                range_data / range_mp
            }
        })
        .collect();

    // first_removed = first index where sigmasq_2 < sigmasq_1.
    let first_removed = match (0..r).find(|&i| sigmasq_2[i] < sigmasq_1[i]) {
        Some(idx) => idx,
        None => r, // no cutoff found — keep everything
    };
    let sigmasq_2_at_cut = if first_removed < r {
        sigmasq_2[first_removed]
    } else {
        // No cut found. Python returns sigmasq_2[first_removed] which would
        // index out of range; in practice the Python comparison always finds
        // *some* index because sigmasq_2 < sigmasq_1 at the noise floor. Fall
        // back to 0 contribution to stay numerically safe.
        0.0
    };
    Some(MpResult {
        first_removed,
        sigmasq_2: sigmasq_2_at_cut,
    })
}

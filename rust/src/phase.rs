//! Phase filtering: 2D FFT + separable Tukey low-pass + 2D IFFT.
//!
//! Python: `filter_phase` in `nordic/denoise.py`.

use ndarray::{Array4, Zip};
use num_complex::Complex32;

use crate::config::TemporalPhase;
use crate::fft2d::{fft2_with_shift, ifft2_with_shift};
use crate::tukey::tukey;

/// Low-pass-filter the phase of a 4D complex volume.
///
/// `temporal_phase`:
/// - `Off`: returns an all-zero array (matches Python's `np.zeros_like(data)`
///   branch).
/// - `One`/`Two`/`Three`: 2D FFT (axes 0,1) → separable Tukey window raised to
///   `phase_filter_width` along x and y → 2D IFFT.
/// - `Two`: additionally replaces the filtered output with the raw data
///   wherever the residual phase deviates by more than 1 radian. The stricter
///   `Three` mask is applied later in the runner (after g-factor demodulation).
pub fn filter_phase(
    data: &Array4<Complex32>,
    phase_filter_width: u32,
    temporal_phase: TemporalPhase,
) -> Array4<Complex32> {
    if matches!(temporal_phase, TemporalPhase::Off) {
        return Array4::<Complex32>::zeros(data.raw_dim());
    }

    let (n_x, n_y, _n_z, _n_t) = data.dim();
    let mut filtered = data.clone();

    // 2D inverse-with-shift along axes 0 and 1.
    ifft2_with_shift(&mut filtered);

    // Separable Tukey window powered to phase_filter_width.
    let tukey_x = tukey(n_x, 1.0).mapv(|v| v.powi(phase_filter_width as i32));
    let tukey_y = tukey(n_y, 1.0).mapv(|v| v.powi(phase_filter_width as i32));

    // Multiply lane-wise. Zip lets us broadcast the 1D windows over (z, t).
    for i in 0..n_x {
        let wx = tukey_x[i];
        let mut slab = filtered.slice_mut(ndarray::s![i, .., .., ..]);
        for j in 0..n_y {
            let w = wx * tukey_y[j];
            slab.slice_mut(ndarray::s![j, .., ..]).mapv_inplace(|c| c * w);
        }
    }

    // Forward 2D FFT with shifts.
    fft2_with_shift(&mut filtered);

    if matches!(temporal_phase, TemporalPhase::Two) {
        apply_spike_mask_simple(&mut filtered, data);
    }

    filtered
}

/// `temporal_phase==2` mask: replace `filtered` with `data` wherever
/// `|angle(data / filtered)| > 1`.
fn apply_spike_mask_simple(filtered: &mut Array4<Complex32>, data: &Array4<Complex32>) {
    Zip::from(filtered).and(data).for_each(|f, d| {
        // angle(d/f) = angle(d) - angle(f). Robust against |f| ~ 0 because we
        // only look at the angle.
        let ratio = if f.norm_sqr() > 0.0 {
            *d / *f
        } else {
            Complex32::new(0.0, 0.0)
        };
        if ratio.arg().abs() > 1.0 {
            *f = *d;
        }
    });
}

/// `temporal_phase==3` mask, applied after g-factor normalisation. Replaces
/// `filtered` with `demod` wherever the residual phase exceeds 1 radian AND
/// the demodulated magnitude exceeds √2.
pub fn apply_spike_mask_strict(filtered: &mut Array4<Complex32>, demod: &Array4<Complex32>) {
    let sqrt2 = std::f32::consts::SQRT_2;
    Zip::from(filtered).and(demod).for_each(|f, d| {
        if d.norm() <= sqrt2 {
            return;
        }
        let ratio = if f.norm_sqr() > 0.0 {
            *d / *f
        } else {
            Complex32::new(0.0, 0.0)
        };
        if ratio.arg().abs() > 1.0 {
            *f = *d;
        }
    });
}

//! Magnitude+phase combination, phase rescaling and absolute-scale logic.
//!
//! Mirrors lines 254–283 of `nordic/denoise.py`.

use ndarray::{Array4, Zip};
use num_complex::Complex32;

use crate::error::NordicError;

/// Bookkeeping needed to reverse the phase rescale at write time.
#[derive(Debug, Clone, Copy)]
pub struct PhaseMeta {
    pub range_norm: f32,
    pub range_center: f32,
}

/// Rescale a 4D phase volume to `[-π, π]`, using the same formula as Python:
/// `(phase / (max-min) - (max+min)/(max-min)/2) * 2π`.
pub fn rescale_phase(pha: &Array4<f32>) -> (Array4<f32>, PhaseMeta) {
    let (mut pmax, mut pmin) = (f32::NEG_INFINITY, f32::INFINITY);
    for &v in pha.iter() {
        if v > pmax {
            pmax = v;
        }
        if v < pmin {
            pmin = v;
        }
    }
    let range_norm = pmax - pmin;
    let range_center = (pmax + pmin) / range_norm * 0.5;
    let two_pi = 2.0 * std::f32::consts::PI;
    let rescaled = pha.mapv(|v| (v / range_norm - range_center) * two_pi);
    (
        rescaled,
        PhaseMeta {
            range_norm,
            range_center,
        },
    )
}

/// `mag * exp(i * pha)` with NaN/Inf scrubbed. Matches `complex_data` in Python.
pub fn build_complex(mag: &Array4<f32>, pha: Option<&Array4<f32>>) -> Array4<Complex32> {
    if let Some(p) = pha {
        let mut out = Array4::<Complex32>::zeros(mag.raw_dim());
        Zip::from(&mut out)
            .and(mag)
            .and(p)
            .for_each(|c, &m, &ph| {
                *c = Complex32::from_polar(m, ph);
            });
        out
    } else {
        mag.mapv(|m| Complex32::new(m, 0.0))
    }
}

/// Find the smallest non-zero magnitude in the first volume and divide the
/// whole 4D array by it (Python: `absolute_scale = min(|II[..., 0]| > 0)`).
pub fn divide_by_absolute_scale(data: &mut Array4<Complex32>) -> Result<f32, NordicError> {
    let first = data.index_axis(ndarray::Axis(3), 0);
    let mut absolute_scale = f32::INFINITY;
    for c in first.iter() {
        let m = c.norm();
        if m > 0.0 && m < absolute_scale {
            absolute_scale = m;
        }
    }
    if !absolute_scale.is_finite() {
        return Err(NordicError::ZeroAbsoluteScale);
    }
    let inv = 1.0 / absolute_scale;
    data.mapv_inplace(|c| c * inv);
    Ok(absolute_scale)
}

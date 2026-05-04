//! SciPy-compatible Tukey (tapered cosine) window.

use ndarray::Array1;

/// Length-`n` Tukey window with shape parameter `alpha`.
///
/// Matches `scipy.signal.windows.tukey(n, alpha, sym=True)`. At `alpha=1.0`
/// this degenerates to a Hann window with both endpoints exactly zero, which
/// is the only value the NORDIC pipeline ever passes (it then raises the
/// window to `phase_filter_width`, so endpoints stay at 0).
pub fn tukey(n: usize, alpha: f32) -> Array1<f32> {
    let mut w = Array1::<f32>::zeros(n);
    if n == 0 {
        return w;
    }
    if n == 1 {
        w[0] = 1.0;
        return w;
    }
    if alpha <= 0.0 {
        // Rectangular: SciPy returns all-ones at alpha=0.
        w.fill(1.0);
        return w;
    }
    let alpha = alpha.min(1.0);

    let m = n as f32;
    let width = (alpha * (m - 1.0) / 2.0).floor() as usize;
    let pi = std::f32::consts::PI;
    let denom = alpha * (m - 1.0);

    // SciPy's three-segment formulation (sym=True).
    for i in 0..n {
        let x = i as f32;
        if i <= width {
            // Rising cosine.
            w[i] = 0.5 * (1.0 + (pi * (-1.0 + 2.0 * x / denom)).cos());
        } else if i < n - 1 - width {
            // Flat top.
            w[i] = 1.0;
        } else {
            // Falling cosine.
            w[i] = 0.5 * (1.0 + (pi * (-2.0 / alpha + 1.0 + 2.0 * x / denom)).cos());
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn tukey_alpha_1_is_hann_with_zero_endpoints() {
        let w = tukey(8, 1.0);
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(w[7], 0.0, epsilon = 1e-6);
        assert!(w[3] > 0.9 && w[4] > 0.9);
    }

    #[test]
    fn tukey_alpha_0_is_rectangular() {
        let w = tukey(5, 0.0);
        for v in w.iter() {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn tukey_n1_is_one() {
        let w = tukey(1, 1.0);
        assert_eq!(w.len(), 1);
        assert_abs_diff_eq!(w[0], 1.0);
    }
}

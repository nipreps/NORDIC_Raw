//! 2D FFT helpers built on top of `rustfft`'s 1D engine.
//!
//! NumPy convention: `fft` is unnormalised, `ifft` divides by N. `rustfft` is
//! unnormalised in both directions, so the inverse helpers here divide by N.

use ndarray::{Array4, ArrayViewMut1, Axis};
use num_complex::Complex32;
use rustfft::FftPlanner;

/// In-place 1D FFT of every line along `axis` of `data`.
fn fft_axis(data: &mut Array4<Complex32>, axis: usize, inverse: bool) {
    let n = data.shape()[axis];
    if n <= 1 {
        return;
    }
    let mut planner = FftPlanner::<f32>::new();
    let plan = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };

    // Walk every other-axis index; for each, take a 1D mutable line along
    // `axis` and FFT it in place. This is the same as np.fft.fft(arr, axis=k).
    let scale = if inverse { 1.0 / n as f32 } else { 1.0 };

    let mut scratch = vec![Complex32::new(0.0, 0.0); plan.get_inplace_scratch_len()];
    let mut lane = vec![Complex32::new(0.0, 0.0); n];

    for mut line in data.lanes_mut(Axis(axis)) {
        // Copy out, transform, copy back. ndarray lanes aren't contiguous in
        // general, so we can't hand the slice directly to rustfft.
        for (dst, src) in lane.iter_mut().zip(line.iter()) {
            *dst = *src;
        }
        plan.process_with_scratch(&mut lane, &mut scratch);
        if inverse {
            for v in lane.iter_mut() {
                *v *= scale;
            }
        }
        for (dst, src) in line.iter_mut().zip(lane.iter()) {
            *dst = *src;
        }
    }
}

pub fn fft_along(data: &mut Array4<Complex32>, axis: usize) {
    fft_axis(data, axis, false);
}

pub fn ifft_along(data: &mut Array4<Complex32>, axis: usize) {
    fft_axis(data, axis, true);
}

/// In-place fftshift along `axis`.
///
/// `np.fft.fftshift(x, axes=k)`: rolls by `n // 2`. For even `n` this is also
/// its own inverse; for odd `n` we need the matching `ifftshift` instead.
pub fn fftshift_along(data: &mut Array4<Complex32>, axis: usize) {
    let n = data.shape()[axis];
    if n < 2 {
        return;
    }
    let half = n / 2; // forward shift = floor(n/2)
    roll_axis(data, axis, half);
}

/// In-place ifftshift along `axis` (`(n+1)/2` roll, undoes fftshift for any n).
pub fn ifftshift_along(data: &mut Array4<Complex32>, axis: usize) {
    let n = data.shape()[axis];
    if n < 2 {
        return;
    }
    let half = (n + 1) / 2; // inverse shift = ceil(n/2)
    roll_axis(data, axis, half);
}

/// Cyclic shift each lane along `axis` by `k` positions to the right.
fn roll_axis(data: &mut Array4<Complex32>, axis: usize, k: usize) {
    let n = data.shape()[axis];
    if n < 2 {
        return;
    }
    let k = k % n;
    if k == 0 {
        return;
    }
    let mut buf = vec![Complex32::new(0.0, 0.0); n];
    for mut line in data.lanes_mut(Axis(axis)) {
        for (i, src) in line.iter().enumerate() {
            buf[(i + k) % n] = *src;
        }
        copy_into_lane(&mut line, &buf);
    }
}

fn copy_into_lane(dst: &mut ArrayViewMut1<Complex32>, src: &[Complex32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s;
    }
}

/// Helper used by `phase::filter_phase`: ifftshift → ifft → ifftshift along
/// each of the first two axes, in order.
pub fn ifft2_with_shift(data: &mut Array4<Complex32>) {
    for axis in 0..2 {
        ifftshift_along(data, axis);
        ifft_along(data, axis);
        ifftshift_along(data, axis);
    }
}

/// Inverse of [`ifft2_with_shift`]: fftshift → fft → fftshift along axes 0,1.
pub fn fft2_with_shift(data: &mut Array4<Complex32>) {
    for axis in 0..2 {
        fftshift_along(data, axis);
        fft_along(data, axis);
        fftshift_along(data, axis);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::s;

    fn array_from_real(re: &[f32], shape: (usize, usize, usize, usize)) -> Array4<Complex32> {
        let data: Vec<Complex32> = re.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        Array4::from_shape_vec(shape, data).unwrap()
    }

    #[test]
    fn fft_then_ifft_roundtrip() {
        let mut a = array_from_real(&[1., 2., 3., 4., 5., 6., 7., 8.], (4, 2, 1, 1));
        let original = a.clone();
        fft_along(&mut a, 0);
        ifft_along(&mut a, 0);
        for (l, r) in a.iter().zip(original.iter()) {
            assert_abs_diff_eq!(l.re, r.re, epsilon = 1e-4);
            assert_abs_diff_eq!(l.im, r.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn fftshift_then_ifftshift_is_identity_even() {
        let mut a = array_from_real(&[1., 2., 3., 4.], (4, 1, 1, 1));
        let original = a.clone();
        fftshift_along(&mut a, 0);
        ifftshift_along(&mut a, 0);
        assert_eq!(
            a.slice(s![.., 0, 0, 0]).iter().collect::<Vec<_>>(),
            original.slice(s![.., 0, 0, 0]).iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn fftshift_then_ifftshift_is_identity_odd() {
        let mut a = array_from_real(&[1., 2., 3., 4., 5.], (5, 1, 1, 1));
        let original = a.clone();
        fftshift_along(&mut a, 0);
        ifftshift_along(&mut a, 0);
        assert_eq!(
            a.slice(s![.., 0, 0, 0]).iter().collect::<Vec<_>>(),
            original.slice(s![.., 0, 0, 0]).iter().collect::<Vec<_>>()
        );
    }
}

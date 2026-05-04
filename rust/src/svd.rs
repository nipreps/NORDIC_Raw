//! Thin wrapper around the dense complex SVD.
//!
//! We take a Casorati patch as `Array2<Complex32>` of shape `(rows, cols)` and
//! return the singular values plus a function to reconstruct the rank-`r`
//! truncation `U[:,:r] * diag(S[:r]) * Vᴴ[:r,:]` without materialising big
//! intermediate matrices.

use faer::Mat;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex32;

/// Outcome of one SVD. Stored in a form that lets us cheaply build the
/// rank-truncated reconstruction.
pub struct SvdResult {
    /// Singular values, descending. Length `min(rows, cols)`.
    pub s: Vec<f32>,
    /// Left singular vectors, `(rows × k)` where `k = min(rows, cols)`.
    pub u: Mat<faer::c32>,
    /// Right singular vectors **conjugate-transposed**, i.e. `Vᴴ` of shape
    /// `(k × cols)`. Storing it this way mirrors NumPy's `V` return from
    /// `numpy.linalg.svd`, which is already `Vᴴ`.
    pub vt: Mat<faer::c32>,
}

/// Compute thin SVD of `m`.
pub fn svd_thin(m: ArrayView2<Complex32>) -> SvdResult {
    let (rows, cols) = m.dim();
    // Copy ndarray -> faer. ndarray is row-major, faer is column-major, so we
    // transpose during the copy. Then we transpose the result back.
    let mut a = Mat::<faer::c32>::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let v = m[(i, j)];
            a[(i, j)] = faer::c32::new(v.re, v.im);
        }
    }

    let svd = a
        .thin_svd()
        .expect("thin SVD failed — patch matrix is non-finite");

    let s_diag = svd.S();
    let s_col = s_diag.column_vector();
    let k = s_col.nrows();
    let mut s = Vec::with_capacity(k);
    for i in 0..k {
        // For Mat<c32>, faer stores singular values as Complex<f32> with imag=0.
        // Take the real part — singular values are non-negative reals.
        s.push(s_col[i].re);
    }
    SvdResult {
        s,
        u: svd.U().to_owned(),
        vt: svd.V().adjoint().to_owned(),
    }
}

/// Reconstruct `A_hat = U * diag(s_kept) * Vᴴ` and write it into `out`.
///
/// Singular values past `keep` are treated as zero. `out` must have the same
/// shape as the original input (`rows × cols`).
pub fn reconstruct_into(svd: &SvdResult, s_kept: &[f32], out: &mut Array2<Complex32>) {
    let (rows, cols) = out.dim();
    let k = svd.s.len();
    // Scale each column of U by s_kept[i], then multiply by Vᴴ.
    let mut us = Mat::<faer::c32>::zeros(rows, k);
    for i in 0..rows {
        for j in 0..k {
            let u = svd.u[(i, j)];
            let scale = s_kept[j];
            us[(i, j)] = faer::c32::new(u.re * scale, u.im * scale);
        }
    }
    let prod = &us * &svd.vt;
    for i in 0..rows {
        for j in 0..cols {
            let c = prod[(i, j)];
            out[(i, j)] = Complex32::new(c.re, c.im);
        }
    }
}

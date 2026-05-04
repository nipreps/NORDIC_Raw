//! NIfTI I/O.
//!
//! Thin wrappers around the `nifti` crate that always materialise into
//! `Array4<f32>` (matching `nibabel.load(...).get_fdata().astype(np.float32)`).

use std::path::{Path, PathBuf};

use ndarray::{Array3, Array4, IxDyn};
use nifti::{writer::WriterOptions, IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};

use crate::error::NordicError;

/// A 4D float volume plus its header (we keep the header so we can copy
/// affine/qform/scl_* fields onto outputs, like `nibabel.Nifti1Image(..., img.header)`).
pub struct NiftiVolume4d {
    pub data: Array4<f32>,
    pub header: NiftiHeader,
}

pub fn load_4d(path: &Path) -> Result<NiftiVolume4d, NordicError> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .map_err(NordicError::from)?;
    let header = obj.header().clone();

    // The `nifti` crate's `into_ndarray::<f32>()` *already* applies
    // `scl_slope` / `scl_inter` (see `convert_and_cast_*` and
    // `nifti_rescale_many_inline` in `nifti-0.17`). Earlier versions of this
    // function re-applied them on top, which silently corrupted any input
    // with non-identity scl — most painfully, real fMRI phase data with
    // negative slope ended up *conjugated*, which inverted the sign of every
    // angle-derived output downstream. Trust the crate and don't re-apply.
    let dyn_arr = obj
        .into_volume()
        .into_ndarray::<f32>()
        .map_err(NordicError::from)?;

    let arr4 = to_4d(dyn_arr)?;
    Ok(NiftiVolume4d {
        data: arr4,
        header,
    })
}

fn to_4d(arr: ndarray::Array<f32, IxDyn>) -> Result<Array4<f32>, NordicError> {
    let dims = arr.ndim();
    let shape = arr.shape().to_vec();
    let arr4 = match dims {
        4 => arr.into_dimensionality::<ndarray::Ix4>().map_err(|_| {
            NordicError::NotFourDimensional { dims }
        })?,
        3 => {
            let arr = arr
                .into_dimensionality::<ndarray::Ix3>()
                .expect("checked dims == 3");
            // Promote (X, Y, Z) -> (X, Y, Z, 1).
            let new_shape = (shape[0], shape[1], shape[2], 1usize);
            Array4::from_shape_vec(new_shape, arr.into_raw_vec_and_offset().0)
                .expect("shape preserved during promotion")
        }
        _ => return Err(NordicError::NotFourDimensional { dims }),
    };
    Ok(arr4)
}

/// Write a 4D float volume out, copying the affine/header from a source.
pub fn save_4d(
    out_path: &Path,
    data: &Array4<f32>,
    template_header: &NiftiHeader,
) -> Result<(), NordicError> {
    let mut header = template_header.clone();
    // NiftiHeader has no `set_data_type` setter — set the underlying fields
    // directly. NiftiType::Float32 = 16, bitpix = 32.
    header.datatype = nifti::NiftiType::Float32 as i16;
    header.bitpix = 32;
    // We've already applied scl_slope/scl_inter on read, so write the raw
    // floats out and disable any further scaling on the next read.
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    // Update dim to reflect (X, Y, Z, T).
    let s = data.shape();
    header.dim = [4, s[0] as u16, s[1] as u16, s[2] as u16, s[3] as u16, 1, 1, 1];

    WriterOptions::new(out_path)
        .reference_header(&header)
        .write_nifti(data)
        .map_err(NordicError::from)?;
    Ok(())
}

/// Write a 3D float volume out, copying the affine/header from a source.
///
/// The g-factor map is 3D — Python writes it via `nb.Nifti1Image(gfactor, ...)`
/// and `nibabel` infers `dim[0]=3` from the array shape; we set that
/// explicitly here.
pub fn save_3d(
    out_path: &Path,
    data: &Array3<f32>,
    template_header: &NiftiHeader,
) -> Result<(), NordicError> {
    let mut header = template_header.clone();
    header.datatype = nifti::NiftiType::Float32 as i16;
    header.bitpix = 32;
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    let s = data.shape();
    header.dim = [3, s[0] as u16, s[1] as u16, s[2] as u16, 1, 1, 1, 1];

    WriterOptions::new(out_path)
        .reference_header(&header)
        .write_nifti(data)
        .map_err(NordicError::from)?;
    Ok(())
}

/// Convenience: form `out_dir / (prefix + name)`.
///
/// `prefix` is concatenated literally — pass `"sub-01_"` to get
/// `<out_dir>/sub-01_magn.nii.gz`. The runner creates `out_dir` up-front, so
/// callers don't need to check it again here.
pub fn out_path(out_dir: &Path, prefix: &str, name: &str) -> PathBuf {
    if prefix.is_empty() {
        out_dir.join(name)
    } else {
        out_dir.join(format!("{prefix}{name}"))
    }
}

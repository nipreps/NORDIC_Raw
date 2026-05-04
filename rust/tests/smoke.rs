//! Tiny end-to-end smoke test. Builds a 3-volume input on the fly, runs each
//! algorithm, and asserts the output NIfTIs exist and have the expected shape.
//!
//! This is intentionally lighter than the parity tests in `parity.rs`: it
//! catches integration regressions (CLI parsing, I/O, basic kernel sizing)
//! without needing the WSL/Python reference machinery.

use std::path::PathBuf;

use approx::assert_abs_diff_eq;
use ndarray::{Array4, Axis};
use nifti::{writer::WriterOptions, NiftiHeader};
use nordic_rs::{run_nordic, Algorithm, NordicConfig, SoftThreshold, TemporalPhase};
use tempfile::TempDir;

fn write_mag(path: &std::path::Path, shape: (usize, usize, usize, usize)) {
    let mut data = Array4::<f32>::zeros(shape);
    // Bumpy phantom + low-amplitude variation per volume so the SVD has signal
    // beyond pure DC.
    for ((i, j, k, t), v) in data.indexed_iter_mut() {
        let r = ((i as f32 - shape.0 as f32 / 2.0).powi(2)
            + (j as f32 - shape.1 as f32 / 2.0).powi(2))
        .sqrt();
        *v = (50.0 - 0.5 * r).max(0.0) + (t as f32) + (k as f32);
    }
    let mut header = NiftiHeader::default();
    header.datatype = nifti::NiftiType::Float32 as i16;
    header.bitpix = 32;
    header.pixdim = [1.0; 8];
    header.dim = [
        4,
        shape.0 as u16,
        shape.1 as u16,
        shape.2 as u16,
        shape.3 as u16,
        1,
        1,
        1,
    ];
    header.scl_slope = 1.0;
    WriterOptions::new(path)
        .reference_header(&header)
        .write_nifti(&data)
        .expect("write smoke fixture");
}

fn smoke_one(algo: Algorithm) {
    let tmp = TempDir::new().expect("tmpdir");
    let mag = tmp.path().join("mag.nii.gz");
    write_mag(&mag, (16, 16, 4, 8));

    let mut cfg = NordicConfig::default();
    cfg.mag_file = mag;
    cfg.out_dir = tmp.path().to_path_buf();
    cfg.algorithm = algo;
    cfg.temporal_phase = TemporalPhase::Off;
    cfg.soft_thrs = SoftThreshold::Auto;
    cfg.kernel_size_pca = Some([4, 4, 2]);
    cfg.kernel_size_gfactor = Some([4, 4, 1, 8]);
    cfg.patch_overlap_gfactor = 2;
    cfg.patch_overlap_pca = 2;
    cfg.save_gfactor_map = false;

    let out = run_nordic(&cfg).expect("run_nordic");
    assert!(out.magnitude_path.exists(), "magn.nii.gz must be written");
    let loaded = nordic_rs::io::load_4d(&out.magnitude_path).expect("re-read magn");
    assert_eq!(loaded.data.shape()[3], 8, "no noRF, expect all volumes back");
    let _ = loaded.data.index_axis(Axis(3), 0);
    let _: PathBuf = out.magnitude_path;
}

#[test]
fn smoke_nordic() {
    smoke_one(Algorithm::Nordic);
}

#[test]
fn smoke_mppca() {
    smoke_one(Algorithm::Mppca);
}

#[test]
fn smoke_gfactor_mppca() {
    smoke_one(Algorithm::GfactorMppca);
}

#[test]
fn save_gfactor_writes_3d_map() {
    let tmp = TempDir::new().expect("tmpdir");
    let mag = tmp.path().join("mag.nii.gz");
    write_mag(&mag, (16, 16, 4, 8));

    let mut cfg = NordicConfig::default();
    cfg.mag_file = mag;
    cfg.out_dir = tmp.path().to_path_buf();
    cfg.algorithm = Algorithm::Nordic;
    cfg.temporal_phase = TemporalPhase::Off;
    cfg.soft_thrs = SoftThreshold::Auto;
    cfg.kernel_size_pca = Some([4, 4, 2]);
    cfg.kernel_size_gfactor = Some([4, 4, 1, 8]);
    cfg.save_gfactor_map = true;

    let out = run_nordic(&cfg).expect("run_nordic");
    let gpath = out.gfactor_path.expect("gfactor path was returned");
    assert!(gpath.exists(), "gfactor.nii.gz must exist on disk");
    // Re-load via load_4d (which auto-promotes 3D to 4D with a singleton T).
    let g = nordic_rs::io::load_4d(&gpath).expect("re-read gfactor");
    assert_eq!(g.data.shape(), &[16, 16, 4, 1]);
}

#[test]
fn auto_creates_out_dir_and_applies_prefix() {
    let tmp = TempDir::new().expect("tmpdir");
    let mag = tmp.path().join("mag.nii.gz");
    write_mag(&mag, (16, 16, 4, 8));

    // Output dir does not exist yet — runner must create it. Use a nested
    // path to check `parents=True` semantics.
    let nested_out = tmp.path().join("nested").join("out");
    assert!(!nested_out.exists());

    let mut cfg = NordicConfig::default();
    cfg.mag_file = mag;
    cfg.out_dir = nested_out.clone();
    cfg.algorithm = Algorithm::Nordic;
    cfg.temporal_phase = TemporalPhase::Off;
    cfg.kernel_size_pca = Some([4, 4, 2]);
    cfg.kernel_size_gfactor = Some([4, 4, 1, 8]);
    cfg.save_gfactor_map = true;
    cfg.prefix = "sub-01_".to_string();

    let out = run_nordic(&cfg).expect("run_nordic");
    assert!(nested_out.exists(), "out_dir was not created");
    assert!(out.magnitude_path.exists());
    assert_eq!(
        out.magnitude_path.file_name().unwrap().to_str().unwrap(),
        "sub-01_magn.nii.gz"
    );
    let g = out.gfactor_path.expect("gfactor returned");
    assert_eq!(
        g.file_name().unwrap().to_str().unwrap(),
        "sub-01_gfactor.nii.gz"
    );
}

#[test]
fn load_4d_does_not_double_apply_scl_slope_inter() {
    // Regression test for the double-scl bug. We need a NIfTI with non-trivial
    // `scl_slope=-0.125`, `scl_inter=4094` (mirroring the real fMRI phase data
    // the parity comparison surfaced this on). Unfortunately the `nifti` crate's
    // `WriterOptions::prepare_header` *hardcodes* scl_slope=1.0 and
    // scl_inter=0.0 regardless of `reference_header(...)` — see
    // `nifti-0.17/src/writer.rs`. So we write through the writer first, then
    // patch the header bytes in place at the standard NIfTI-1 offsets:
    //   scl_slope at byte 112..116 (f32 LE)
    //   scl_inter at byte 116..120
    let tmp = TempDir::new().expect("tmpdir");
    let path = tmp.path().join("scaled.nii"); // no .gz so we can patch in place

    let shape = (4_usize, 4_usize, 2_usize, 1_usize);
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    let raw: Vec<u16> = (0..n as u16).collect();
    let raw_arr = ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[shape.0, shape.1, shape.2, shape.3]),
        raw.clone(),
    )
    .expect("u16 fixture shape");

    let mut header = NiftiHeader::default();
    header.datatype = nifti::NiftiType::Uint16 as i16;
    header.bitpix = 16;
    header.pixdim = [1.0; 8];
    header.dim = [
        4,
        shape.0 as u16,
        shape.1 as u16,
        shape.2 as u16,
        shape.3 as u16,
        1,
        1,
        1,
    ];
    WriterOptions::new(&path)
        .reference_header(&header)
        .write_nifti(&raw_arr)
        .expect("write scaled fixture");

    // Patch scl_slope and scl_inter into the header on disk.
    let mut bytes = std::fs::read(&path).expect("read fixture back");
    bytes[112..116].copy_from_slice(&(-0.125_f32).to_le_bytes());
    bytes[116..120].copy_from_slice(&(4094.0_f32).to_le_bytes());
    std::fs::write(&path, &bytes).expect("write patched fixture");

    let v = nordic_rs::io::load_4d(&path).expect("load_4d");
    let arr = &v.data;
    assert_eq!(arr.shape(), &[shape.0, shape.1, shape.2, shape.3]);
    // The nifti crate applies scl once on load. We must NOT apply it again.
    // Expected loaded value at raw=k: -0.125 * k + 4094.
    let flat: Vec<f32> = arr.iter().copied().collect();
    for k in 0..n {
        let expected = (k as f32) * -0.125 + 4094.0;
        assert_abs_diff_eq!(flat[k], expected, epsilon = 1e-3);
    }
    // Also confirm the header round-trips the patched values (sanity).
    assert_abs_diff_eq!(v.header.scl_slope, -0.125, epsilon = 1e-6);
    assert_abs_diff_eq!(v.header.scl_inter, 4094.0, epsilon = 1e-3);
}

#[test]
fn n_jobs_one_runs_serially() {
    // Pick a 1-thread pool. The algorithm should still complete and produce
    // identical-shape output to the multi-threaded run. We don't assert
    // bit-equality (the par_iter/sort path produces identical sums anyway),
    // just that the code path doesn't deadlock or panic.
    let tmp = TempDir::new().expect("tmpdir");
    let mag = tmp.path().join("mag.nii.gz");
    write_mag(&mag, (16, 16, 4, 8));

    let mut cfg = NordicConfig::default();
    cfg.mag_file = mag;
    cfg.out_dir = tmp.path().to_path_buf();
    cfg.algorithm = Algorithm::Nordic;
    cfg.temporal_phase = TemporalPhase::Off;
    cfg.kernel_size_pca = Some([4, 4, 2]);
    cfg.kernel_size_gfactor = Some([4, 4, 1, 8]);
    cfg.n_jobs = Some(1);

    let out = run_nordic(&cfg).expect("run_nordic with n_jobs=1");
    assert!(out.magnitude_path.exists());
}

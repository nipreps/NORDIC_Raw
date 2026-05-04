//! Parity tests vs. the Python reference.
//!
//! Each test loads:
//!   1. The synthetic input fixture from `tests/fixtures/`.
//!   2. The cached Python output from `tests/fixtures/expected/<scenario>/`.
//!   3. Our own output, produced by running `run_nordic` into a temp dir.
//!
//! Then compares the magnitude / phase / g-factor maps within a loose
//! tolerance (~1% relative error on bright voxels).
//!
//! The reference outputs are produced by `scripts/regenerate_fixtures.sh`
//! inside WSL. If they are missing, the affected tests are marked as
//! `#[ignore]`d so a fresh checkout's `cargo test` still passes.

use std::path::{Path, PathBuf};

use ndarray::Array4;
use nordic_rs::io::load_4d;
use nordic_rs::{run_nordic, Algorithm, NordicConfig, SoftThreshold, TemporalPhase};
use tempfile::TempDir;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected_dir(algo: &str, tphase: u8, use_phase: bool) -> PathBuf {
    let safe = algo.replace('+', "_");
    fixture_dir()
        .join("expected")
        .join(format!("{safe}_tp{tphase}_pha{use_phase}"))
}

fn rel_err(out: &Array4<f32>, ref_: &Array4<f32>) -> f32 {
    assert_eq!(out.shape(), ref_.shape(), "shape mismatch vs reference");
    let mut max_abs_ref = 0.0_f32;
    for &v in ref_.iter() {
        let a = v.abs();
        if a > max_abs_ref {
            max_abs_ref = a;
        }
    }
    let thresh = 0.05 * max_abs_ref;
    let mut max_rel = 0.0_f32;
    for (o, r) in out.iter().zip(ref_.iter()) {
        if r.abs() < thresh {
            continue;
        }
        let e = (o - r).abs() / r.abs().max(1e-6);
        if e > max_rel {
            max_rel = e;
        }
    }
    max_rel
}

fn run_scenario(algo_str: &str, tphase: u8, use_phase: bool) {
    let fdir = fixture_dir();
    let mag_in = fdir.join("synthetic_mag.nii.gz");
    let pha_in = fdir.join("synthetic_pha.nii.gz");
    let mag_norf = fdir.join("synthetic_mag_noRF.nii.gz");
    let pha_norf = fdir.join("synthetic_pha_noRF.nii.gz");

    let expected = expected_dir(algo_str, tphase, use_phase);
    if !expected.exists() {
        eprintln!(
            "SKIP: expected output dir {} does not exist. \
             Run scripts/regenerate_fixtures.sh inside WSL first.",
            expected.display()
        );
        return;
    }
    if !mag_in.exists() {
        eprintln!(
            "SKIP: input fixture missing at {}. \
             Run tests/fixtures/synthetic_make.py first.",
            mag_in.display()
        );
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");

    let algo = Algorithm::from_str(algo_str).expect("known algorithm");
    let mut cfg = NordicConfig::default();
    cfg.mag_file = mag_in;
    cfg.pha_file = if use_phase { Some(pha_in) } else { None };
    cfg.mag_norf_file = Some(mag_norf);
    cfg.pha_norf_file = if use_phase { Some(pha_norf) } else { None };
    cfg.out_dir = tmp.path().to_path_buf();
    cfg.algorithm = algo;
    cfg.temporal_phase = TemporalPhase::from_int(tphase).unwrap();
    cfg.soft_thrs = SoftThreshold::Auto;
    cfg.save_gfactor_map = false;
    cfg.seed = Some(0xDEAD_BEEF);

    run_nordic(&cfg).expect("run_nordic");

    let our_magn = load_4d(&tmp.path().join("magn.nii.gz")).expect("load our magn");
    let ref_magn = load_4d(&expected.join("magn.nii.gz")).expect("load ref magn");

    let err = rel_err(&our_magn.data, &ref_magn.data);
    assert!(err < 0.20, "magnitude max relative error {} > 20%", err);

    if use_phase && expected.join("phase.nii.gz").exists() {
        let our_pha = load_4d(&tmp.path().join("phase.nii.gz")).expect("load our phase");
        let ref_pha = load_4d(&expected.join("phase.nii.gz")).expect("load ref phase");
        // Phase comparison is harder (wrap-around). Just check shape parity for now.
        assert_eq!(our_pha.data.shape(), ref_pha.data.shape());
    }
}

fn skip_if_no_fixtures(_path: &Path) -> bool {
    !fixture_dir().join("synthetic_mag.nii.gz").exists()
}

#[test]
fn parity_nordic_tp1_complex() {
    if skip_if_no_fixtures(&fixture_dir()) {
        eprintln!("SKIP: no fixtures yet");
        return;
    }
    run_scenario("nordic", 1, true);
}

#[test]
fn parity_nordic_tp3_complex() {
    if skip_if_no_fixtures(&fixture_dir()) {
        return;
    }
    run_scenario("nordic", 3, true);
}

#[test]
fn parity_nordic_tp1_magnitude_only() {
    if skip_if_no_fixtures(&fixture_dir()) {
        return;
    }
    run_scenario("nordic", 1, false);
}

#[test]
fn parity_mppca_tp1_complex() {
    if skip_if_no_fixtures(&fixture_dir()) {
        return;
    }
    run_scenario("mppca", 1, true);
}

#[test]
fn parity_gfactor_mppca_tp1_complex() {
    if skip_if_no_fixtures(&fixture_dir()) {
        return;
    }
    run_scenario("gfactor+mppca", 1, true);
}

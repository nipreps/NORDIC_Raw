//! Minimal "open file → run NORDIC → read output" example.
//!
//! Run with:  `cargo run --release --example end_to_end -- mag.nii.gz out_dir`

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use nordic_rs::{run_nordic, Algorithm, NordicConfig, SoftThreshold, TemporalPhase};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: end_to_end <mag.nii.gz> <out_dir>");
        return ExitCode::from(2);
    }
    let mag = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);

    let mut cfg = NordicConfig::default();
    cfg.mag_file = mag;
    cfg.out_dir = out;
    cfg.algorithm = Algorithm::Mppca;
    cfg.temporal_phase = TemporalPhase::One;
    cfg.soft_thrs = SoftThreshold::Auto;

    match run_nordic(&cfg) {
        Ok(o) => {
            println!("magnitude written to {}", o.magnitude_path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("nordic-rs failed: {e}");
            ExitCode::FAILURE
        }
    }
}

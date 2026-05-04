//! `nordic-rs` CLI — single-run wrapper that mirrors `nordic_filewise.py`.

use std::path::PathBuf;

use clap::Parser;
use nordic_rs::{run_nordic, Algorithm, NordicConfig, NordicError, SoftThreshold, TemporalPhase};

#[derive(Parser, Debug)]
#[command(version, about = "NORDIC denoising (Rust)", long_about = None)]
struct Cli {
    #[arg(short = 'm', long = "magnitude")]
    mag_file: PathBuf,

    #[arg(short = 'p', long = "phase")]
    pha_file: Option<PathBuf>,

    #[arg(long = "mag-norf")]
    mag_norf_file: Option<PathBuf>,

    #[arg(long = "phase-norf")]
    pha_norf_file: Option<PathBuf>,

    #[arg(long = "out-dir")]
    out_dir: PathBuf,

    #[arg(long = "factor-error", default_value_t = 1.0)]
    factor_error: f32,

    #[arg(long = "full-dynamic-range", default_value_t = false)]
    full_dynamic_range: bool,

    #[arg(long = "temporal-phase", default_value_t = 1)]
    temporal_phase: u8,

    #[arg(long = "algorithm", default_value = "nordic")]
    algorithm: String,

    #[arg(long = "patch-overlap-gfactor", default_value_t = 2)]
    patch_overlap_gfactor: usize,

    #[arg(long = "patch-overlap-pca", default_value_t = 2)]
    patch_overlap_pca: usize,

    #[arg(long = "phase-slice-average-for-kspace-centering", default_value_t = false)]
    phase_slice_average_for_kspace_centering: bool,

    #[arg(long = "phase-filter-width", default_value_t = 3)]
    phase_filter_width: u32,

    #[arg(long = "save-gfactor-map", default_value_t = false)]
    save_gfactor_map: bool,

    #[arg(long = "soft-thrs", default_value = "auto")]
    soft_thrs: String,

    #[arg(long = "debug", default_value_t = false)]
    debug: bool,

    #[arg(long = "scale-patches", default_value_t = false)]
    scale_patches: bool,

    #[arg(long = "patch-average", default_value_t = false)]
    patch_average: bool,

    #[arg(long = "llr-scale", default_value_t = 1.0)]
    llr_scale: f32,

    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Worker threads for the patch loop. Omit (or pass `0`) to use rayon's
    /// default (= os::cpu_count()). `1` runs serially.
    #[arg(long = "n-jobs")]
    n_jobs: Option<usize>,

    /// Prepended to every output filename (e.g. `--prefix sub-01_` writes
    /// `sub-01_magn.nii.gz`). Concatenated literally, so include any
    /// separator yourself.
    #[arg(long = "prefix", default_value = "")]
    prefix: String,
}

fn parse_soft_thrs(s: &str) -> SoftThreshold {
    match s {
        "auto" => SoftThreshold::Auto,
        "none" | "None" => SoftThreshold::None,
        v => match v.parse::<f32>() {
            Ok(x) => SoftThreshold::Value(x),
            Err(_) => SoftThreshold::Auto,
        },
    }
}

fn main() -> Result<(), NordicError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let algorithm = Algorithm::from_str(&cli.algorithm).unwrap_or_else(|| {
        eprintln!(
            "warning: unknown algorithm {:?}, falling back to nordic",
            cli.algorithm
        );
        Algorithm::Nordic
    });

    let cfg = NordicConfig {
        mag_file: cli.mag_file,
        pha_file: cli.pha_file,
        mag_norf_file: cli.mag_norf_file,
        pha_norf_file: cli.pha_norf_file,
        out_dir: cli.out_dir,
        factor_error: cli.factor_error,
        full_dynamic_range: cli.full_dynamic_range,
        temporal_phase: TemporalPhase::from_int(cli.temporal_phase)?,
        algorithm,
        patch_overlap_gfactor: cli.patch_overlap_gfactor,
        kernel_size_gfactor: None,
        patch_overlap_pca: cli.patch_overlap_pca,
        kernel_size_pca: None,
        phase_slice_average_for_kspace_centering: cli.phase_slice_average_for_kspace_centering,
        phase_filter_width: cli.phase_filter_width,
        save_gfactor_map: cli.save_gfactor_map,
        soft_thrs: parse_soft_thrs(&cli.soft_thrs),
        debug: cli.debug,
        scale_patches: cli.scale_patches,
        patch_average: cli.patch_average,
        llr_scale: cli.llr_scale,
        seed: cli.seed,
        // `--n-jobs 0` is interpreted as "use rayon's default global pool",
        // matching how Python's `n_jobs=None` means auto.
        n_jobs: cli.n_jobs.and_then(|n| if n == 0 { None } else { Some(n) }),
        prefix: cli.prefix,
    };

    let out = run_nordic(&cfg)?;
    println!("magnitude written to {}", out.magnitude_path.display());
    if let Some(p) = out.phase_path {
        println!("phase written to {}", p.display());
    }
    if let Some(p) = out.gfactor_path {
        println!("gfactor written to {}", p.display());
    }
    Ok(())
}

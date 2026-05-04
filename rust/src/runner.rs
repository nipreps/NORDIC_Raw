//! Top-level orchestration: `run_nordic`.

use std::path::PathBuf;

use ndarray::{s, Array3, Array4, Axis, Zip};
use num_complex::Complex32;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::complex::{build_complex, divide_by_absolute_scale, rescale_phase, PhaseMeta};
use crate::config::{NordicConfig, TemporalPhase};
use crate::denoise::denoise_data;
use crate::error::NordicError;
use crate::gfactor::{default_kernel_size as gfactor_default_kernel, estimate_gfactor};
use crate::io::{load_4d, out_path, save_3d, save_4d, NiftiVolume4d};
use crate::phase::{apply_spike_mask_strict, filter_phase};

/// Final outputs of the run, including the on-disk paths.
pub struct NordicOutput {
    pub magnitude_path: PathBuf,
    pub phase_path: Option<PathBuf>,
    pub gfactor_path: Option<PathBuf>,
}

pub fn run_nordic(cfg: &NordicConfig) -> Result<NordicOutput, NordicError> {
    cfg.validate()?;

    // Build a rayon thread pool sized by `cfg.n_jobs` and run all work inside
    // it via `install`. Every `par_iter` in this crate that fires inside the
    // closure picks up this pool implicitly. `None` keeps rayon's global
    // default (= os::cpu_count()), which is what most callers want.
    match cfg.n_jobs {
        None => run_nordic_inner(cfg),
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build rayon thread pool");
            pool.install(|| run_nordic_inner(cfg))
        }
    }
}

fn run_nordic_inner(cfg: &NordicConfig) -> Result<NordicOutput, NordicError> {
    // Auto-create the output directory (matches Python's mkdir(parents=True,
    // exist_ok=True)). Idempotent — succeeds if the dir already exists.
    std::fs::create_dir_all(&cfg.out_dir)?;

    let mag = load_4d(&cfg.mag_file)?;
    let mut mag_data = mag.data;
    let img_header = mag.header;
    let mag_template = NiftiVolume4d {
        data: Array4::<f32>::zeros((1, 1, 1, 1)),
        header: img_header.clone(),
    };
    let _ = mag_template; // referenced by docstrings; the header is what we keep

    let has_complex = cfg.pha_file.is_some();

    // -------- noRF concatenation --------
    let n_noise_vols = if let Some(mag_norf_path) = &cfg.mag_norf_file {
        let mag_norf = load_4d(mag_norf_path)?;
        let nv = mag_norf.data.shape()[3];
        mag_data = concat_along_t(&mag_data, &mag_norf.data)?;
        nv
    } else {
        0
    };

    let mut pha_data_opt: Option<Array4<f32>> = if let Some(pha_path) = &cfg.pha_file {
        let pha = load_4d(pha_path)?;
        if pha.data.shape() != mag_template_shape_after_norf(&mag_data, n_noise_vols).as_slice() {
            // We've already extended mag_data; phase needs the same extension.
            // Defer the shape check to after we concatenate phase too.
        }
        Some(pha.data)
    } else {
        None
    };

    if let (Some(_pha_path), Some(pha_norf_path)) = (&cfg.pha_file, &cfg.pha_norf_file) {
        let pha_norf = load_4d(pha_norf_path)?;
        if pha_norf.data.shape()[3] != n_noise_vols {
            return Err(NordicError::NorfShapeMismatch);
        }
        let pha_data = pha_data_opt.as_ref().unwrap();
        let new = concat_along_t(pha_data, &pha_norf.data)?;
        pha_data_opt = Some(new);
    }

    // Shape check now that both are extended.
    if let Some(pha) = &pha_data_opt {
        if pha.shape() != mag_data.shape() {
            return Err(NordicError::MagPhaseShapeMismatch);
        }
    }

    // -------- magnitude prep + phase rescale --------
    // |mag| as f32 (Python: `np.abs(mag_data).astype(np.float32)`).
    mag_data.mapv_inplace(|v| v.abs());

    let phase_meta_opt: Option<PhaseMeta>;
    let mut complex_data: Array4<Complex32> = if let Some(pha) = pha_data_opt.take() {
        let (rescaled, meta) = rescale_phase(&pha);
        tracing::info!(
            "Phase range: {} {}",
            rescaled.iter().cloned().fold(f32::INFINITY, f32::min),
            rescaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        let combined = build_complex(&mag_data, Some(&rescaled));
        phase_meta_opt = Some(meta);
        combined
    } else {
        phase_meta_opt = None;
        build_complex(&mag_data, None)
    };
    drop(mag_data);

    let absolute_scale = divide_by_absolute_scale(&mut complex_data)?;
    let (n_x, n_y, n_slices, n_vols_total) = complex_data.dim();

    if n_vols_total < 6 {
        return Err(NordicError::TooFewVolumes {
            found: n_vols_total,
        });
    }

    tracing::info!("Estimating slice-dependent phases ...");

    // -------- phase filter --------
    let mut filtered_phase = filter_phase(
        &complex_data,
        cfg.phase_filter_width,
        cfg.temporal_phase,
    );

    // demod = complex * exp(-i*angle(filtered_phase))
    let mut demod = complex_data.clone();
    demodulate_inplace(&mut demod, &filtered_phase);
    scrub_nonfinite(&mut demod);

    tracing::info!("Completed estimating slice-dependent phases");

    if cfg.debug && has_complex {
        // Mirrors Python's `magn_pregfactor_normalized.nii.gz` /
        // `phase_pregfactor_normalized.nii.gz`: |demod * absolute_scale| and
        // its angle, where multiplying by absolute_scale undoes the
        // normalisation done in `divide_by_absolute_scale`.
        let meta = phase_meta_opt.expect("phase_meta when has_complex");
        write_magn_debug(&cfg, "magn_pregfactor_normalized.nii.gz",
                         &demod, absolute_scale, &img_header)?;
        write_phase_debug(&cfg, "phase_pregfactor_normalized.nii.gz",
                          &demod, absolute_scale, meta, &img_header)?;
    }

    // -------- g-factor --------
    let gfactor: Array3<f32>;
    let mut data_has_zero_elements = false;
    let mut gfactor_path: Option<PathBuf> = None;

    if cfg.algorithm.uses_gfactor() {
        // Reduce to first ≤90 (or `kernel_size_gfactor[3]`) volumes.
        let n_keep_for_gf = match cfg.kernel_size_gfactor {
            Some(k) => k[3].min(n_vols_total),
            None => 90.min(n_vols_total),
        };
        let trimmed = demod.slice(s![.., .., .., ..n_keep_for_gf]).to_owned();
        let kernel_size_3d = cfg
            .kernel_size_gfactor
            .map(|k| [k[0], k[1], k[2]])
            .unwrap_or_else(gfactor_default_kernel);
        let gf_out = estimate_gfactor(&trimmed, Some(kernel_size_3d), cfg.patch_overlap_gfactor);
        let mut gf = gf_out.gfactor;

        if cfg.debug {
            // Per-patch diagnostic accumulators from the g-factor pass. These
            // are the four 3D maps Python writes inside `estimate_gfactor`
            // when debug=True; same names, same units (means over patches).
            save_3d(&out_path(&cfg.out_dir, &cfg.prefix,
                              "gfactor_n_components_dropped.nii.gz"),
                    &gf_out.component_threshold, &img_header)?;
            save_3d(&out_path(&cfg.out_dir, &cfg.prefix,
                              "gfactor_energy_removed.nii.gz"),
                    &gf_out.energy_removed, &img_header)?;
            save_3d(&out_path(&cfg.out_dir, &cfg.prefix,
                              "gfactor_SNR_weight.nii.gz"),
                    &gf_out.snr_weight, &img_header)?;
            let n_runs = gf_out.total_weights.mapv(|v| v as f32);
            save_3d(&out_path(&cfg.out_dir, &cfg.prefix,
                              "gfactor_n_patch_runs.nii.gz"),
                    &n_runs, &img_header)?;
        }

        // Python writes the g-factor map from inside `estimate_gfactor`,
        // which runs *before* `data_has_zero_elements` median-fills the
        // uncovered high-x edge. Save it here too, before the fill, so the
        // on-disk map matches Python (zeros at the edge instead of the
        // median value).
        if cfg.save_gfactor_map {
            let mut gfactor_magn = gf.mapv(|v| v.abs());
            for v in gfactor_magn.iter_mut() {
                if v.is_nan() {
                    *v = 0.0;
                }
            }
            if cfg.full_dynamic_range {
                apply_full_dynamic_range_3d(&mut gfactor_magn);
            }
            let path = out_path(&cfg.out_dir, &cfg.prefix, "gfactor.nii.gz");
            save_3d(&path, &gfactor_magn, &img_header)?;
            gfactor_path = Some(path);
        }

        if gf.iter().any(|&v| v == 0.0 || v.is_nan()) {
            // Median-fill zero / NaN entries (matches Python's
            // `gfactor[np.isnan]=0; gfactor[gfactor<1]=median(non-zero)`).
            data_has_zero_elements = true;
            for v in gf.iter_mut() {
                if v.is_nan() {
                    *v = 0.0;
                }
            }
            let nz: Vec<f32> = gf.iter().copied().filter(|&v| v != 0.0).collect();
            let med = median(&nz);
            for v in gf.iter_mut() {
                if *v < 1.0 {
                    *v = med;
                }
            }
        }

        gfactor = gf;
    } else {
        gfactor = Array3::<f32>::ones((n_x, n_y, n_slices));
    }
    drop(demod);

    // -------- second demodulation, with g-factor normalisation --------
    // demod = complex * exp(-i*angle(meanphase)) / gfactor   — meanphase is
    // zero in the standard path, so exp(-i*0) = 1, leaving complex/gfactor.
    let mut demod2 = complex_data;
    if cfg.phase_slice_average_for_kspace_centering {
        // Build meanphase as the mean of the non-noise volumes.
        let mut meanphase = Array3::<Complex32>::zeros((n_x, n_y, n_slices));
        let n_signal = n_vols_total - n_noise_vols;
        for it in 0..n_signal {
            let slab = demod2.index_axis(Axis(3), it);
            Zip::from(&mut meanphase).and(&slab).for_each(|m, s| *m += *s);
        }
        meanphase.mapv_inplace(|v| v / n_signal as f32);
        // demod2 *= exp(-i*angle(meanphase[..., None]))
        for it in 0..n_vols_total {
            let mut slab = demod2.index_axis_mut(Axis(3), it);
            Zip::from(&mut slab).and(&meanphase).for_each(|d, m| {
                let ang = m.arg();
                *d = *d * Complex32::from_polar(1.0, -ang);
            });
        }
    }
    // divide by g-factor.
    for it in 0..n_vols_total {
        let mut slab = demod2.index_axis_mut(Axis(3), it);
        Zip::from(&mut slab).and(&gfactor).for_each(|d, &g| {
            if g != 0.0 {
                *d = *d / g;
            }
        });
    }

    if cfg.debug && has_complex {
        // Python writes `magn_gfactor_normalized.nii.gz` /
        // `phase_gfactor_normalized.nii.gz` after the g-factor divide.
        let meta = phase_meta_opt.expect("phase_meta when has_complex");
        write_magn_debug(&cfg, "magn_gfactor_normalized.nii.gz",
                         &demod2, absolute_scale, &img_header)?;
        write_phase_debug(&cfg, "phase_gfactor_normalized.nii.gz",
                          &demod2, absolute_scale, meta, &img_header)?;
    }

    // -------- noise level --------
    // Match Python's `np.std(noise_data[noise_data != 0])` exactly. For a
    // complex array NumPy computes `sqrt(mean(|c - mean|^2))` — i.e. the
    // population std treating each complex value as a 2D point. Computing
    // `std(|c|)` instead (which is what we did originally) gives the std of
    // the magnitudes (Rayleigh-distributed for circular Gaussians), which is
    // ~0.65σ rather than ~σ√2 — off by ~2.16× and produced a NORDIC NVR
    // threshold ~2× too small.
    let measured_noise = if n_noise_vols > 0 {
        let noise_slab = demod2.slice(s![.., .., .., n_vols_total - n_noise_vols..]);
        let zero = Complex32::new(0.0, 0.0);
        // First pass: complex mean of nonzero entries.
        let mut sum_re = 0.0_f64;
        let mut sum_im = 0.0_f64;
        let mut n = 0usize;
        for c in noise_slab.iter() {
            if *c != zero {
                sum_re += c.re as f64;
                sum_im += c.im as f64;
                n += 1;
            }
        }
        if n == 0 {
            1.0
        } else {
            let mean_re = sum_re / n as f64;
            let mean_im = sum_im / n as f64;
            // Second pass: mean of squared deviation magnitudes.
            let mut sum_dsq = 0.0_f64;
            for c in noise_slab.iter() {
                if *c != zero {
                    let dr = c.re as f64 - mean_re;
                    let di = c.im as f64 - mean_im;
                    sum_dsq += dr * dr + di * di;
                }
            }
            let std = (sum_dsq / n as f64).sqrt() as f32;
            if has_complex {
                std / std::f32::consts::SQRT_2
            } else {
                std
            }
        }
    } else {
        1.0
    };

    if matches!(cfg.temporal_phase, TemporalPhase::Three) {
        apply_spike_mask_strict(&mut filtered_phase, &demod2);
    }

    if cfg.debug {
        // Python writes filtered_phase_{magn,phase,real}.nii.gz right before
        // the third demodulation. Compute each 4D float view one at a time
        // and drop before allocating the next so we only keep one
        // `nx*ny*nz*nt*4` byte scratch buffer alive (~1.25 GB for real fMRI)
        // instead of three.
        let mut buf = Array4::<f32>::zeros(filtered_phase.raw_dim());
        Zip::from(&mut buf).and(&filtered_phase).for_each(|m, c| *m = c.norm());
        save_4d(&out_path(&cfg.out_dir, &cfg.prefix, "filtered_phase_magn.nii.gz"),
                &buf, &img_header)?;
        Zip::from(&mut buf).and(&filtered_phase).for_each(|p, c| *p = c.arg());
        save_4d(&out_path(&cfg.out_dir, &cfg.prefix, "filtered_phase_phase.nii.gz"),
                &buf, &img_header)?;
        Zip::from(&mut buf).and(&filtered_phase).for_each(|r, c| *r = c.re);
        save_4d(&out_path(&cfg.out_dir, &cfg.prefix, "filtered_phase_real.nii.gz"),
                &buf, &img_header)?;
    }

    // -------- third demodulation by filtered phase --------
    demodulate_inplace(&mut demod2, &filtered_phase);
    scrub_nonfinite(&mut demod2);

    if data_has_zero_elements {
        // Fill all-zero voxels (across all volumes) with i.i.d. complex N(0,1)/sqrt(2).
        let mut rng = StdRng::seed_from_u64(cfg.seed.unwrap_or(0xC0FFEE));
        let normal = Normal::new(0.0_f32, 1.0).expect("Normal::new(0,1) is always valid");
        let inv_sqrt2 = 1.0 / std::f32::consts::SQRT_2;
        for ix in 0..n_x {
            for iy in 0..n_y {
                for iz in 0..n_slices {
                    let mut sum_abs = 0.0_f32;
                    for it in 0..n_vols_total {
                        sum_abs += demod2[(ix, iy, iz, it)].norm();
                    }
                    if sum_abs == 0.0 {
                        for it in 0..n_vols_total {
                            let r = normal.sample(&mut rng);
                            let i = normal.sample(&mut rng);
                            demod2[(ix, iy, iz, it)] =
                                Complex32::new(r * inv_sqrt2, i * inv_sqrt2);
                        }
                    }
                }
            }
        }
    }

    // -------- denoise --------
    let denoise_out = denoise_data(
        &demod2,
        cfg.kernel_size_pca,
        cfg.patch_overlap_pca,
        measured_noise,
        cfg.factor_error,
        has_complex,
        cfg.algorithm,
        cfg.soft_thrs.resolve(cfg.algorithm),
        cfg.llr_scale,
        cfg.scale_patches,
        cfg.seed,
    );

    // Compute residual = demod2 - denoised BEFORE we drop either, since the
    // Python `residual_*.nii.gz` files are at the SVD's output scale (pre
    // rescale-by-gfactor / phase / absolute_scale). Reuse `demod2` as the
    // residual buffer instead of cloning — `demod2` isn't read again after
    // `denoise_data` returned, and on real fMRI the clone would be ~2.5 GB.
    let residual_complex = if cfg.debug {
        Zip::from(&mut demod2).and(&denoise_out.denoised).for_each(|x, &d| *x -= d);
        Some(demod2)
    } else {
        drop(demod2);
        None
    };

    if cfg.debug {
        // Per-patch diagnostic accumulators from the denoise pass.
        // `noise.nii.gz` in Python is `|sqrt(noise_var)|` where noise_var is
        // the variance accumulator divided by weights; my DenoiseOutput
        // already holds the divided form, so just sqrt and write.
        let noise_magn = denoise_out.noise.mapv(|v| v.max(0.0).sqrt().abs());
        save_3d(&out_path(&cfg.out_dir, &cfg.prefix, "noise.nii.gz"),
                &noise_magn, &img_header)?;
        save_3d(&out_path(&cfg.out_dir, &cfg.prefix, "energy_removed.nii.gz"),
                &denoise_out.energy_removed, &img_header)?;
        save_3d(&out_path(&cfg.out_dir, &cfg.prefix, "snr_weight.nii.gz"),
                &denoise_out.snr_weight, &img_header)?;
        save_3d(&out_path(&cfg.out_dir, &cfg.prefix, "n_components_removed.nii.gz"),
                &denoise_out.component_threshold, &img_header)?;
        let n_runs = denoise_out.total_weights.mapv(|v| v as f32);
        save_3d(&out_path(&cfg.out_dir, &cfg.prefix, "n_patch_runs.nii.gz"),
                &n_runs, &img_header)?;
    }

    let mut denoised = denoise_out.denoised;

    if let Some(residual) = residual_complex {
        let mut rmagn = Array4::<f32>::zeros(residual.raw_dim());
        Zip::from(&mut rmagn).and(&residual).for_each(|m, c| *m = c.norm());
        save_4d(&out_path(&cfg.out_dir, &cfg.prefix, "residual_magn.nii.gz"),
                &rmagn, &img_header)?;
        if has_complex {
            let mut rphase = Array4::<f32>::zeros(residual.raw_dim());
            Zip::from(&mut rphase).and(&residual).for_each(|p, c| *p = c.arg());
            save_4d(&out_path(&cfg.out_dir, &cfg.prefix, "residual_phase.nii.gz"),
                    &rphase, &img_header)?;
        }
    }

    // -------- rescale: gfactor → e^{i*angle(filtered_phase)} → absolute_scale --------
    for it in 0..n_vols_total {
        let mut slab = denoised.index_axis_mut(Axis(3), it);
        Zip::from(&mut slab).and(&gfactor).for_each(|d, &g| {
            *d = *d * g;
        });
    }
    drop(gfactor);
    Zip::from(&mut denoised).and(&filtered_phase).for_each(|d, p| {
        let ang = p.arg();
        *d = *d * Complex32::from_polar(1.0, ang);
    });
    drop(filtered_phase);
    denoised.mapv_inplace(|c| {
        if c.re.is_nan() || c.im.is_nan() {
            Complex32::new(0.0, 0.0)
        } else {
            c * absolute_scale
        }
    });

    // -------- magnitude output --------
    let mut magn = Array4::<f32>::zeros(denoised.raw_dim());
    Zip::from(&mut magn).and(&denoised).for_each(|m, c| {
        *m = c.norm();
    });
    if cfg.full_dynamic_range {
        apply_full_dynamic_range(&mut magn);
    }
    let magn = if n_noise_vols > 0 {
        magn.slice(s![.., .., .., ..n_vols_total - n_noise_vols]).to_owned()
    } else {
        magn
    };
    let magn_path = out_path(&cfg.out_dir, &cfg.prefix, "magn.nii.gz");
    save_4d(&magn_path, &magn, &img_header)?;

    // -------- phase output --------
    let phase_path = if has_complex {
        let meta = phase_meta_opt.expect("phase_meta is set when has_complex is true");
        let mut phase = Array4::<f32>::zeros(denoised.raw_dim());
        Zip::from(&mut phase).and(&denoised).for_each(|p, c| {
            *p = c.arg();
        });
        let two_pi = 2.0 * std::f32::consts::PI;
        phase.mapv_inplace(|v| (v / two_pi + meta.range_center) * meta.range_norm);
        let phase = if n_noise_vols > 0 {
            phase
                .slice(s![.., .., .., ..n_vols_total - n_noise_vols])
                .to_owned()
        } else {
            phase
        };
        let path = out_path(&cfg.out_dir, &cfg.prefix, "phase.nii.gz");
        save_4d(&path, &phase, &img_header)?;
        Some(path)
    } else {
        None
    };

    tracing::info!("Done!");
    Ok(NordicOutput {
        magnitude_path: magn_path,
        phase_path,
        gfactor_path,
    })
}

/// `dst *= exp(-i*angle(src))`, in place.
fn demodulate_inplace(dst: &mut Array4<Complex32>, src: &Array4<Complex32>) {
    Zip::from(dst).and(src).for_each(|d, s| {
        let ang = s.arg();
        *d = *d * Complex32::from_polar(1.0, -ang);
    });
}

fn scrub_nonfinite(a: &mut Array4<Complex32>) {
    a.mapv_inplace(|c| {
        if c.re.is_nan() || c.im.is_nan() || c.re.is_infinite() || c.im.is_infinite() {
            Complex32::new(0.0, 0.0)
        } else {
            c
        }
    });
}

fn concat_along_t(a: &Array4<f32>, b: &Array4<f32>) -> Result<Array4<f32>, NordicError> {
    if a.shape()[..3] != b.shape()[..3] {
        return Err(NordicError::MagPhaseShapeMismatch);
    }
    let mut combined =
        Array4::<f32>::zeros((a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3] + b.shape()[3]));
    combined.slice_mut(s![.., .., .., ..a.shape()[3]]).assign(a);
    combined
        .slice_mut(s![.., .., .., a.shape()[3]..])
        .assign(b);
    Ok(combined)
}

fn mag_template_shape_after_norf(mag: &Array4<f32>, _n_noise: usize) -> Vec<usize> {
    mag.shape().to_vec()
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 0 {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    } else {
        v[n / 2]
    }
}

fn apply_full_dynamic_range(magn: &mut Array4<f32>) {
    // Python: tmp = sort(flat); sn_scale = 2 * tmp[round(0.99*len)-1];
    // gain = floor(log2(32000 / sn_scale)); magn *= 2**gain.
    let mult = compute_full_dynamic_range_mult(magn.iter().copied());
    if let Some(m) = mult {
        magn.mapv_inplace(|v| v * m);
    }
}

fn apply_full_dynamic_range_3d(gfactor_magn: &mut Array3<f32>) {
    let mult = compute_full_dynamic_range_mult(gfactor_magn.iter().copied());
    if let Some(m) = mult {
        gfactor_magn.mapv_inplace(|v| v * m);
    }
}

/// Write |complex * scale| as a 4D float NIfTI. Used by the debug output
/// path to mirror Python's `magn_pregfactor_normalized.nii.gz` etc.
fn write_magn_debug(
    cfg: &NordicConfig,
    name: &str,
    complex: &Array4<Complex32>,
    scale: f32,
    header: &nifti::NiftiHeader,
) -> Result<(), NordicError> {
    let mut out = Array4::<f32>::zeros(complex.raw_dim());
    Zip::from(&mut out).and(complex).for_each(|m, c| {
        *m = (c * scale).norm();
    });
    save_4d(&out_path(&cfg.out_dir, &cfg.prefix, name), &out, header)
}

/// Write `(angle(complex * scale) / 2π + range_center) * range_norm` — the
/// inverse of the [-π, π] phase rescale applied at load time. Mirrors
/// `phase_pregfactor_normalized.nii.gz` etc.
fn write_phase_debug(
    cfg: &NordicConfig,
    name: &str,
    complex: &Array4<Complex32>,
    scale: f32,
    meta: PhaseMeta,
    header: &nifti::NiftiHeader,
) -> Result<(), NordicError> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = Array4::<f32>::zeros(complex.raw_dim());
    Zip::from(&mut out).and(complex).for_each(|p, c| {
        *p = (c * scale).arg();
    });
    out.mapv_inplace(|v| (v / two_pi + meta.range_center) * meta.range_norm);
    save_4d(&out_path(&cfg.out_dir, &cfg.prefix, name), &out, header)
}

fn compute_full_dynamic_range_mult<I: Iterator<Item = f32>>(values: I) -> Option<f32> {
    let mut flat: Vec<f32> = values.collect();
    if flat.is_empty() {
        return None;
    }
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (0.99 * flat.len() as f64).round() as usize;
    let idx = idx.max(1) - 1;
    let sn_scale = 2.0 * flat[idx];
    if sn_scale > 0.0 {
        let gain = (32000.0_f32 / sn_scale).log2().floor();
        Some(2.0_f32.powf(gain))
    } else {
        None
    }
}

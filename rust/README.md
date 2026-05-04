# nordic-rs

Rust port of the [NORDIC](https://github.com/SteenMoeller/NORDIC_Raw) denoising
pipeline. Mirrors the Python reference implementation in [`nordic/denoise.py`](../nordic/denoise.py)
of this same repository, which is itself a clean-up of [`matlab/NIFTI_NORDIC.m`](../matlab/NIFTI_NORDIC.m).

## Build

```bash
cargo build --release
```

## CLI

The binary mirrors `python -m nordic.cli.nordic_filewise`:

```bash
target/release/nordic-rs \
    -m path/to/mag.nii.gz \
    -p path/to/pha.nii.gz \
    --mag-norf path/to/mag_noRF.nii.gz \
    --phase-norf path/to/pha_noRF.nii.gz \
    --out-dir out/ \
    --algorithm nordic \
    --temporal-phase 1
```

## Library

```rust
use nordic_rs::{run_nordic, Algorithm, NordicConfig, SoftThreshold, TemporalPhase};

let mut cfg = NordicConfig::default();
cfg.mag_file = "mag.nii.gz".into();
cfg.out_dir = "out/".into();
cfg.algorithm = Algorithm::Nordic;
cfg.temporal_phase = TemporalPhase::One;
cfg.soft_thrs = SoftThreshold::Auto;
let out = run_nordic(&cfg)?;
```

See `examples/end_to_end.rs` for a runnable version.

## Tests

```bash
# Unit tests (Tukey, FFT round-trips, patch indexer, …)
cargo test --lib

# Integration smoke tests (each algorithm over a synthetic 3-vol input)
cargo test --test smoke

# Parity vs. the Python reference. Reference outputs must first be baked under
# tests/fixtures/expected/ — run scripts/regenerate_fixtures.sh inside WSL.
cargo test --test parity
```

## Differences vs. the Python reference

Bit-exact parity with NumPy is not a goal. We use:

- A different RNG (`rand::rngs::StdRng` rather than NumPy's PCG).
- A different SVD implementation (`faer` rather than LAPACK `gesdd`).
- `Complex<f32>` end-to-end (NumPy's `linalg.svd` upcasts to `complex128` then
  re-downcasts).

The parity tests therefore use a loose `~1%` relative tolerance on bright voxels.

## Layout

| File | Maps to in `nordic/denoise.py` |
| --- | --- |
| `src/runner.rs` | `run_nordic` (top-level orchestrator) |
| `src/gfactor.rs` | `estimate_gfactor` |
| `src/denoise.rs` | `denoise_data` |
| `src/subpatch.rs` | `subfunction_loop_for_nvr_avg_update` (Y/Z patch loop) |
| `src/patch.rs` | the `kept_x_patches` / `y_patches.append(...)` indexing tricks |
| `src/mppca.rs` | the `soft_thrs == 10` MP-PCA cutoff branch |
| `src/svd.rs` | `np.linalg.svd(..., full_matrices=False)` wrapper |
| `src/threshold.rs` | the 10-draw NVR threshold mean |
| `src/phase.rs` | `filter_phase` + spike masks |
| `src/fft2d.rs`, `src/tukey.rs` | the 2D FFT and Tukey window used inside `filter_phase` |
| `src/complex.rs` | mag+phase → complex, phase rescale, absolute_scale |
| `src/io.rs` | `nb.load(...).get_fdata()` / `nb.Nifti1Image(...).to_filename(...)` |
| `src/config.rs` | the `run_nordic(...)` keyword arguments |
| `src/error.rs` | typed errors (replaces Python's `assert` and `raise`) |

## RNG and reproducibility

The NVR threshold computation and the data-has-zero-elements zero-fill both
need a Gaussian RNG. Pass `cfg.seed = Some(seed)` to make a run reproducible;
otherwise the crate seeds with a fixed default (`0xC0FFEE`), so it is
deterministic but not the same as Python (which uses an unseeded global).

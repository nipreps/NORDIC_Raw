//! `nordic-rs` — Rust port of the NORDIC denoising pipeline.
//!
//! See [`run_nordic`] for the public entry point and [`NordicConfig`] for the
//! knobs. The Python source of truth is `nordic/denoise.py` in the same
//! repository.
//!
//! ## Differences vs. the Python reference
//!
//! Bit-exact parity is *not* a goal. We use a different RNG and a different
//! complex SVD (faer instead of LAPACK). The chosen tolerance for the parity
//! tests in `tests/parity.rs` is ~1% relative error on the denoised magnitude.

pub mod complex;
pub mod config;
pub mod denoise;
pub mod error;
pub mod fft2d;
pub mod gfactor;
pub mod io;
pub mod mppca;
pub mod patch;
pub mod phase;
pub mod runner;
pub mod subpatch;
pub mod svd;
pub mod threshold;
pub mod tukey;

pub use config::{Algorithm, NordicConfig, SoftThreshold, TemporalPhase};
pub use error::NordicError;
pub use runner::{run_nordic, NordicOutput};

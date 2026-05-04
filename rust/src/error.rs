//! Error type for the public API.

use std::path::PathBuf;

use thiserror::Error;

/// All failures `nordic-rs` can return.
///
/// Variants are deliberately specific: callers (and especially the CLI) can
/// match on these and decide whether to retry, log, or bail.
#[derive(Debug, Error)]
pub enum NordicError {
    #[error("input file does not exist: {0}")]
    MissingInput(PathBuf),

    #[error("magnitude and phase data must have the same shape")]
    MagPhaseShapeMismatch,

    #[error("magnitude and phase noRF data must have the same shape")]
    NorfShapeMismatch,

    #[error(
        "if mag+phase data are provided and a mag noRF file is provided, \
         a phase noRF file is required as well"
    )]
    NorfPhaseRequired,

    #[error("too few volumes ({found}); NORDIC requires at least 6")]
    TooFewVolumes { found: usize },

    #[error("invalid temporal_phase {0}; must be one of 0,1,2,3")]
    InvalidTemporalPhase(u8),

    #[error("invalid phase_filter_width {0}; must be in 1..=10")]
    InvalidPhaseFilterWidth(i32),

    #[error("patch_average=true is unimplemented (Python raises NotImplementedError)")]
    PatchAverageUnsupported,

    #[error("input arrays must be 4D, got {dims} dims")]
    NotFourDimensional { dims: usize },

    #[error("absolute_scale was zero — first volume has no non-zero magnitude voxels")]
    ZeroAbsoluteScale,

    #[error("NIfTI I/O error: {0}")]
    Nifti(#[from] nifti::NiftiError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

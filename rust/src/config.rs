//! User-facing configuration for [`crate::run_nordic`].
//!
//! Mirrors the keyword arguments of `nordic.denoise.run_nordic` in Python.

use std::path::PathBuf;

use crate::error::NordicError;

/// Three top-level pipelines, exactly as in Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Nordic,
    Mppca,
    GfactorMppca,
}

impl Algorithm {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "nordic" => Some(Self::Nordic),
            "mppca" => Some(Self::Mppca),
            "gfactor+mppca" => Some(Self::GfactorMppca),
            _ => None,
        }
    }

    /// Whether this pipeline runs the g-factor estimation pass at all.
    pub fn uses_gfactor(self) -> bool {
        matches!(self, Self::Nordic | Self::GfactorMppca)
    }

    /// Mirror of Python's `'mppca' in algorithm` test for the `'auto'` soft-thrs branch.
    pub fn is_mppca_family(self) -> bool {
        matches!(self, Self::Mppca | Self::GfactorMppca)
    }
}

/// Phase-correction mode — directly maps to Python's `temporal_phase` int.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalPhase {
    Off,
    One,
    Two,
    Three,
}

impl TemporalPhase {
    pub fn from_int(v: u8) -> Result<Self, NordicError> {
        match v {
            0 => Ok(Self::Off),
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            other => Err(NordicError::InvalidTemporalPhase(other)),
        }
    }

    pub fn as_int(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

/// Sum type for `soft_thrs`.
///
/// Python accepts either the string `'auto'`, a number, or `None`. In Rust we
/// use a tagged enum and resolve `Auto` against the algorithm at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoftThreshold {
    Auto,
    Value(f32),
    None,
}

impl SoftThreshold {
    /// Resolve `Auto` against the algorithm exactly as Python does:
    /// MPPCA family → 10.0, NORDIC → None.
    pub fn resolve(self, algo: Algorithm) -> Option<f32> {
        match self {
            Self::Auto => {
                if algo.is_mppca_family() {
                    Some(10.0)
                } else {
                    None
                }
            }
            Self::Value(v) => Some(v),
            Self::None => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NordicConfig {
    pub mag_file: PathBuf,
    pub pha_file: Option<PathBuf>,
    pub mag_norf_file: Option<PathBuf>,
    pub pha_norf_file: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub factor_error: f32,
    pub full_dynamic_range: bool,
    pub temporal_phase: TemporalPhase,
    pub algorithm: Algorithm,
    pub patch_overlap_gfactor: usize,
    pub kernel_size_gfactor: Option<[usize; 4]>,
    pub patch_overlap_pca: usize,
    pub kernel_size_pca: Option<[usize; 3]>,
    pub phase_slice_average_for_kspace_centering: bool,
    pub phase_filter_width: u32,
    pub save_gfactor_map: bool,
    pub soft_thrs: SoftThreshold,
    pub debug: bool,
    pub scale_patches: bool,
    pub patch_average: bool,
    pub llr_scale: f32,
    /// Optional deterministic seed for the NVR threshold draws and zero-fill.
    /// `None` means "use the same fixed seed", since Python is also unseeded.
    pub seed: Option<u64>,
    /// `None` → use rayon's default thread pool. `Some(1)` → run in the
    /// calling thread. Anything else picks a custom rayon pool size.
    pub n_jobs: Option<usize>,
    /// String prepended to every output filename. Empty by default. The
    /// caller supplies any separator they want — `prefix` is concatenated
    /// literally to the front of `magn.nii.gz`, `phase.nii.gz`, etc.
    pub prefix: String,
}

impl Default for NordicConfig {
    fn default() -> Self {
        Self {
            mag_file: PathBuf::new(),
            pha_file: None,
            mag_norf_file: None,
            pha_norf_file: None,
            out_dir: PathBuf::from("."),
            factor_error: 1.0,
            full_dynamic_range: false,
            temporal_phase: TemporalPhase::One,
            algorithm: Algorithm::Nordic,
            patch_overlap_gfactor: 2,
            kernel_size_gfactor: None,
            patch_overlap_pca: 2,
            kernel_size_pca: None,
            phase_slice_average_for_kspace_centering: false,
            phase_filter_width: 3,
            save_gfactor_map: true,
            soft_thrs: SoftThreshold::Auto,
            debug: false,
            scale_patches: false,
            patch_average: false,
            llr_scale: 1.0,
            seed: None,
            n_jobs: None,
            prefix: String::new(),
        }
    }
}

impl NordicConfig {
    /// Mirrors Python's pre-flight asserts at the top of `run_nordic`.
    pub fn validate(&self) -> Result<(), NordicError> {
        if !self.mag_file.exists() {
            return Err(NordicError::MissingInput(self.mag_file.clone()));
        }
        if let Some(p) = &self.pha_file {
            if !p.exists() {
                return Err(NordicError::MissingInput(p.clone()));
            }
        }
        if let Some(p) = &self.mag_norf_file {
            if !p.exists() {
                return Err(NordicError::MissingInput(p.clone()));
            }
        }
        if let Some(p) = &self.pha_norf_file {
            if !p.exists() {
                return Err(NordicError::MissingInput(p.clone()));
            }
        }
        if self.pha_file.is_some()
            && self.mag_norf_file.is_some()
            && self.pha_norf_file.is_none()
        {
            return Err(NordicError::NorfPhaseRequired);
        }
        if !(1..=10).contains(&(self.phase_filter_width as i32)) {
            return Err(NordicError::InvalidPhaseFilterWidth(
                self.phase_filter_width as i32,
            ));
        }
        if self.patch_average {
            return Err(NordicError::PatchAverageUnsupported);
        }
        Ok(())
    }
}

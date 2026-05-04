"""Generate small deterministic NIfTI fixtures for nordic-rs parity tests.

Produces:
  synthetic_mag.nii.gz       (32, 32, 8, 20)  int16 magnitude
  synthetic_pha.nii.gz       (32, 32, 8, 20)  int16 phase, scaled to int range
  synthetic_mag_noRF.nii.gz  (32, 32, 8, 2)   int16 magnitude (noRF, signal-free)
  synthetic_pha_noRF.nii.gz  (32, 32, 8, 2)   int16 phase   (noRF, signal-free)

Run from inside `nordicenv`:
    micromamba run -p $HOME/micromamba/envs/nordicenv \\
        python tests/fixtures/synthetic_make.py
"""

import numpy as np
import nibabel as nb

NX, NY, NZ, NT = 32, 32, 8, 20
N_NOISE = 2
SEED = 0xC0FFEE


def main():
    rng = np.random.default_rng(SEED)

    # Phantom: a smooth Gaussian blob plus a small block, both with a slow
    # phase ramp. Adding mild i.i.d. noise simulates the NORDIC use case.
    xs = np.arange(NX) - NX / 2
    ys = np.arange(NY) - NY / 2
    zs = np.arange(NZ) - NZ / 2
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    R2 = X * X + Y * Y + Z * Z * 4
    base_mag = 100.0 * np.exp(-R2 / 80.0) + 10.0
    block = np.zeros_like(base_mag)
    block[10:18, 10:18, 2:6] = 30.0
    base_mag = base_mag + block

    # Slow phase ramp.
    base_phase = 0.01 * (X + 0.5 * Y) + 0.0 * Z

    # Time series: amplitude is the same baseline plus some temporal
    # fluctuation so denoising has signal to estimate.
    t_axis = np.linspace(0, 1, NT)
    activation = 5.0 * np.sin(2 * np.pi * t_axis)[None, None, None, :]
    noise_std = 4.0
    noise_real = rng.normal(0, noise_std, size=(NX, NY, NZ, NT))
    noise_imag = rng.normal(0, noise_std, size=(NX, NY, NZ, NT))

    mag_signal = base_mag[..., None] + activation
    pha_signal = base_phase[..., None] + 0.0 * activation
    complex_signal = mag_signal * np.exp(1j * pha_signal) + (noise_real + 1j * noise_imag)

    mag = np.abs(complex_signal)
    pha = np.angle(complex_signal)

    # noRF (no-excitation) — pure noise, same dist as the signal noise.
    norf_real = rng.normal(0, noise_std, size=(NX, NY, NZ, N_NOISE))
    norf_imag = rng.normal(0, noise_std, size=(NX, NY, NZ, N_NOISE))
    norf_complex = norf_real + 1j * norf_imag
    mag_norf = np.abs(norf_complex)
    pha_norf = np.angle(norf_complex)

    # Save as int16 with appropriate scl_slope/scl_inter to test the integer
    # rescaling branch end-to-end.
    affine = np.eye(4, dtype=np.float32)

    # --- magnitude: store raw float32 cast to int16 with slope=1, inter=0 ---
    save_int16(mag, affine, "synthetic_mag.nii.gz", scale=1.0)
    save_int16(mag_norf, affine, "synthetic_mag_noRF.nii.gz", scale=1.0)

    # --- phase: scale [-pi, pi] to int16 range [-2048, 2047] for realism ---
    pha_int_range = 2048.0
    save_int16(
        (pha / np.pi) * pha_int_range,
        affine,
        "synthetic_pha.nii.gz",
        scale=1.0,
    )
    save_int16(
        (pha_norf / np.pi) * pha_int_range,
        affine,
        "synthetic_pha_noRF.nii.gz",
        scale=1.0,
    )


def save_int16(arr, affine, path, scale=1.0):
    arr16 = np.clip(np.round(arr / scale), -32768, 32767).astype(np.int16)
    img = nb.Nifti1Image(arr16, affine)
    img.header.set_slope_inter(scale, 0.0)
    img.to_filename(path)


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

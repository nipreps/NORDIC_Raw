"""Another Python attempt at NORDIC."""

import os
from copy import deepcopy
from pathlib import Path

import nibabel as nb
import numpy as np
from scipy.signal.windows import tukey


def estimate_noise_level(norf_data, is_complex=False):
    """Estimate the noise level in a noise scan file."""
    norf_data[np.isnan(norf_data)] = 0
    norf_data[np.isinf(norf_data)] = 0
    noise_level = np.std(norf_data[norf_data != 0])
    if is_complex:
        noise_level = noise_level / np.sqrt(2)
    return noise_level


def run_nordic(
    mag_file,
    pha_file,
    mag_norf_file,
    pha_norf_file,
    out_dir,
    factor_error=1,
    full_dynamic_range=False,
    temporal_phase=1,
    algorithm="nordic",
    kernel_size_gfactor=None,
    kernel_size_PCA=None,
    phase_slice_average_for_kspace_centering=False,
    phase_filter_width=3,
    NORDIC_patch_overlap=2,
    gfactor_patch_overlap=2,
    save_gfactor_map=True,
):
    """Run NORDIC.

    Parameters
    ----------
    factor_error : float
        error in gfactor estimation.
        >1 use higher noisefloor. <1 use lower noisefloor. Default is 1.
    full_dynamic_range : bool
        False keep the input scale, output maximizes range. Default is False.
    temporal_phase : {1, 2, 3}
        Correction for slice and time-specific phase.
        Values > 0 will calculate a standard low-pass filtered phase map.
        Value == 2 will perform a secondary step for filtered phase with residual spikes.
        Value == 3 will do the same thing as 2, but with a more aggressive mask and
        done after other steps, including g-factor normalization.
        1 was default, 3 now in dMRI due to phase errors in some data.
    algorithm : {'nordic', 'mppca', 'mppca+nordic'}
        'mppca+nordic': NORDIC gfactor with MP estimation. ARG.MP = 1 and ARG.NORDIC = 0
        'mppca': MP without gfactor correction. ARG.MP = 2 and ARG.NORDIC = 0
        'nordic': NORDIC only. ARG.MP = 0 and ARG.NORDIC = 1
    kernel_size_gfactor : len-3 list
        defautl is []
    kernel_size_PCA : None or len-3 list
        Default is None.
        default is val1=val2=val3; ratio of 11:1 between spatial and temproal voxels
    phase_slice_average_for_kspace_centering : bool
        if False, not used, if True the series average pr slice is first removed
        default is now False
    phase_filter_width : int
        Specifies the width of the smoothing filter for the phase.
        Must be an int between 1 and 10.
        Default is now 3.
    save_gfactor_map : bool
        saves the RELATIVE gfactor, 2 saves the gfactor and does not complete the NORDIC processing
    NORDIC_patch_overlap
        Default is 2.
    gfactor_patch_overlap
        Default is 2.

    Notes
    -----
    The basic procedure, per Vizioli et al. (2021), is as follows:

    1.  Estimate the geometry-factor (g-factor) noise map based on Moeller et al. (2020).
        -   The formula is designed for GRAPPA reconstruction and detailed in
            Breuer et al. (2009).
        -   This reflects noise amplification from the mathematical algorithm
            used to resolve aliased signals in accelerated acquisitions.

    2.  To ensure i.i.d. noise the series is normalized with the calculated g-factor
        kernels as m(r, t) / g(r).
    3.  From the normalized series, the Casorati matrix Y = [y1, y2, ..., yN] is formed,
        where yj is a column vector that contains the voxel values in each patch.
        -   The concept of NORDIC is to estimate the underlying matrix X in the
            model where Y = X + N, where N is additive Gaussian noise.
        -   For NORDIC, patch size is selected to be a sufficiently small size
            so that no two voxels within the patch are unaliased from the same
            acquired data for the given acceleration rate,
            ensuring that the noise in the pixels of the patch are all independent.
            -   TS: I don't know what this means.

    4.  The low-rank estimate of Y is calculated as YL = U * S_lambda_thr * V.T,
        where the singular values in S are set to 0 if S(i) < lambda_thr.
    5.  After re-forming the series mLLR(r, t) with patch averaging,
        the normalization of the calculated g-factor is reversed as
        mNORDIC(r, t) = mLLR(r, t) * g(r).

    Other miscellany:

    - np.angle(complex) returns the phase in radians.
    - np.abs(complex) returns the magnitude.
    - np.exp(1j * phase) returns the complex number with the given phase.
    - mag * np.exp(1j * phase) returns the complex number.
    - NVR = noise variance reduction
    - LLR = locally low-rank
    - In MATLAB, min(complex) and max(complex) use the magnitude of the complex number.
    - In Python, min(complex) and max(complex) use the whole complex number,
      so you need to do np.abs(complex) before np.min/np.max.
    - In MATLAB, niftiread appears to modify the data. I ended up making a mat file
      containing the data loaded by MATLAB and converting it to a NIfTI image using
      Python for testing.
    """
    out_dir = Path(out_dir)

    ARG = {
        "factor_error": factor_error,
        "temporal_phase": temporal_phase,
        "algorithm": algorithm,
        "kernel_size_gfactor": kernel_size_gfactor,
        "kernel_size_PCA": kernel_size_PCA,
        "phase_slice_average_for_kspace_centering": phase_slice_average_for_kspace_centering,
        "phase_filter_width": phase_filter_width,
        "NORDIC_patch_overlap": NORDIC_patch_overlap,
        "gfactor_patch_overlap": gfactor_patch_overlap,
        "save_gfactor_map": save_gfactor_map,
        "out_dir": out_dir,
        "filename": str(out_dir / "out"),
    }
    QQ = {}

    mag_img = nb.load(mag_file)
    mag_data = mag_img.get_fdata()

    if pha_file:
        has_complex = True
        pha_img = nb.load(pha_file)
        pha_data = pha_img.get_fdata().astype(np.float32)

    n_noise_vols = 0
    if mag_norf_file:
        mag_norf_img = nb.load(mag_norf_file)
        mag_norf_data = mag_norf_img.get_fdata()
        n_noise_vols = mag_norf_data.shape[3]
        mag_data = np.concatenate((mag_data, mag_norf_data), axis=3)

        if has_complex:
            pha_norf_img = nb.load(pha_norf_file)
            pha_norf_data = pha_norf_img.get_fdata().astype(np.float32)
            pha_data = np.concatenate((pha_data, pha_norf_data), axis=3)

    # Take the absolute value of the magnitude data
    mag_data = np.abs(mag_data).astype(np.float32)

    if has_complex:
        # Scale the phase data (with noise volumes) to -pi to pi
        phase_range = np.max(pha_data)
        phase_range_min = np.min(pha_data)
        range_norm = phase_range - phase_range_min
        range_center = (phase_range + phase_range_min) / range_norm * 1 / 2
        pha_data = (pha_data / range_norm - range_center) * 2 * np.pi
        print("Phase range: ", np.min(pha_data), np.max(pha_data))
    else:
        raise ValueError("Phase data is required for NORDIC.")

    # Combine magnitude and phase into complex-valued data
    complex_data = mag_data * np.exp(1j * pha_data)
    n_x, n_y, n_slices, n_vols = complex_data.shape

    # Select the first volume of the complex data and get the magnitude
    first_volume = np.abs(complex_data[..., 0])

    # Find the minimum non-zero magnitude in the first volume and divide the complex data by it
    ARG["ABSOLUTE_SCALE"] = np.min(first_volume[first_volume != 0])
    complex_data = complex_data / ARG["ABSOLUTE_SCALE"]

    if complex_data.shape[3] < 6:
        raise ValueError("Two few volumes in the input data")

    print("Estimating slice-dependent phases ...")

    # Create mean 3D array from all non-noise volumes of shape (X, Y, Z)
    # XXX: What is meanphase?
    meanphase = np.mean(complex_data[..., :-n_noise_vols], axis=3)
    # Multiply the mean array by either 1 or 0 (default is 0)
    meanphase = meanphase * ARG["phase_slice_average_for_kspace_centering"]
    # Now this is just an array of all -0.+0.j
    # np.exp(-1j * np.angle(meanphase[..., None])) is just an array of all 1.+0.j

    # Preallocate 4D array of zeros
    # XXX: WHAT IS DD_phase?
    # XXX: DD_phase results are very similar between MATLAB and Python at this point.
    # The difference image looks like white noise.
    DD_phase = np.zeros_like(complex_data)

    # If the temporal phase is 1 - 3, smooth the phase data
    # Except it's not just the phase data???
    if ARG["temporal_phase"] > 0:
        # Loop over slices backwards
        for i_slice in range(n_slices)[::-1]:
            # Loop over volumes forward, including the noise volumes(???)
            for j_vol in range(n_vols):
                # Grab the 2D slice of the 4D array
                slice_data = complex_data[:, :, i_slice, j_vol]

                # Apply 1D FFT to the 2D slice
                for k_dim in range(2):
                    slice_data = np.fft.ifftshift(
                        np.fft.ifft(
                            np.fft.ifftshift(slice_data, axes=[k_dim]),
                            n=None,
                            axis=k_dim,
                        ),
                        axes=[k_dim],
                    )

                # Apply Tukey window to the filtered 2D slice
                # I've checked that this works on simulated data.
                # tmp = bsxfun(@times,tmp,reshape(tukeywin(n_y,1).^phase_filter_width,[1 n_y]));
                tukey_window = tukey(n_y, 1) ** ARG["phase_filter_width"]
                tukey_window_reshaped = tukey_window.reshape(1, n_y)
                slice_data = slice_data * tukey_window_reshaped
                # tmp = bsxfun(@times,tmp,reshape(tukeywin(n_x,1).^phase_filter_width,[n_x 1]));
                tukey_window = tukey(n_x, 1).T ** ARG["phase_filter_width"]
                tukey_window_reshaped = tukey_window.reshape(n_x, 1)
                slice_data = slice_data * tukey_window_reshaped

                # Apply 1D IFFT to the filtered 2D slice and store in the 4D array
                for k_dim in range(2):
                    slice_data = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.fftshift(slice_data, axes=[k_dim]),
                            n=None,
                            axis=k_dim,
                        ),
                        axes=[k_dim],
                    )
                DD_phase[:, :, i_slice, j_vol] = slice_data

    # Multiply the 4D array by the exponential of the angle of the filtered phase
    # np.angle(complex) = phase in real radian values
    KSP2 = complex_data * np.exp(-1j * np.angle(DD_phase))

    print("Completed estimating slice-dependent phases")
    if not ARG["kernel_size_gfactor"]:
        # Select first 90 (or fewer, if run is shorter) volumes from 4D array
        KSP2 = KSP2[:, :, :, : min(90, n_vols + 1)]
    else:
        # Select first N volumes from 4D array, based on kernel_size_gfactor(4)
        KSP2 = KSP2[:, :, :, : min(ARG["kernel_size_gfactor"][3], n_vols + 1)]

    # Replace NaNs and Infs with zeros
    KSP2[np.isnan(KSP2)] = 0
    KSP2[np.isinf(KSP2)] = 0

    # Write out corrected magnitude and phase images
    if has_complex:
        mag_data = np.abs(KSP2 * ARG["ABSOLUTE_SCALE"])
        mag_img = nb.Nifti1Image(mag_data, mag_img.affine, mag_img.header)
        mag_img.to_filename(out_dir / "magn_pregfactor_normalized.nii.gz")

        pha_data = np.angle(KSP2 * ARG["ABSOLUTE_SCALE"])
        pha_data = (pha_data / (2 * np.pi) + range_center) * range_norm
        pha_img = nb.Nifti1Image(pha_data, mag_img.affine, mag_img.header)
        pha_img.to_filename(out_dir / "phase_pregfactor_normalized.nii.gz")
        del mag_data, pha_data

    if not ARG["kernel_size_gfactor"]:
        ARG["kernel_size"] = [14, 14, 1]
    else:
        if not np.array_equal(ARG["kernel_size_gfactor"][:3], ARG["kernel_size"]):
            print(
                f"Changing kernel size from {ARG['kernel_size']} to "
                f"{ARG['kernel_size_gfactor'][:3]} for g-factor estimation"
            )
        ARG["kernel_size"] = [int(i) for i in ARG["kernel_size_gfactor"][:3]]

    n_x_patches = n_x - ARG["kernel_size"][0]
    QQ["KSP_processed"] = np.zeros(n_x_patches, dtype=int)
    ARG["patch_average"] = False
    ARG["patch_average_sub"] = ARG["gfactor_patch_overlap"]
    ARG["LLR_scale"] = 0
    ARG["NVR_threshold"] = 1
    ARG["soft_thrs"] = 10  # MPPCa   (When Noise varies)
    # ARG['soft_thrs'] = []  # NORDIC (When noise is flat)

    # Preallocate 3D arrays of zeros
    total_patch_weights = np.zeros(KSP2.shape[:3], dtype=int)
    NOISE = np.zeros_like(KSP2[..., 0])
    Component_threshold = np.zeros(KSP2.shape[:3], dtype=float)
    energy_removed = np.zeros(KSP2.shape[:3], dtype=float)
    SNR_weight = np.zeros(KSP2.shape[:3], dtype=float)
    # The original code re-creates KSP_processed here for no reason

    # patch_average is hardcoded as False so this block is always executed.
    if not ARG["patch_average"]:
        # This section seems to set the KSP_processed array to 2 at certain intervals
        # so that those patches will be skipped in the next stage.
        val = int(max(1, int(np.floor(ARG["kernel_size"][0] / ARG["patch_average_sub"]))))
        for nw1 in range(1, val):
            # KSP_processed(1,nw1 : max(1,floor(ARG.ARG['kernel_size'](1)/ARG.patch_average_sub)):end)=2;
            QQ["KSP_processed"][nw1::val] = 2
        QQ["KSP_processed"][-1] = 0

    print("Estimating g-factor ...")
    QQ["ARG"] = deepcopy(ARG)
    master_fast = 1  # What is this?
    # Preallocate 4D array of zeros
    KSP_recon = np.zeros_like(KSP2)
    # Loop over patches in the x-direction
    # Looping over y and z happens within the sub_LLR_Processing function
    for i_x_patch in range(n_x_patches):
        (
            KSP_recon,
            _,
            total_patch_weights,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        ) = sub_LLR_Processing(
            KSP_recon=KSP_recon,
            KSP2=KSP2,
            ARG=ARG,
            patch_num=i_x_patch,
            QQ=QQ,
            master=master_fast,
            total_patch_weights=total_patch_weights,
            NOISE=NOISE,
            Component_threshold=Component_threshold,
            energy_removed=energy_removed,
            SNR_weight=SNR_weight,
        )

    KSP_recon = KSP_recon / total_patch_weights[..., None]
    ARG["NOISE"] = np.sqrt(NOISE / total_patch_weights)
    ARG["Component_threshold"] = Component_threshold / total_patch_weights
    ARG["energy_removed"] = energy_removed / total_patch_weights
    ARG["SNR_weight"] = SNR_weight / total_patch_weights

    component_threshold_img = nb.Nifti1Image(
        ARG["Component_threshold"], mag_img.affine, mag_img.header
    )
    component_threshold_img.to_filename(out_dir / "n_components_retained.nii.gz")

    # ARG2 = ARG
    print("Completed estimating g-factor")
    gfactor = ARG["NOISE"].copy()

    if n_vols < 6:
        gfactor[np.isnan(gfactor)] = 0
        gfactor[gfactor == 0] = np.median(gfactor[gfactor != 0])

    ARG["data_has_zero_elements"] = 0
    if np.sum(gfactor == 0) > 0:
        gfactor[np.isnan(gfactor)] = 0
        gfactor[gfactor < 1] = np.median(gfactor[gfactor != 0])
        ARG["data_has_zero_elements"] = 1

    if algorithm == "mppca":
        gfactor = np.ones(gfactor.shape)

    if save_gfactor_map:
        # Convert complex-valued g-factor to magnitude (absolute)
        gfactor_for_img = np.abs(gfactor)
        gfactor_for_img[np.isnan(gfactor_for_img)] = 0

        tmp = np.sort(np.abs(gfactor_for_img.flatten()))
        # added -1 to match MATLAB indexing
        sn_scale = (2 * tmp[int(np.round(0.99 * len(tmp))) - 1])
        gain_level = np.floor(np.log2(32000 / sn_scale))
        if not full_dynamic_range:
            gain_level = 0

        gfactor_for_img = np.abs(gfactor_for_img) * (2**gain_level)
        gfactor_img = nb.Nifti1Image(gfactor_for_img, mag_img.affine, mag_img.header)
        gfactor_img.to_filename(out_dir / "gfactor.nii.gz")

        n_patch_runs = total_patch_weights.copy()
        n_patch_runs_img = nb.Nifti1Image(n_patch_runs, mag_img.affine, mag_img.header)
        n_patch_runs_img.to_filename(out_dir / "n_patch_runs_gfactor.nii.gz")
        del n_patch_runs, n_patch_runs_img

    # Overwrite KSP2 with the original data
    # meanphase isn't anything useful (just complex-valued zeros)
    KSP2 = complex_data.copy() * np.exp(-1j * np.angle(meanphase[..., None]))
    KSP2 = KSP2 / gfactor[..., None]

    # Write out corrected magnitude and phase images
    if has_complex:
        mag_data = np.abs(KSP2 * ARG["ABSOLUTE_SCALE"])
        mag_img = nb.Nifti1Image(mag_data, mag_img.affine, mag_img.header)
        mag_img.to_filename(out_dir / "magn_gfactor_normalized.nii.gz")

        pha_data = np.angle(KSP2 * ARG["ABSOLUTE_SCALE"])
        pha_data = (pha_data / (2 * np.pi) + range_center) * range_norm
        pha_img = nb.Nifti1Image(pha_data, mag_img.affine, mag_img.header)
        pha_img.to_filename(out_dir / "phase_gfactor_normalized.nii.gz")
        del mag_data, pha_data

    # Calculate noise level from noise volumes
    ARG["measured_noise"] = 1
    if n_noise_vols > 0:
        # BUG: MATLAB version only uses the first noise volume
        KSP2_NOISE = KSP2[..., -n_noise_vols:]
        KSP2_NOISE[np.isnan(KSP2_NOISE)] = 0
        KSP2_NOISE[np.isinf(KSP2_NOISE)] = 0
        ARG["measured_noise"] = np.std(KSP2_NOISE[KSP2_NOISE != 0])

    if has_complex:
        # Rescale the noise level for complex data
        ARG["measured_noise"] = ARG["measured_noise"] / np.sqrt(2)

    if ARG["temporal_phase"] == 3:
        # Secondary step for filtered phase with residual spikes
        for i_slice in range(n_slices)[::-1]:
            for j_vol in range(n_vols):
                KSP2_slice = KSP2[:, :, i_slice, j_vol]
                DD_phase_slice = DD_phase[:, :, i_slice, j_vol]
                phase_diff = np.angle(KSP2_slice / DD_phase_slice)
                mask = (np.abs(phase_diff) > 1) * (np.abs(KSP2_slice) > np.sqrt(2))
                DD_phase2 = DD_phase_slice.copy()
                DD_phase2[mask] = KSP2_slice[mask]
                DD_phase[:, :, i_slice, j_vol] = DD_phase2

    DD_phase_img = nb.Nifti1Image(DD_phase.astype(float), mag_img.affine, mag_img.header)
    DD_phase_img.to_filename(out_dir / "DD_phase.nii.gz")

    KSP2 = KSP2 * np.exp(-1j * np.angle(DD_phase))
    KSP2[np.isnan(KSP2)] = 0
    KSP2[np.isinf(KSP2)] = 0

    if ARG["data_has_zero_elements"]:
        # Fill in zero elements with random noise?
        MASK = np.sum(np.abs(KSP2), axis=3) == 0
        num_zero_elements = np.sum(MASK)
        for i_vol in range(n_vols):
            tmp = KSP2[:, :, :, i_vol]
            tmp[MASK] = (
                np.random.normal(size=num_zero_elements)
                + 1j * np.random.normal(size=num_zero_elements)
            ) / np.sqrt(2)
            KSP2[:, :, :, i_vol] = tmp

    KSP_recon = np.zeros_like(KSP2)
    ARG["kernel_size"] = np.ones(3, dtype=int) * int(np.round(np.cbrt(n_vols * 11)))
    if ARG["kernel_size_PCA"]:
        if not np.array_equal(ARG["kernel_size_PCA"], ARG["kernel_size"]):
            print(
                f"Changing kernel size from {ARG['kernel_size']} to "
                f"{ARG['kernel_size_PCA']} for PCA"
            )
        ARG["kernel_size"] = [int(i) for i in ARG["kernel_size_PCA"]]

    if n_slices <= ARG["kernel_size"][2]:  # Number of slices is less than cubic kernel
        old_kernel_size = ARG["kernel_size"][:]
        ARG["kernel_size"] = (
            np.ones(3, dtype=int) * int(np.round(np.sqrt(n_vols * 11 / n_slices)))
        )
        ARG["kernel_size"][2] = n_slices
        print(
            f"Number of slices is less than cubic kernel. "
            f"Changing kernel size from {old_kernel_size} to {ARG['kernel_size']} for PCA"
        )

    ARG["patch_average"] = False
    ARG["patch_average_sub"] = ARG["NORDIC_patch_overlap"]
    ARG["LLR_scale"] = 1
    ARG["soft_thrs"] = None  # NORDIC (When noise is flat)

    # Build threshold from mean first singular value of random data
    n_iters = 10
    ARG["NVR_threshold"] = 0
    for _ in range(n_iters):
        _, S, _ = np.linalg.svd(np.random.normal(size=(np.prod(ARG["kernel_size"]), n_vols)))
        ARG["NVR_threshold"] += S[0]

    ARG["NVR_threshold"] /= n_iters

    # Scale NVR threshold by measured noise level and error factor
    ARG["NVR_threshold"] *= ARG["measured_noise"] * ARG["factor_error"]
    if has_complex:
        # Scale NVR threshold for complex data
        # Since measured_noise is scaled by sqrt(2) earlier and is only used for NVR_threshold,
        # why not just keep it as-is?
        ARG["NVR_threshold"] *= np.sqrt(2)

    if algorithm in ["mppca", "mppca+nordic"]:
        ARG["soft_thrs"] = 10

    total_patch_weights = np.zeros(KSP2.shape[:3], dtype=int)
    NOISE = np.zeros_like(KSP2[..., 0])  # complex
    Component_threshold = np.zeros(KSP2.shape[:3], dtype=float)
    energy_removed = np.zeros(KSP2.shape[:3], dtype=float)
    SNR_weight = np.zeros(KSP2.shape[:3], dtype=float)

    n_x_patches = n_x - ARG["kernel_size"][0]
    # Reset KSP_processed to zeros for next stage
    QQ["KSP_processed"] = np.zeros(n_x_patches, dtype=int)

    if not ARG["patch_average"]:
        val = max(1, int(np.floor(ARG["kernel_size"][0] / ARG["patch_average_sub"])))
        for nw1 in range(1, val):
            QQ["KSP_processed"][nw1::val] = 2
        QQ["KSP_processed"][-1] = 0

    print("Starting NORDIC ...")
    QQ["ARG"] = deepcopy(ARG)
    master_fast = 1
    # Loop over patches in the x-direction
    # Looping over y and z happens within the sub_LLR_Processing function
    for i_x_patch in range(n_x_patches):
        (
            KSP_recon,
            _,
            total_patch_weights,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        ) = sub_LLR_Processing(
            KSP_recon=KSP_recon,
            KSP2=KSP2,
            ARG=ARG,
            patch_num=i_x_patch,
            QQ=QQ,
            master=master_fast,
            total_patch_weights=total_patch_weights,
            NOISE=NOISE,
            Component_threshold=Component_threshold,
            energy_removed=energy_removed,
            SNR_weight=SNR_weight,
        )

    # Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA.
    # These arrays are summed over patches and need to be scaled by the patch scaling factor,
    # which is typically just the number of patches that contribute to each voxel.
    KSP_recon = KSP_recon / total_patch_weights[..., None]
    ARG["NOISE"] = np.sqrt(NOISE / total_patch_weights)
    ARG["Component_threshold"] = Component_threshold / total_patch_weights
    ARG["energy_removed"] = energy_removed / total_patch_weights
    ARG["SNR_weight"] = SNR_weight / total_patch_weights
    print("Completed NORDIC")

    if has_complex:
        # magn_nordic decays weirdly over time.
        # Write out KSP_recon
        KSP_recon_magn = np.abs(KSP_recon)
        KSP_recon_magn_img = nb.Nifti1Image(KSP_recon_magn, mag_img.affine, mag_img.header)
        KSP_recon_magn_img.to_filename(out_dir / "magn_nordic.nii.gz")
        del KSP_recon_magn, KSP_recon_magn_img

        KSP_recon_phase = np.angle(KSP_recon)
        KSP_recon_phase = (KSP_recon_phase / (2 * np.pi) + range_center) * range_norm
        KSP_recon_phase_img = nb.Nifti1Image(KSP_recon_phase, mag_img.affine, mag_img.header)
        KSP_recon_phase_img.to_filename(out_dir / "phase_nordic.nii.gz")
        del KSP_recon_phase, KSP_recon_phase_img

    residual = KSP2 - KSP_recon

    # Split residuals into magnitude and phase
    residual_magn = np.abs(residual)
    residual_magn_img = nb.Nifti1Image(residual_magn, mag_img.affine, mag_img.header)
    residual_magn_img.to_filename(out_dir / "residual_magn.nii.gz")
    del residual_magn, residual_magn_img

    residual_phase = np.angle(residual)
    residual_phase_img = nb.Nifti1Image(residual_phase, mag_img.affine, mag_img.header)
    residual_phase_img.to_filename(out_dir / "residual_phase.nii.gz")
    del residual, residual_phase, residual_phase_img

    denoised_complex = KSP_recon.copy()
    denoised_complex = denoised_complex * gfactor[:, :, :, None]
    denoised_complex *= np.exp(1j * np.angle(DD_phase))
    denoised_complex = denoised_complex * ARG["ABSOLUTE_SCALE"]  # rescale the data
    denoised_complex[np.isnan(denoised_complex)] = 0

    # Write out number of components removed
    n_components_removed = ARG["Component_threshold"].copy()
    n_components_removed_img = nb.Nifti1Image(n_components_removed, mag_img.affine, mag_img.header)
    n_components_removed_img.to_filename(out_dir / "n_components_removed.nii.gz")
    del n_components_removed, n_components_removed_img

    n_patch_runs = total_patch_weights.copy()
    n_patch_runs_img = nb.Nifti1Image(n_patch_runs, mag_img.affine, mag_img.header)
    n_patch_runs_img.to_filename(out_dir / "n_patch_runs.nii.gz")
    del n_patch_runs, n_patch_runs_img

    if has_complex:
        denoised_magn = np.abs(denoised_complex)  # remove g-factor and noise for DUAL 1
        tmp = np.sort(denoised_magn.flatten())
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        gain_level = np.floor(np.log2(32000 / sn_scale))

        if not full_dynamic_range:
            gain_level = 0

        denoised_magn = denoised_magn * (2**gain_level)
        denoised_magn = nb.Nifti1Image(denoised_magn, mag_img.affine, mag_img.header)
        denoised_magn.to_filename(out_dir / "magn.nii.gz")

        denoised_phase = np.angle(denoised_complex)
        denoised_phase = (denoised_phase / (2 * np.pi) + range_center) * range_norm
        denoised_phase = nb.Nifti1Image(denoised_phase, mag_img.affine, mag_img.header)
        denoised_phase.to_filename(out_dir / "phase.nii.gz")
    else:
        denoised_complex = np.abs(denoised_complex)
        denoised_complex[np.isnan(denoised_complex)] = 0
        tmp = np.sort(np.abs(denoised_complex).flatten())
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        gain_level = np.floor(np.log2(32000 / sn_scale))

        if not full_dynamic_range:
            gain_level = 0

        denoised_complex = np.abs(denoised_complex) * (2**gain_level)
        denoised_img = nb.Nifti1Image(denoised_complex, mag_img.affine, mag_img.header)
        denoised_img.to_filename(out_dir / "magn.nii.gz")

    print("Done!")


def sub_LLR_Processing(
    KSP_recon,
    KSP2,
    ARG,
    patch_num,
    QQ,
    master,
    total_patch_weights,
    NOISE=None,
    Component_threshold=None,
    energy_removed=None,
    SNR_weight=None,
):
    """Perform locally low-rank processing on a chunk of voxels.

    Parameters
    ----------
    KSP_recon : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    KSP2 : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    ARG : dict
        A dictionary of arguments. I will break this up into individual variables later.
        kernel_size, filename, NVR_threshold, soft_thrs, patch_average_sub, patch_scale,
        patch_average.
        NVR_threshold: noise variance reduction threshold
    patch_num : int
        Patch number. Each patch is processed separately.
    QQ : dict
        A dictionary of arguments. I will break this up into individual variables later.
        "KSP_processed" is an np.ndarray of shape (n_x_patches,) and is used
        to track the processing status of each patch. Values may be 0, 1, 2, or 3.
        I don't know what the different values mean though.
        KSP2, ARG['LLR_scale']
    master : int
    total_patch_weights : np.ndarray of shape (n_x, n_y, n_slices)
        Used to scale the outputs, including KSP_recon, NOISE, Component_threshold,
        and energy_removed.
    NOISE : np.ndarray of shape (n_x, n_y, n_slices)
    Component_threshold : np.ndarray of shape (n_x, n_y, n_slices)
        The mean number of singular values removed by the denoising step.
    energy_removed : np.ndarray of shape (n_x, n_y, n_slices)
    SNR_weight : np.ndarray of shape (n_x, n_y, n_slices)

    Returns
    -------
    KSP_recon : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    KSP2 : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    total_patch_weights : np.ndarray of shape (n_x, n_y, n_slices)
    NOISE : np.ndarray of shape (n_x, n_y, n_slices)
    Component_threshold : np.ndarray of shape (n_x, n_y, n_slices)
    energy_removed : np.ndarray of shape (n_x, n_y, n_slices)
    SNR_weight : np.ndarray of shape (n_x, n_y, n_slices)
    """
    import pickle

    _, n_y, _, _ = KSP2.shape
    x_patch_idx = np.arange(0, ARG["kernel_size"][0], dtype=int) + patch_num

    # not being processed also not completed yet
    # DATA_full2 is (x_patch_size, n_y, n_z, n_vols)
    DATA_full2 = None
    if QQ["KSP_processed"][patch_num] not in (1, 3):
        # processed but not added
        if QQ["KSP_processed"][patch_num] == 2 and master == 1:
            # loading instead of processing
            # load file as soon as save, if more than 10 sec, just do the recon instead.
            data_file = f'{ARG["filename"]}slice{patch_num}.pkl'
            # if file doesn't exist go to next slice
            if not os.path.isfile(data_file):
                # identified as bad file and being identified for reprocessing
                QQ["KSP_processed"][patch_num] = 0
                return (
                    KSP_recon,
                    KSP2,
                    total_patch_weights,
                    NOISE,
                    Component_threshold,
                    energy_removed,
                    SNR_weight,
                )
            else:
                with open(data_file, "rb") as f:
                    DATA_full2 = pickle.load(f)
                raise NotImplementedError("This block is never executed.")

        if QQ["KSP_processed"][patch_num] != 2:
            # block for other processes
            QQ["KSP_processed"][patch_num] = 1
            if DATA_full2 is None:
                if master == 0:
                    QQ["KSP_processed"][patch_num] = 1  # STARTING
                    # TODO: Check the index here
                    KSP2_x_patch = QQ["KSP2"][x_patch_idx, :, :, :]
                    lambda_ = QQ["ARG"]["LLR_scale"] * ARG["NVR_threshold"]
                    raise NotImplementedError("This block is never executed.")
                else:
                    QQ["KSP_processed"][patch_num] = 1  # STARTING
                    KSP2_x_patch = KSP2[x_patch_idx, :, :, :]
                    lambda_ = QQ["ARG"]["LLR_scale"] * ARG["NVR_threshold"]

                if ARG["patch_average"]:  # patch_average is always False
                    DATA_full2, total_patch_weights = subfunction_loop_for_NVR_avg(
                        KSP2_x_patch=KSP2_x_patch,
                        kernel_size_z=ARG["kernel_size"][2],
                        kernel_size_y=ARG["kernel_size"][1],
                        lambda2=lambda_,
                        soft_thrs=ARG["soft_thrs"],
                        total_patch_weights=total_patch_weights,
                        patch_average_sub=ARG["patch_average_sub"],
                    )
                    raise NotImplementedError("This block is never executed.")
                else:
                    x_patch_weights = total_patch_weights[x_patch_idx, :, :]
                    NOISE_x_patch = NOISE[x_patch_idx, :, :]
                    Component_threshold_x_patch = Component_threshold[x_patch_idx, :, :]
                    energy_removed_x_patch = energy_removed[x_patch_idx, :, :]
                    SNR_weight_x_patch = SNR_weight[x_patch_idx, :, :]

                    (
                        DATA_full2,
                        x_patch_weights,
                        NOISE_x_patch,
                        Component_threshold_x_patch,
                        energy_removed_x_patch,
                        SNR_weight_x_patch,
                    ) = subfunction_loop_for_NVR_avg_update(
                        KSP2_x_patch=KSP2_x_patch,
                        kernel_size_z=ARG["kernel_size"][2],
                        kernel_size_y=ARG["kernel_size"][1],
                        x_patch=patch_num,
                        lambda2=lambda_,
                        patch_avg=True,
                        soft_thrs=ARG["soft_thrs"],
                        total_patch_weights=x_patch_weights,
                        NOISE=NOISE_x_patch,
                        KSP2_tmp_update_threshold=Component_threshold_x_patch,
                        energy_removed=energy_removed_x_patch,
                        SNR_weight=SNR_weight_x_patch,
                        patch_scale=ARG.get("patch_scale", 1),
                        patch_average_sub=ARG["patch_average_sub"],
                    )

                    total_patch_weights[x_patch_idx, :, :] = x_patch_weights
                    NOISE[x_patch_idx, :, :] = NOISE_x_patch
                    Component_threshold[x_patch_idx, :, :] = Component_threshold_x_patch
                    energy_removed[x_patch_idx, :, :] = energy_removed_x_patch
                    SNR_weight[x_patch_idx, :, :] = SNR_weight_x_patch

        if master == 0:
            if QQ["KSP_processed"][patch_num] != 2:
                data_file = f'{ARG["filename"]}slice{patch_num}.pkl'
                with open(data_file, "wb") as f:
                    pickle.dump(DATA_full2, f)
                QQ["KSP_processed"][patch_num] = 2  # COMPLETED
            raise NotImplementedError("This block is never executed.")
        else:
            if ARG["patch_average"]:  # patch_average is always False
                KSP_recon[x_patch_idx, ...] += DATA_full2
                raise NotImplementedError("This block is never executed.")
            else:
                KSP_recon[x_patch_idx, : n_y, ...] += DATA_full2

            QQ["KSP_processed"][patch_num] = 3

    return (
        KSP_recon,
        KSP2,
        total_patch_weights,
        NOISE,
        Component_threshold,
        energy_removed,
        SNR_weight,
    )


def subfunction_loop_for_NVR_avg(
    KSP2_x_patch,
    kernel_size_z,
    kernel_size_y,
    lambda2,
    patch_avg=True,
    soft_thrs=None,
    total_patch_weights=None,
    patch_average_sub=None,
):
    """Do something.

    This is only called if ARG['patch_average'] is True, which it **never** is.

    Parameters
    ----------
    KSP2_x_patch : np.ndarray of shape (kernel_size_x, n_y, n_z, n_vols)
        An x patch of KSP2 data. Y, Z, and T are full length.
    """
    raise NotImplementedError("This block is never executed.")
    KSP2_tmp_update = np.zeros(KSP2_x_patch.shape)
    sigmasq_2 = None

    spacing = max(1, int(np.floor(kernel_size_y / patch_average_sub)))
    last = KSP2_x_patch.shape[1] - kernel_size_y + 1
    y_patches = list(np.arange(0, last, spacing, dtype=int))
    for y_patch in y_patches:
        y_patch_idx = np.arange(kernel_size_y, dtype=int) + y_patch
        spacing = max(1, int(np.floor(kernel_size_z / patch_average_sub)))
        last = KSP2_x_patch.shape[2] - kernel_size_z + 1
        z_patches = list(np.arange(0, last, spacing, dtype=int))
        for z_patch in z_patches:
            z_patch_idx = np.arange(kernel_size_z, dtype=int) + z_patch
            KSP2_patch = KSP2_x_patch[:, y_patch_idx, :, :]
            KSP2_patch = KSP2_patch[:, :, z_patch_idx, :]
            KSP2_patch_2d = np.reshape(KSP2_patch, (np.prod(KSP2_patch.shape[:3]), KSP2_patch.shape[3]))

            # svd(KSP2_patch_2d, 'econ') in MATLAB
            # S is 1D in Python, 2D diagonal matrix in MATLAB
            U, S, V = np.linalg.svd(KSP2_patch_2d, full_matrices=False)

            idx = np.sum(S < lambda2)
            if soft_thrs is None:
                S[S < lambda2] = 0
            elif soft_thrs == 10:  # Using MPPCA
                centering = 0
                n_voxels_in_patch = KSP2_patch_2d.shape[0]
                n_volumes = KSP2_patch_2d.shape[1]
                R = np.min((n_voxels_in_patch, n_volumes))
                scaling = (np.max((n_voxels_in_patch, n_volumes)) - np.arange(R - centering, dtype=int)) / n_volumes
                vals = S
                vals = (vals**2) / n_volumes

                # First estimation of Sigma^2;  Eq 1 from ISMRM presentation
                csum = np.cumsum(vals[::-1][:R - centering])
                cmean = (csum[::-1][:R - centering][:, None] / np.arange(1, R + 1 - centering)[::-1][None, :]).T
                sigmasq_1 = (cmean.T / scaling).T

                # Second estimation of Sigma^2; Eq 2 from ISMRM presentation
                gamma = (n_voxels_in_patch - np.arange(R - centering, dtype=int)) / n_volumes
                rangeMP = 4 * np.sqrt(gamma)
                rangeData = vals[: R - centering + 1] - vals[R - centering - 1]
                sigmasq_2 = (rangeData[:, None] / rangeMP[None, :]).T
                temp_idx = np.where(sigmasq_2 < sigmasq_1)
                t = np.vstack(temp_idx)[:, 0]  # first index where sigmasq_2 < sigmasq_1
                S[t:] = 0
            else:
                S[np.max((1, S.shape[0] - int(np.floor(idx * soft_thrs)))) :] = 0

            denoised_patch = np.dot(np.dot(U, np.diag(S)), V.T)
            denoised_patch = np.reshape(denoised_patch, KSP2_patch.shape)

            if patch_avg:
                # Use np.ix_ to create a broadcastable indexing array
                w2_slicex, w3_slicex = np.ix_(y_patch_idx, z_patch_idx)

                KSP2_tmp_update[:, w2_slicex, w3_slicex, :] += denoised_patch
                total_patch_weights[:, w2_slicex, w3_slicex] += 1
            else:
                w2_tmp = int(np.round(kernel_size_y / 2)) + (y_patch - 1)
                w3_tmp = int(np.round(kernel_size_z / 2)) + (z_patch - 1)
                KSP2_tmp_update[:, w2_tmp, w3_tmp, :] += denoised_patch[
                    0,
                    int(np.round(denoised_patch.shape[1] / 2)),
                    int(np.round(denoised_patch.shape[2] / 2)),
                    :,
                ]
                total_patch_weights[:, w2_tmp, w3_tmp] += 1

    return KSP2_tmp_update, total_patch_weights


def subfunction_loop_for_NVR_avg_update(
    KSP2_x_patch,
    x_patch,
    kernel_size_z,
    kernel_size_y,
    lambda2,
    total_patch_weights,
    NOISE,
    KSP2_tmp_update_threshold,
    energy_removed,
    SNR_weight,
    patch_avg=True,
    soft_thrs=1,
    patch_scale=1,
    patch_average_sub=None,
):
    """Do something.

    Parameters
    ----------
    KSP2_x_patch : np.ndarray of shape (kernel_size_x, n_y, n_z, n_vols)
        An x patch of KSP2 data. Y, Z, and T are full length.
    kernel_size_z : int
        Size of the kernel in the z-direction.
    kernel_size_y : int
        Size of the kernel in the y-direction.
    lambda2 : float
        Threshold for singular values.
    patch_avg : bool
        Hardcoded as True. Seems unrelated to ARG['patch_average'].
    soft_thrs : float
        Threshold for soft thresholding.
        None for NORDIC, 10 for g-factor estimation. Other values are supported, but unused.
    total_patch_weights : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Weighting array for KSP2_x_patch(?).
    NOISE : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Noise array for KSP2_x_patch(?).
    KSP2_tmp_update_threshold : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Thresholded update array for KSP2_x_patch(?).
    energy_removed : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Energy removed array for KSP2_x_patch(?).
    SNR_weight : np.ndarray of shape (kernel_size_x, n_y, n_z)
        SNR weighting array for KSP2_x_patch(?).
    patch_scale : float
        Scaling factor for the patch. Always set to 1.
    patch_average_sub : int
        Subsampling factor for patch averaging.
        Typically 2 for both NORDIC and g-factor estimation.

    Returns
    -------
    KSP2_tmp_update : np.ndarray of shape (kernel_size_x, n_y, n_z, n_vols)
        Updated KSP2 data.
    total_patch_weights : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Updated weighting array.
    NOISE : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Updated noise array.
    KSP2_tmp_update_threshold : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Updated thresholded update array.
    energy_removed : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Updated energy removed array.
    SNR_weight : np.ndarray of shape (kernel_size_x, n_y, n_z)
        Updated SNR weighting array.
    """
    total_patch_weights = total_patch_weights.copy()
    KSP2_tmp_update_threshold = KSP2_tmp_update_threshold.copy()
    energy_removed = energy_removed.copy()
    SNR_weight = SNR_weight.copy()
    KSP2_tmp_update = np.zeros_like(KSP2_x_patch)

    # Created in MATLAB version but not used
    # NOISE_x_patch = np.zeros(KSP2_x_patch.shape[:3])
    sigmasq_2 = None

    y_spacing = max(1, int(np.floor(kernel_size_y / patch_average_sub)))
    last_y = KSP2_x_patch.shape[1] - kernel_size_y + 1
    y_patches = list(np.arange(0, last_y, y_spacing, dtype=int))
    # Can introduce duplicate, but that's a bug in MATLAB
    y_patches.append(KSP2_x_patch.shape[1] - kernel_size_y)
    # y_patches = sorted(set(y_patches))

    z_spacing = max(1, int(np.floor(kernel_size_z / patch_average_sub)))
    last_z = KSP2_x_patch.shape[2] - kernel_size_z + 1
    z_patches = list(np.arange(0, last_z, z_spacing, dtype=int))
    # Can introduce duplicate, but that's a bug in MATLAB
    z_patches.append(KSP2_x_patch.shape[2] - kernel_size_z)
    # z_patches = sorted(set(z_patches))

    for y_patch in y_patches:
        y_patch_idx = np.arange(kernel_size_y, dtype=int) + y_patch

        for z_patch in z_patches:
            z_patch_idx = np.arange(kernel_size_z, dtype=int) + z_patch

            # Use np.ix_ to create a broadcastable indexing array
            w2_slicex, w3_slicex = np.ix_(y_patch_idx, z_patch_idx)

            KSP2_patch = KSP2_x_patch[:, w2_slicex, w3_slicex, :]
            # Reshape into Casorati matrix (X*Y*Z, T)
            KSP2_patch_2d = np.reshape(KSP2_patch, (np.prod(KSP2_patch.shape[:3]), KSP2_patch.shape[3]))

            U, S, V = np.linalg.svd(KSP2_patch_2d, full_matrices=False)

            n_removed_components = np.sum(S < lambda2)
            if soft_thrs is None:  # NORDIC
                # MATLAB code used .\, which seems to be switched element-wise division
                # MATLAB: 5 .\ 2 = 2 ./ 5
                energy_scrub = np.sqrt(np.sum(S[S < lambda2])) / np.sqrt(np.sum(S))
                S[S < lambda2] = 0
                # This is number of zero elements in array, not index of last non-zero element
                # BUG???
                first_removed_component = n_removed_components
                # Lots of S arrays that are *just* zeros
            elif soft_thrs != 10:
                S = S - lambda2 * soft_thrs
                S[S < 0] = 0
                energy_scrub = 0
                first_removed_component = 0
                raise NotImplementedError("This block is never executed.")
            elif soft_thrs == 10:  # USING MPPCA (gfactor estimation)
                voxelwise_sums = np.sum(KSP2_patch_2d, axis=1)
                n_zero_voxels_in_patch = np.sum(voxelwise_sums == 0)
                centering = 0
                # Correction for some zero entries
                n_nonzero_voxels_in_patch = KSP2_patch_2d.shape[0] - n_zero_voxels_in_patch
                if n_nonzero_voxels_in_patch > 0:
                    n_volumes = KSP2_patch_2d.shape[1]
                    R = np.min((n_nonzero_voxels_in_patch, n_volumes))
                    scaling = (max(n_nonzero_voxels_in_patch, n_volumes) - np.arange(R - centering, dtype=int)) / n_volumes
                    scaling = scaling.flatten()
                    vals = (S**2) / n_volumes

                    # First estimation of Sigma^2;  Eq 1 from ISMRM presentation
                    csum = np.cumsum(vals[::-1][:R - centering])
                    cmean = (csum[::-1][:R - centering] / np.arange(1, R + 1 - centering)[::-1])
                    sigmasq_1 = cmean / scaling

                    # Second estimation of Sigma^2; Eq 2 from ISMRM presentation
                    gamma = (n_nonzero_voxels_in_patch - np.arange(R - centering, dtype=int)) / n_volumes
                    rangeMP = 4 * np.sqrt(gamma)
                    rangeData = vals[: R - centering + 1] - vals[R - centering - 1]
                    sigmasq_2 = rangeData / rangeMP
                    first_removed_component = np.where(sigmasq_2 < sigmasq_1)[0][0]
                    n_removed_components = S.size - first_removed_component

                    # MATLAB code used .\, which seems to be switched element-wise division
                    # MATLAB: 5 .\ 2 = 2 ./ 5
                    energy_scrub = np.sqrt(np.sum(S[first_removed_component:])) / np.sqrt(np.sum(S))

                    # first_removed_component is 2D index (a, b), but S is a 1D array! (13, 13) vs (13,)
                    # And yet somehow the max idx seems to always be n_volumes,
                    # so maybe it's okay.
                    S[first_removed_component:] = 0
                else:  # all zero entries
                    first_removed_component = 0
                    energy_scrub = 0
                    sigmasq_2 = None

            else:  # SHOULD BE UNREACHABLE
                S[np.max((1, S.shape[0] - int(np.floor(n_removed_components * soft_thrs)))) :] = 0
                raise NotImplementedError("This block is never executed.")

            # Based on numpy svd documentation. Don't do np.dot(np.dot(U, np.diag(S)), V.T)!
            denoised_patch = np.dot(U * S, V)
            denoised_patch = np.reshape(denoised_patch, KSP2_patch.shape)

            if patch_scale != 1:
                patch_scale = S.shape[0] - n_removed_components
                raise NotImplementedError("This block is never executed.")

            if first_removed_component is None:
                # XXX: SHOULD BE UNREACHABLE
                first_removed_component = 0
                raise NotImplementedError("This block is never executed.")

            if patch_avg:
                # Update the entire patch
                KSP2_tmp_update[:, w2_slicex, w3_slicex, :] = (
                    KSP2_tmp_update[:, w2_slicex, w3_slicex, :] + (patch_scale * denoised_patch)
                )
                # total scaling factor across patches affecting a given voxel
                total_patch_weights[:, w2_slicex, w3_slicex] += patch_scale
                # number of singular values *removed*
                KSP2_tmp_update_threshold[:, w2_slicex, w3_slicex] += n_removed_components
                energy_removed[:, w2_slicex, w3_slicex] += energy_scrub

                # Was
                SNR_weight[:, w2_slicex, w3_slicex] += S[0] / S[max(0, first_removed_component - 2)]
                # but was getting divide-by-zero warnings
                # The issue is that first_removed_component is the index of the last non-zero element in S
                # before values > lambda2 are zeroed out.
                # So first_removed_component might end up indexing a zero element when soft_thrs is None
                # if S[0] != 0:
                #     with warnings.catch_warnings():
                #         warnings.filterwarnings("error")
                #         try:
                #             SNR_weight[:, w2_slicex, w3_slicex] += S[0] / S[max(0, first_removed_component - 2)]
                #         except RuntimeWarning:
                #             raise Exception(
                #                 f"S_upda: {S}\n"
                #                 f"S_orig: {S_orig}\n"
                #                 f"first_removed_component: {first_removed_component}\n"
                #             )

                if sigmasq_2 is not None:
                    x_patch_idx = np.arange(KSP2_x_patch.shape[0])
                    w1_slicex, w2_slicex, w3_slicex = np.ix_(x_patch_idx, y_patch_idx, z_patch_idx)
                    NOISE[w1_slicex, w2_slicex, w3_slicex] += sigmasq_2.flatten()[first_removed_component]

            else:
                # Only update a single voxel in the middle of the patch
                w2_tmp = int(np.round(kernel_size_y / 2)) + y_patch
                w3_tmp = int(np.round(kernel_size_z / 2)) + z_patch

                KSP2_tmp_update[:, w2_tmp, w3_tmp, :] += (
                    patch_scale
                    * denoised_patch[
                        0,
                        int(np.round(denoised_patch.shape[1] / 2)),
                        int(np.round(denoised_patch.shape[2] / 2)),
                        :,
                    ]
                )
                total_patch_weights[:, w2_tmp, w3_tmp] += patch_scale
                KSP2_tmp_update_threshold[:, w2_tmp, w3_tmp, :] += n_removed_components
                energy_removed[:, w2_tmp, w3_tmp] += energy_scrub
                SNR_weight[:, w2_tmp, w3_tmp] += S[0] / S[max(0, first_removed_component - 2)]
                if sigmasq_2 is not None:  # sigmasq_2 is only defined when soft_thrs == 10
                    NOISE[:, w2_tmp, w3_tmp] += sigmasq_2[first_removed_component, first_removed_component]
                raise NotImplementedError("This block is never executed.")

    return (
        KSP2_tmp_update,
        total_patch_weights,
        NOISE,
        KSP2_tmp_update_threshold,
        energy_removed,
        SNR_weight,
    )

"""Another Python attempt at NORDIC."""

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
    kernel_size_gfactor=[],
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
        Correction for slice and time-specific phase
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
    mag_data = np.abs(mag_img.get_fdata()).astype(np.float32)

    if pha_file:
        has_complex = True
        pha_img = nb.load(pha_file)
        pha_data = pha_img.get_fdata().astype(np.float32)

        # Scale the phase data to -pi to pi
        phase_range = np.max(pha_data)
        phase_range_min = np.min(pha_data)
        range_norm = phase_range - phase_range_min
        range_center = (phase_range + phase_range_min) / range_norm * 1 / 2
        pha_data = (pha_data / range_norm - range_center) * 2 * np.pi

    n_noise_volumes = 0
    if mag_norf_file:
        mag_norf_img = nb.load(mag_norf_file)
        mag_norf_data = mag_norf_img.get_fdata().astype(np.float32)
        n_noise_volumes = mag_norf_data.shape[3]
        mag_data = np.concatenate((mag_data, mag_norf_data), axis=3)

        if has_complex:
            pha_norf_img = nb.load(pha_norf_file)
            pha_norf_data = pha_norf_img.get_fdata().astype(np.float32)
            pha_data = np.concatenate((pha_data, pha_norf_data), axis=3)

    # Combine magnitude and phase into complex-valued data
    complex_data = mag_data * np.exp(1j * pha_data)

    # Select the first volume of the complex data and take the absolute values
    TEMPVOL = np.abs(complex_data[..., 0])

    # Find the minimum non-zero value in the first volume and divide the complex data by it
    ARG["ABSOLUTE_SCALE"] = np.min(TEMPVOL[TEMPVOL != 0])
    complex_data = complex_data / ARG["ABSOLUTE_SCALE"]

    if complex_data.shape[3] < 6:
        raise ValueError("Two few volumes in the input data")

    # Create a copy of the complex data for processing
    KSP2 = complex_data.copy()
    n_x, n_y, n_slices, n_vols = KSP2.shape

    # Create mean 3D array from all non-noise volumes of shape (X, Y, Z)
    meanphase = np.mean(complex_data, axis=3)
    # Multiply the mean array by either 1 or 0 (default is 0)
    meanphase = meanphase * ARG["phase_slice_average_for_kspace_centering"]

    # Preallocate 4D array of zeros
    DD_phase = np.zeros_like(complex_data)

    # If the temporal phase is 1 - 3, do a standard low-pass filtered map
    if ARG["temporal_phase"] > 0:
        # Loop over slices backwards
        for i_slice in range(n_slices)[::-1]:
            # Loop over volumes forward, including the noise volumes(???)
            for j_vol in range(n_vols):
                # Grab the 2D slice of the 4D array
                tmp = complex_data[:, :, i_slice, j_vol]

                # Apply 1D FFT to the 2D slice
                for k_dim in range(2):
                    tmp = np.fft.ifftshift(
                        np.fft.ifft(
                            np.fft.ifftshift(tmp, axes=[k_dim]),
                            n=None,
                            axis=k_dim,
                        ),
                        axes=[k_dim],
                    )

                # Apply Tukey window to the filtered 2D slice
                # TODO: Verify that this works.
                # tmp = bsxfun(@times,tmp,reshape(tukeywin(n_y,1).^phase_filter_width,[1 n_y]));
                tukey_window = np.outer(
                    tukey(n_y, 1) ** ARG["phase_filter_width"],
                    tukey(n_y, 1) ** ARG["phase_filter_width"],
                )
                tmp *= tukey_window
                # tmp = bsxfun(@times,tmp,reshape(tukeywin(n_x,1).^phase_filter_width,[n_x 1]));
                tukey_window = np.outer(
                    tukey(n_x, 1) ** ARG["phase_filter_width"],
                    tukey(n_x, 1) ** ARG["phase_filter_width"],
                )
                tmp *= tukey_window

                # Apply 1D IFFT to the filtered 2D slice and store in the 4D array
                for k_dim in range(2):
                    tmp = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.fftshift(tmp, axes=[k_dim]),
                            n=None,
                            axis=k_dim,
                        ),
                        axes=[k_dim],
                    )

                DD_phase[:, :, i_slice, j_vol] = tmp

    # Multiply the 4D array by the exponential of the angle of the filtered phase
    KSP2 = KSP2 * np.exp(-1j * np.angle(DD_phase))

    print("Completed estimating slice-dependent phases ...")
    if not ARG["kernel_size_gfactor"]:
        # Select first 90 (or fewer, if run is shorter) volumes from 4D array
        KSP2 = KSP2[:, :, :, : min(90, n_vols + 1)]
    else:
        # Select first N volumes from 4D array, based on kernel_size_gfactor(4)
        KSP2 = KSP2[:, :, :, : min(ARG["kernel_size_gfactor"][3], n_vols + 1)]

    # Replace NaNs and Infs with zeros
    KSP2[np.isnan(KSP2)] = 0
    KSP2[np.isinf(KSP2)] = 0

    # Preallocate 4D array of zeros
    KSP_recon = np.zeros_like(KSP2)

    if not ARG["kernel_size_gfactor"]:
        ARG["kernel_size"] = [14, 14, 1]
    else:
        ARG["kernel_size"] = [int(i) for i in ARG["kernel_size_gfactor"][:3]]

    n_patches = n_x - ARG["kernel_size"][0]
    QQ["KSP_processed"] = np.zeros(n_patches, dtype=int)
    ARG["patch_average"] = False
    ARG["patch_average_sub"] = ARG["gfactor_patch_overlap"]
    ARG["LLR_scale"] = 0
    ARG["NVR_threshold"] = 1
    ARG["soft_thrs"] = 10  # MPPCa   (When Noise varies)
    # ARG['soft_thrs'] = []  # NORDIC (When noise is flat)

    # Preallocate 3D arrays of zeros
    KSP_weight = np.zeros((n_x, n_y, n_slices))
    NOISE = np.zeros_like(KSP_weight)
    Component_threshold = np.zeros_like(KSP_weight)
    energy_removed = np.zeros_like(KSP_weight)
    SNR_weight = np.zeros_like(KSP_weight)
    # The original code re-creates KSP_processed here for no reason

    # patch_average is hardcoded as False so this block is always executed.
    if not ARG["patch_average"]:
        KSP_processed = QQ["KSP_processed"] * 0  # pointless
        # WTAF is this loop doing?
        val = int(max(1, int(np.floor(ARG["kernel_size"][0] / ARG["patch_average_sub"]))))
        for nw1 in range(1, val):
            # KSP_processed(1,nw1 : max(1,floor(ARG.ARG['kernel_size'](1)/ARG.patch_average_sub)):end)=2;
            KSP_processed[nw1:val:] = 2
        KSP_processed[-1] = 0
        QQ["KSP_processed"] = KSP_processed

    print("Estimating g-factor")
    QQ["ARG"] = ARG
    master_fast = 1
    for i_patch in range(n_patches):
        (
            KSP_recon,
            _,
            KSP_weight,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        ) = sub_LLR_Processing(
            KSP_recon=KSP_recon,
            KSP2=KSP2,
            ARG=ARG,
            patch_num=i_patch,
            QQ=QQ,
            master=master_fast,
            KSP2_weight=KSP_weight,
            NOISE=NOISE,
            Component_threshold=Component_threshold,
            energy_removed=energy_removed,
            SNR_weight=SNR_weight,
        )

    KSP_recon = KSP_recon / KSP_weight[..., None]
    ARG["NOISE"] = np.sqrt(NOISE / KSP_weight)
    ARG["Component_threshold"] = Component_threshold / KSP_weight
    ARG["energy_removed"] = energy_removed / KSP_weight
    ARG["SNR_weight"] = SNR_weight / KSP_weight
    IMG2 = KSP_recon
    # ARG2 = ARG
    print("completed estimating g-factor")
    gfactor = ARG["NOISE"]

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
        gfactor_for_img = np.abs(
            gfactor
        )  # Absolute value isn't used for correction, so why is the output map absolute?
        gfactor_for_img[np.isnan(gfactor_for_img)] = 0

        tmp = np.sort(np.abs(gfactor_for_img.flatten()))
        sn_scale = (
            2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        )  # added -1 to match MATLAB indexing
        gain_level = np.floor(np.log2(32000 / sn_scale))
        if not full_dynamic_range:
            gain_level = 0

        gfactor_for_img = np.abs(gfactor_for_img) * (2**gain_level)
        gfactor_img = nb.Nifti1Image(gfactor_for_img, mag_img.affine, mag_img.header)
        gfactor_img.to_filename(out_dir / "gfactor.nii.gz")

    KSP2 = complex_data.copy()
    n_x, n_y, n_slices, n_vols = KSP2.shape

    KSP2 = KSP2 * np.exp(-1j * np.angle(meanphase[..., None]))
    KSP2 = KSP2 / gfactor[..., None]

    if n_noise_volumes > 0:
        KSP2_NOISE = KSP2[..., -n_noise_volumes:]

    if ARG["temporal_phase"] == 3:
        # Secondary step for filtered phase with residual spikes
        for i_slice in range(n_slices)[::-1]:
            for j_vol in range(n_vols):
                phase_diff = np.angle(
                    KSP2[:, :, i_slice, j_vol] / DD_phase[:, :, i_slice, j_vol]
                )
                mask = np.abs(phase_diff) > 1
                mask2 = np.abs(KSP2[:, :, i_slice, j_vol]) > np.sqrt(2)
                DD_phase2 = DD_phase[:, :, i_slice, j_vol]
                tmp = KSP2[:, :, i_slice, j_vol]
                DD_phase2[mask * mask2] = tmp[mask * mask2]
                DD_phase[:, :, i_slice, j_vol] = DD_phase2

    KSP2 = KSP2 * np.exp(-1j * np.angle(DD_phase))
    KSP2[np.isnan(KSP2)] = 0
    KSP2[np.isinf(KSP2)] = 0

    ARG["measured_noise"] = 1
    if n_noise_volumes > 0:
        tmp_noise = KSP2_NOISE.copy()
        tmp_noise[np.isnan(tmp_noise)] = 0
        tmp_noise[np.isinf(tmp_noise)] = 0
        ARG["measured_noise"] = np.std(
            tmp_noise[tmp_noise != 0]
        )  # sqrt(2) for real and complex

    if has_complex:
        ARG["measured_noise"] = ARG["measured_noise"] / np.sqrt(2)

    if ARG["data_has_zero_elements"]:
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
        ARG["kernel_size"] = [int(i) for i in ARG["kernel_size_PCA"]]
        raise Exception(f'{ARG["kernel_size_PCA"]}\n\n{ARG["kernel_size"]}')

    if n_slices <= ARG["kernel_size"][2]:  # Number of slices is less than cubic kernel
        ARG["kernel_size"] = (
            np.ones(3, dtype=int) * int(np.round(np.sqrt(n_vols * 11 / n_slices)))
        )
        ARG["kernel_size"][2] = n_slices

    ARG["patch_average"] = False
    ARG["patch_average_sub"] = ARG["NORDIC_patch_overlap"]
    ARG["LLR_scale"] = 1
    ARG["NVR_threshold"] = 0
    ARG["soft_thrs"] = None  # NORDIC (When noise is flat)
    ARG["NVR_threshold"] = 0  # for no reason

    for _ in range(10):
        _, S, _ = np.linalg.svd(
            np.random.normal(size=(np.prod(ARG["kernel_size"]), n_vols))
        )
        ARG["NVR_threshold"] = ARG["NVR_threshold"] + S[0]

    # XXX: This divides by 10, but multiplies by the others. Is that expected?
    if has_complex:
        # sqrt(2) due to complex  1.20 due to understimate of g-factor
        ARG["NVR_threshold"] = (
            ARG["NVR_threshold"]
            / 10
            * np.sqrt(2)
            * ARG["measured_noise"]
            * ARG["factor_error"]
        )
    else:
        ARG["NVR_threshold"] = (
            ARG["NVR_threshold"] / 10 * ARG["measured_noise"] * ARG["factor_error"]
        )

    if algorithm in ["mppca", "mppca+nordic"]:
        ARG["soft_thrs"] = 10

    KSP_weight = np.zeros_like(KSP2[..., 0])
    NOISE = KSP_weight.copy()
    Component_threshold = KSP_weight.copy()
    energy_removed = KSP_weight.copy()
    SNR_weight = KSP_weight.copy()

    n_patches = n_x - ARG["kernel_size"][0]
    QQ["KSP_processed"] = np.zeros(n_patches, dtype=int)

    if not ARG["patch_average"]:
        KSP_processed = QQ["KSP_processed"] * 0
        val = max(1, int(np.floor(ARG["kernel_size"][0] / ARG["patch_average_sub"])))
        for nw1 in range(1, val):
            KSP_processed[nw1:val:] = 2
        KSP_processed[-1] = 0
        QQ["KSP_processed"] = KSP_processed

    print("Starting NORDIC")
    QQ["ARG"] = ARG
    master_fast = 1
    for i_patch in range(n_patches):
        (
            KSP_recon,
            _,
            KSP_weight,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        ) = sub_LLR_Processing(
            KSP_recon=KSP_recon,
            KSP2=KSP2,
            ARG=ARG,
            patch_num=i_patch,
            QQ=QQ,
            master=master_fast,
            KSP2_weight=KSP_weight,
            NOISE=NOISE,
            Component_threshold=Component_threshold,
            energy_removed=energy_removed,
            SNR_weight=SNR_weight,
        )

    # Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
    KSP_recon = KSP_recon / KSP_weight[..., None]
    ARG["NOISE"] = np.sqrt(NOISE / KSP_weight)
    ARG["Component_threshold"] = Component_threshold / KSP_weight
    ARG["energy_removed"] = energy_removed / KSP_weight
    ARG["SNR_weight"] = SNR_weight / KSP_weight
    IMG2 = KSP_recon
    print("Completing NORDIC")

    residual = KSP2 - KSP_recon
    # Split residuals into magnitude and phase
    residual_magn = np.abs(residual)
    residual_phase = np.angle(residual)
    residual_magn_img = nb.Nifti1Image(residual_magn, mag_img.affine, mag_img.header)
    residual_magn_img.to_filename(out_dir / "residual_magn.nii.gz")
    residual_phase_img = nb.Nifti1Image(residual_phase, mag_img.affine, mag_img.header)
    residual_phase_img.to_filename(out_dir / "residual_phase.nii.gz")
    del residual, residual_magn, residual_phase
    del residual_magn_img, residual_phase_img

    IMG2 = IMG2 * gfactor[:, :, :, None]
    IMG2 *= np.exp(1j * np.angle(DD_phase))
    IMG2 = IMG2 * ARG["ABSOLUTE_SCALE"]
    IMG2[np.isnan(IMG2)] = 0

    if has_complex:
        IMG2_tmp = np.abs(IMG2)  # remove g-factor and noise for DUAL 1
        IMG2_tmp[np.isnan(IMG2_tmp)] = 0
        tmp = np.sort(np.abs(IMG2_tmp.flatten()))
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        gain_level = np.floor(np.log2(32000 / sn_scale))

        if not full_dynamic_range:
            gain_level = 0

        IMG2_tmp = np.abs(IMG2_tmp) * (2**gain_level)
        IMG2_img = nb.Nifti1Image(IMG2_tmp, mag_img.affine, mag_img.header)
        IMG2_img.to_filename(out_dir / "magn.nii.gz")

        IMG2_tmp = np.angle(IMG2)
        IMG2_tmp = (IMG2_tmp / (2 * np.pi) + range_center) * range_norm
        IMG2_img = nb.Nifti1Image(IMG2_tmp, mag_img.affine, mag_img.header)
        IMG2_img.to_filename(out_dir / "phase.nii.gz")
    else:
        IMG2 = np.abs(IMG2)
        IMG2[np.isnan(IMG2)] = 0
        tmp = np.sort(np.abs(IMG2).flatten())
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        gain_level = np.floor(np.log2(32000 / sn_scale))

        if not full_dynamic_range:
            gain_level = 0

        IMG2 = np.abs(IMG2) * (2**gain_level)
        IMG2_img = nb.Nifti1Image(IMG2, mag_img.affine, mag_img.header)
        IMG2_img.to_filename(out_dir / "magn.nii.gz")

    print("Done!")


def sub_LLR_Processing(
    KSP_recon,
    KSP2,
    ARG,
    patch_num,
    QQ,
    master,
    KSP2_weight,
    NOISE=None,
    Component_threshold=None,
    energy_removed=None,
    SNR_weight=None,
):
    """Do something.

    Parameters
    ----------
    KSP_recon : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    KSP2 : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    ARG : dict
        A dictionary of arguments. I will break this up into individual variables later.
    patch_num : int
        Patch number. Each patch is processed separately.
    QQ : dict
        A dictionary of arguments. I will break this up into individual variables later.
    master : int
    KSP2_weight : np.ndarray of shape (n_x, n_y, n_slices)
    NOISE : np.ndarray of shape (n_x, n_y, n_slices)
    Component_threshold : np.ndarray of shape (n_x, n_y, n_slices)
    energy_removed : np.ndarray of shape (n_x, n_y, n_slices)
    SNR_weight : np.ndarray of shape (n_x, n_y, n_slices)

    Returns
    -------
    KSP_recon : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    KSP2 : np.ndarray of shape (n_x, n_y, n_slices, n_vols)
    KSP2_weight : np.ndarray of shape (n_x, n_y, n_slices)
    NOISE : np.ndarray of shape (n_x, n_y, n_slices)
    Component_threshold : np.ndarray of shape (n_x, n_y, n_slices)
    energy_removed : np.ndarray of shape (n_x, n_y, n_slices)
    SNR_weight : np.ndarray of shape (n_x, n_y, n_slices)
    """
    import pickle
    import os

    _, n_y, _, _ = KSP2.shape
    patch_idx = np.arange(0, ARG["kernel_size"][0], dtype=int) + (patch_num - 1)

    # not being processed also not completed yet
    # DATA_full2 is (patch_size, n_y, n_z, n_vols)
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
            else:
                with open(data_file, "rb") as f:
                    DATA_full2 = pickle.load(f)

        if QQ["KSP_processed"][patch_num] != 2:
            # block for other processes
            QQ["KSP_processed"][patch_num] = 1
            if DATA_full2 is None:
                ARG2 = QQ["ARG"]
                if master == 0:
                    QQ["KSP_processed"][patch_num] = 1  # STARTING
                    # TODO: Check the index here
                    KSP2a = QQ["KSP2"][patch_idx, :, :, :]
                    lambda_ = ARG2["LLR_scale"] * ARG["NVR_threshold"]
                else:
                    QQ["KSP_processed"][patch_num] = 1  # STARTING
                    KSP2a = KSP2[patch_idx, :, :, :]
                    lambda_ = ARG2["LLR_scale"] * ARG["NVR_threshold"]

                if ARG["patch_average"]:
                    DATA_full2, KSP2_weight = subfunction_loop_for_NVR_avg(
                        KSP2a=KSP2a,
                        w3=ARG["kernel_size"][2],
                        w2=ARG["kernel_size"][1],
                        lambda2=lambda_,
                        soft_thrs=ARG["soft_thrs"],
                        KSP2_weight=KSP2_weight,
                    )
                else:
                    KSP2_weight_tmp = KSP2_weight[patch_idx, :, :]
                    NOISE_tmp = NOISE[patch_idx, :, :]
                    Component_threshold_tmp = Component_threshold[patch_idx, :, :]
                    energy_removed_tmp = energy_removed[patch_idx, :, :]
                    SNR_weight_tmp = SNR_weight[patch_idx, :, :]

                    (
                        DATA_full2,
                        KSP2_weight_tmp,
                        NOISE_tmp,
                        Component_threshold_tmp,
                        energy_removed_tmp,
                        SNR_weight_tmp,
                    ) = subfunction_loop_for_NVR_avg_update(
                        KSP2a=KSP2a,
                        w3=ARG["kernel_size"][2],
                        w2=ARG["kernel_size"][1],
                        lambda2=lambda_,
                        patch_avg=True,
                        soft_thrs=ARG["soft_thrs"],
                        KSP2_weight=KSP2_weight_tmp,
                        ARG=ARG,
                        NOISE=NOISE_tmp,
                        KSP2_tmp_update_threshold=Component_threshold_tmp,
                        energy_removed=energy_removed_tmp,
                        SNR_weight=SNR_weight_tmp,
                    )

                    KSP2_weight[patch_idx, :, :] = KSP2_weight_tmp
                    NOISE[patch_idx, :, :] = NOISE_tmp
                    Component_threshold[patch_idx, :, :] = Component_threshold_tmp
                    energy_removed[patch_idx, :, :] = energy_removed_tmp
                    SNR_weight[patch_idx, :, :] = SNR_weight_tmp

        if master == 0:
            if QQ["KSP_processed"][patch_num] != 2:
                data_file = f'{ARG["filename"]}slice{patch_num}.pkl'
                with open(data_file, "wb") as f:
                    pickle.dump(DATA_full2, f)
                QQ["KSP_processed"][patch_num] = 2  # COMPLETED
        else:
            if ARG["patch_average"]:
                KSP_recon[patch_idx, ...] += DATA_full2
            else:
                KSP_recon[patch_idx, : n_y, ...] += DATA_full2

            QQ["KSP_processed"][patch_num] = 3

    return (
        KSP_recon,
        KSP2,
        KSP2_weight,
        NOISE,
        Component_threshold,
        energy_removed,
        SNR_weight,
    )


def subfunction_loop_for_NVR_avg(
    KSP2a, w3, w2, lambda2, patch_avg=True, soft_thrs=None, KSP2_weight=None, ARG=None
):
    """Do something."""
    if KSP2_weight is None:
        KSP2_weight = np.zeros(KSP2a.shape[:3])

    KSP2_tmp_update = np.zeros(KSP2a.shape)
    sigmasq_2 = None

    spacing = max(1, int(np.floor(w2 / ARG["patch_average_sub"])))
    last = KSP2a.shape[1] - w2 + 1
    n2_range = list(np.arange(0, last, spacing, dtype=int))
    for n2 in n2_range:
        w2_slice = np.arange(w2, dtype=int) + n2
        spacing = max(1, int(np.floor(w3 / ARG["patch_average_sub"])))
        last = KSP2a.shape[2] - w3 + 1
        n3_range = list(np.arange(0, last, spacing, dtype=int))
        for n3 in n3_range:
            w3_slice = np.arange(w3, dtype=int) + n3
            KSP2_tmp = KSP2a[:, w2_slice, :, :]
            KSP2_tmp = KSP2_tmp[:, :, w3_slice, :]
            tmp1 = np.reshape(KSP2_tmp, (np.prod(KSP2_tmp.shape[:3]), KSP2_tmp.shape[3]))

            # svd(tmp1, 'econ') in MATLAB
            # S is 1D in Python, 2D diagonal matrix in MATLAB
            U, S, V = np.linalg.svd(tmp1, full_matrices=False)

            idx = np.sum(S < lambda2)
            if soft_thrs is None:
                S[S < lambda2] = 0
            elif soft_thrs == 10:  # Using MPPCA
                centering = 0
                MM = tmp1.shape[0]
                NNN = tmp1.shape[1]
                R = np.min((MM, NNN))
                scaling = (np.max((MM, NNN)) - np.arange(R - centering, dtype=int)) / NNN
                vals = S
                vals = (vals**2) / NNN

                # First estimation of Sigma^2;  Eq 1 from ISMRM presentation
                csum = np.cumsum(vals[::-1][:R - centering])
                cmean = csum[::-1][:R - centering] / np.arange(R - centering, dtype=int)[::-1]
                sigmasq_1 = cmean / scaling

                # Second estimation of Sigma^2; Eq 2 from ISMRM presentation
                gamma = (MM - np.arange(R - centering, dtype=int)) / NNN
                rangeMP = 4 * np.sqrt(gamma)
                rangeData = vals[: R - centering + 1] - vals[R - centering - 1]
                sigmasq_2 = rangeData / rangeMP
                t = np.where(sigmasq_2 < sigmasq_1)[0][0]
                S[t:] = 0
            else:
                S[np.max((1, S.shape[0] - int(np.floor(idx * soft_thrs)))) :] = 0

            tmp1 = np.dot(np.dot(U, np.diag(S)), V.T)
            tmp1 = np.reshape(tmp1, KSP2_tmp.shape)

            if patch_avg:
                # Use np.ix_ to create a broadcastable indexing array
                w2_slicex, w3_slicex = np.ix_(w2_slice, w3_slice)

                KSP2_tmp_update[:, w2_slicex, w3_slicex, :] += tmp1
                KSP2_weight[:, w2_slicex, w3_slicex] += 1
            else:
                w2_tmp = int(np.round(w2 / 2)) + (n2 - 1)
                w3_tmp = int(np.round(w3 / 2)) + (n3 - 1)
                KSP2_tmp_update[:, w2_tmp, w3_tmp, :] += tmp1[
                    0,
                    int(np.round(tmp1.shape[1] / 2)),
                    int(np.round(tmp1.shape[2] / 2)),
                    :,
                ]
                KSP2_weight[:, w2_tmp, w3_tmp] += 1

    return KSP2_tmp_update, KSP2_weight


def subfunction_loop_for_NVR_avg_update(
    KSP2a,
    w3,
    w2,
    lambda2,
    patch_avg=True,
    soft_thrs=1,
    KSP2_weight=None,
    ARG=None,
    NOISE=None,
    KSP2_tmp_update_threshold=None,
    energy_removed=None,
    SNR_weight=None,
):
    """Do something."""
    patch_scale = ARG.get("patch_scale", 1)
    if KSP2_weight is None:
        KSP2_weight = np.zeros(KSP2a.shape[:3])

    # Created in MATLAB version but not used
    # NOISE_tmp = np.zeros(KSP2a.shape[:3])
    sigmasq_2 = None

    if KSP2_tmp_update_threshold is None:
        KSP2_tmp_update_threshold = np.zeros(KSP2a.shape[:3], dtype=KSP2a.dtype)

    if energy_removed is None:
        energy_removed = np.zeros(KSP2a.shape[:3])

    if SNR_weight is None:
        SNR_weight = np.zeros(KSP2a.shape[:3])

    KSP2_tmp_update = np.zeros_like(KSP2a)

    spacing = max(1, int(np.floor(w2 / ARG["patch_average_sub"])))
    last = KSP2a.shape[1] - w2 + 1
    n2_range = list(np.arange(0, last, spacing, dtype=int))
    for n2 in n2_range:
        w2_slice = np.arange(w2, dtype=int) + n2
        spacing = max(1, int(np.floor(w3 / ARG["patch_average_sub"])))
        last = KSP2a.shape[2] - w3 + 1
        n3_range = list(np.arange(0, last, spacing, dtype=int))
        for n3 in n3_range:
            w3_slice = np.arange(w3, dtype=int) + n3
            KSP2_tmp = KSP2a[:, w2_slice, :, :]
            KSP2_tmp = KSP2_tmp[:, :, w3_slice, :]
            tmp1 = np.reshape(KSP2_tmp, (np.prod(KSP2_tmp.shape[:3]), KSP2_tmp.shape[3]))

            U, S, V = np.linalg.svd(tmp1, full_matrices=False)

            idx = np.sum(S < lambda2)
            if soft_thrs is None:
                energy_scrub = np.sqrt(np.sum(S)) / np.sqrt(np.sum(S[S < lambda2]))
                S[S < lambda2] = 0
                t = idx
            elif soft_thrs != 10:
                S = S - lambda2 * soft_thrs
                S[S < 0] = 0
                energy_scrub = 0
                t = 1
            elif soft_thrs == 10:  # USING MPPCA
                Test_mat = np.sum(tmp1, axis=1)
                MM0 = np.sum(Test_mat == 0)
                centering = 0
                MM = tmp1.shape[0] - MM0  # Correction for some zero entries
                if MM > 0:
                    NNN = tmp1.shape[1]
                    R = np.min((MM, NNN))
                    scaling = (np.max((MM, NNN)) - np.arange(R - centering, dtype=int)) / NNN
                    scaling = scaling.flatten()
                    vals = S.copy()
                    vals = (vals**2) / NNN
                    # First estimation of Sigma^2;  Eq 1 from ISMRM presentation
                    csum = np.cumsum(vals[::-1][:R - centering])
                    cmean = csum[::-1][:R - centering] / np.arange(R - centering, dtype=int)[::-1]
                    sigmasq_1 = cmean / scaling
                    # Second estimation of Sigma^2; Eq 2 from ISMRM presentation
                    gamma = (MM - np.arange(R - centering, dtype=int)) / NNN
                    rangeMP = 4 * np.sqrt(gamma)
                    rangeData = vals[: R - centering + 1] - vals[R - centering - 1]
                    sigmasq_2 = rangeData / rangeMP
                    t = np.where(sigmasq_2 < sigmasq_1)[0][0]
                    idx = S[t:].shape[0]
                    energy_scrub = np.sqrt(np.sum(S)) / np.sqrt(np.sum(S[t:]))
                    S[t:] = 0
                else:  # all zero entries
                    t = 1
                    energy_scrub = 0
                    sigmasq_2 = 0

            else:  # SHOULD BE UNREACHABLE
                S[np.max((1, S.shape[0] - int(np.floor(idx * soft_thrs)))) :] = 0

            tmp1 = np.dot(np.dot(U, np.diag(S)), V.T)
            tmp1 = np.reshape(tmp1, KSP2_tmp.shape)

            if patch_scale != 1:
                patch_scale = S.shape[0] - idx

            if t is None:
                t = 1

            if patch_avg:
                # Use np.ix_ to create a broadcastable indexing array
                w2_slicex, w3_slicex = np.ix_(w2_slice, w3_slice)
                KSP2_tmp_update[:, w2_slicex, w3_slicex, :] = (
                    KSP2_tmp_update[:, w2_slicex, w3_slicex, :] + (patch_scale * tmp1)
                )
                KSP2_weight[:, w2_slicex, w3_slicex] += patch_scale
                KSP2_tmp_update_threshold[:, w2_slicex, w3_slicex] += idx
                energy_removed[:, w2_slicex, w3_slicex] += energy_scrub
                SNR_weight[:, w2_slicex, w3_slicex] += S[0] / S[max(1, t - 1)]
                if sigmasq_2 is not None:
                    NOISE[: KSP2a.shape[0], w2_slicex, w3_slicex] += sigmasq_2[t]

            else:
                w2_tmp = int(np.round(w2 / 2)) + n2
                w3_tmp = int(np.round(w3 / 2)) + n3

                KSP2_tmp_update[:, w2_tmp, w3_tmp, :] += (
                    patch_scale
                    * tmp1[
                        0,
                        int(np.round(tmp1.shape[1] / 2)),
                        int(np.round(tmp1.shape[2] / 2)),
                        :,
                    ]
                )
                KSP2_weight[:, w2_tmp, w3_tmp] += patch_scale
                KSP2_tmp_update_threshold[:, w2_tmp, w3_tmp, :] += idx
                energy_removed[:, w2_tmp, w3_tmp] += energy_scrub
                SNR_weight[:, w2_tmp, w3_tmp] += S[0] / S[max(1, t - 1)]
                if sigmasq_2 is not None:
                    NOISE[:, w2_tmp, w3_tmp] += sigmasq_2[t]

    return (
        KSP2_tmp_update,
        KSP2_weight,
        NOISE,
        KSP2_tmp_update_threshold,
        energy_removed,
        SNR_weight,
    )

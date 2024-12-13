"""Another Python attempt at NORDIC."""

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
    algorithm='nordic',
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
    ARG = {
        'factor_error': factor_error,
        'full_dynamic_range': full_dynamic_range,
        'temporal_phase': temporal_phase,
        'algorithm': algorithm,
        'kernel_size_gfactor': kernel_size_gfactor,
        'kernel_size_PCA': kernel_size_PCA,
        'phase_slice_average_for_kspace_centering': phase_slice_average_for_kspace_centering,
        'phase_filter_width': phase_filter_width,
        'NORDIC_patch_overlap': NORDIC_patch_overlap,
        'gfactor_patch_overlap': gfactor_patch_overlap,
        'save_gfactor_map': save_gfactor_map,
        'out_dir': out_dir,
    }
    QQ = {}

    mag_img = nb.load(mag_file)
    mag_data = np.abs(mag_img.get_fdata())

    if pha_file:
        has_complex = True
        pha_img = nb.load(pha_file)
        pha_data = pha_img.get_fdata()

        # Scale the phase data to -pi to pi
        phase_range = np.max(pha_data)
        phase_range_min = np.min(pha_data)
        range_norm = phase_range - phase_range_min
        range_center = (phase_range + phase_range_min) / range_norm * 1 / 2
        pha_data = (pha_data / range_norm - range_center) * 2 * np.pi

    n_noise_volumes = 0
    if mag_norf_file:
        mag_norf_img = nb.load(mag_norf_file)
        mag_norf_data = mag_norf_img.get_fdata()
        n_noise_volumes = mag_norf_data.shape[3]
        mag_data = np.concatenate((mag_data, mag_norf_data), axis=3)

        if has_complex:
            pha_norf_img = nb.load(pha_norf_file)
            pha_norf_data = pha_norf_img.get_fdata()
            pha_data = np.concatenate((pha_data, pha_norf_data), axis=3)

    # Combine magnitude and phase into complex-valued data
    complex_data = mag_data * np.exp(1j * pha_data)

    # Select the first volume of the complex data and take the absolute values
    TEMPVOL = np.abs(complex_data[..., 0])

    # Find the minimum non-zero value in the first volume and divide the complex data by it
    ARG['ABSOLUTE_SCALE'] = np.min(TEMPVOL[TEMPVOL != 0])
    complex_data = complex_data / ARG['ABSOLUTE_SCALE']

    if complex_data.shape[3] < 6:
        raise ValueError("Two few volumes in the input data")

    # Create a copy of the complex data for processing
    KSP2 = complex_data.copy()
    matdim = KSP2.shape

    # Create mean 3D array from all non-noise volumes of shape (X, Y, Z)
    meanphase = np.mean(complex_data, axis=3)
    # Multiply the mean array by either 1 or 0 (default is 0)
    meanphase = meanphase * ARG['phase_slice_average_for_kspace_centering']

    # Preallocate 4D array of zeros
    DD_phase = np.zeros_like(complex_data)

    # If the temporal phase is 1 - 3, do a standard low-pass filtered map
    if ARG['temporal_phase'] > 0:
        # Loop over slices backwards
        for i_slice in range(matdim[2])[::-1]:
            # Loop over volumes forward, including the noise volumes(???)
            for j_vol in range(matdim[3]):
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
                n_x, n_y = tmp.shape
                # TODO: Verify that this works.
                # tmp = bsxfun(@times,tmp,reshape(tukeywin(n_y,1).^phase_filter_width,[1 n_y]));
                tukey_window = np.outer(tukey(n_y, 1) ** ARG['phase_filter_width'], tukey(n_y, 1) ** ARG['phase_filter_width'])
                tmp *= tukey_window
                # tmp = bsxfun(@times,tmp,reshape(tukeywin(n_x,1).^phase_filter_width,[n_x 1]));
                tukey_window = np.outer(tukey(n_x, 1) ** ARG['phase_filter_width'], tukey(n_x, 1) ** ARG['phase_filter_width'])
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

    # Loop over slices backwards
    # XXX: Do we need this loop? Can't we just directly multiply?
    for i_slice in range(matdim[2])[::-1]:
        # Loop over volumes forward, including the noise volumes(???)
        for j_vol in range(matdim[3]):
            # Multiply the 4D array by the exponential of the angle of the filtered phase
            KSP2[:, :, i_slice, j_vol] = KSP2[:, :, i_slice, j_vol] * np.exp(-1j * np.angle(DD_phase[:, :, i_slice, j_vol]))

    print('Completed estimating slice-dependent phases ...')
    if not ARG['kernel_size_gfactor']:
        # Select first 90 (or fewer, if run is shorter) volumes from 4D array
        KSP2 = KSP2[:, :, :, :min(90, matdim[3] + 1)]
    else:
        # Select first N volumes from 4D array, based on kernel_size_gfactor(4)
        KSP2 = KSP2[:, :, :, :min(ARG['kernel_size_gfactor'][3], matdim[3] + 1)]

    # Replace NaNs and Infs with zeros
    KSP2[np.isnan(KSP2)] = 0
    KSP2[np.isinf(KSP2)] = 0
    master_fast = 1

    # Preallocate 4D array of zeros
    KSP_recon = np.zeros_like(KSP2)

    if not ARG['kernel_size_gfactor']:
        ARG['kernel_size'] = [14, 14, 1]
    else:
        ARG['kernel_size'] = ARG['kernel_size_gfactor'][:3]

    QQ['KSP_processed'] = np.zeros((1, matdim[0] - ARG['kernel_size'][0]))
    ARG['patch_average'] = 0
    ARG['patch_average_sub'] = ARG['gfactor_patch_overlap']

    ARG['LLR_scale'] = 0
    ARG['NVR_threshold'] = 1
    ARG['soft_thrs'] = 10  # MPPCa   (When Noise varies)
    # ARG['soft_thrs'] = []  # NORDIC (When noise is flat)

    # Preallocate 3D arrays of zeros
    KSP_weight = np.zeros((matdim[0], matdim[1], matdim[2]))
    NOISE = np.zeros_like(KSP_weight)
    Component_threshold = np.zeros_like(KSP_weight)
    energy_removed = np.zeros_like(KSP_weight)
    SNR_weight = np.zeros_like(KSP_weight)
    # The original code re-creates KSP_processed here for no reason

    # patch_average is hardcoded as 0 so this block is always executed.
    if ARG['patch_average'] == 0:
        KSP_processed = QQ['KSP_processed'] * 0  # pointless
        # WTAF is this loop doing?
        val = max(1, np.floor(ARG['kernel_size'][0] / ARG['patch_average_sub']))
        for nw1 in range(1, val):
            # KSP_processed(1,nw1 : max(1,floor(ARG.ARG['kernel_size'](1)/ARG.patch_average_sub)):end)=2;
            KSP_processed[0, nw1:val:] = 2
        KSP_processed[-1] = 0
        QQ['KSP_processed'] = KSP_processed

    print('Estimating g-factor')
    QQ['ARG'] = ARG
    for n1 in range(QQ['KSP_processed'].shape[1]):
        (
            KSP_recon,
            _,
            KSP_weight,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        ) = sub_LLR_Processing(
            KSP_recon,
            KSP2,
            ARG,
            n1,
            QQ,
            master_fast,
            KSP_weight,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        )

    KSP_recon = KSP_recon / KSP_weight[..., None]
    ARG['NOISE'] = np.sqrt(NOISE / KSP_weight)
    ARG['Component_threshold'] = Component_threshold / KSP_weight
    ARG['energy_removed'] = energy_removed / KSP_weight
    ARG['SNR_weight'] = SNR_weight / KSP_weight
    IMG2 = KSP_recon
    ARG2 = ARG
    print('completed estimating g-factor')
    gfactor = ARG['NOISE']

    if KSP2.shape[3] < 6:
        gfactor[np.isnan(gfactor)] = 0
        gfactor[gfactor == 0] = np.median(gfactor[gfactor != 0])

    ARG['data_has_zero_elements'] = 0
    if np.sum(gfactor == 0) > 0:
        gfactor[np.isnan(gfactor)] = 0
        gfactor[gfactor < 1] = np.median(gfactor[gfactor != 0])
        ARG['data_has_zero_elements'] = 1

    if algorithm == 'mppca':
        gfactor = np.ones(gfactor.shape)

    if save_gfactor_map:
        gfactor_for_img = np.abs(gfactor)  # Absolute value isn't used for correction, so why is the output map absolute?
        gfactor_for_img[np.isnan(gfactor_for_img)] = 0

        tmp = np.sort(np.abs(gfactor_for_img.flatten()))
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]  # added -1 to match MATLAB indexing
        gain_level = np.floor(np.log2(32000 / sn_scale))
        if not ARG['full_dynamic_range']:
            gain_level = 0

        gfactor_for_img = np.abs(gfactor_for_img) * (2 ** gain_level)
        gfactor_img = nb.Nifti1Image(gfactor_for_img, mag_img.affine, mag_img.header)
        gfactor_img.to_filename(out_dir / 'gfactor.nii.gz')

    KSP2 = complex_data.copy()
    matdim = KSP2.shape

    for i_slice in range(matdim[2])[::-1]:
        for j_vol in range(matdim[3]):  # include the noise
            KSP2[:, :, i_slice, j_vol] = KSP2[:, :, i_slice, j_vol] * np.exp(-1j * np.angle(meanphase[:, :, i_slice]))

    for j_vol in range(matdim[3]):
        KSP2[:, :, :, j_vol] = KSP2[:, :, :, j_vol] / gfactor

    if n_noise_volumes > 0:
        KSP2_NOISE = KSP2[..., -n_noise_volumes:]

    if ARG['temporal_phase'] == 3:
        # Secondary step for filtered phase with residual spikes
        for i_slice in range(matdim[2])[::-1]:
            for j_vol in range(matdim[3]):
                phase_diff = np.angle(KSP2[:, :, i_slice, j_vol] / DD_phase[:, :, i_slice, j_vol])
                mask = np.abs(phase_diff) > 1
                mask2 = np.abs(KSP2[:, :, i_slice, j_vol]) > np.sqrt(2)
                DD_phase2 = DD_phase[:, :, i_slice, j_vol]
                tmp = KSP2[:, :, i_slice, j_vol]
                DD_phase2[mask * mask2] = tmp[mask * mask2]
                DD_phase[:, :, i_slice, j_vol] = DD_phase2

    for i_slice in range(matdim[2])[::-1]:
        for j_vol in range(matdim[3]):
            KSP2[:, :, i_slice, j_vol] = KSP2[:, :, i_slice, j_vol] * np.exp(-1j * np.angle(DD_phase[:, :, i_slice, j_vol]))

    KSP2[np.isnan(KSP2)] = 0
    KSP2[np.isinf(KSP2)] = 0

    ARG['measured_noise'] = 1
    if n_noise_volumes > 0:
        tmp_noise = KSP2_NOISE.copy()
        tmp_noise[np.isnan(tmp_noise)] = 0
        tmp_noise[np.isinf(tmp_noise)] = 0
        ARG['measured_noise'] = np.std(tmp_noise[tmp_noise != 0])  # sqrt(2) for real and complex

    if has_complex:
        ARG['measured_noise'] = ARG['measured_noise'] / np.sqrt(2)

    if ARG['data_has_zero_elements']:
        MASK = np.sum(np.abs(KSP2), axis=3) == 0
        num_zero_elements = np.sum(MASK)
        for i_vol in range(matdim[3]):
            tmp = KSP2[:, :, :, i_vol]
            tmp[MASK] = (np.random.normal(size=num_zero_elements) + 1j * np.random.normal(size=num_zero_elements)) / np.sqrt(2)
            KSP2[:, :, :, i_vol] = tmp

    master_fast = 1
    KSP_recon = np.zeros_like(KSP2)
    ARG['kernel_size'] = np.full((1, 3), int(np.round(np.cbrt(KSP2.shape[3] * 11))))
    if ARG['kernel_size_PCA']:
        ARG['kernel_size'] = ARG['kernel_size_PCA']

    if matdim[2] <= ARG['kernel_size'][2]:  # Number of slices is less than cubic kernel
        ARG['kernel_size'] = np.full((1, 3), int(np.round(np.sqrt(KSP2.shape[3] * 11 / matdim[2]))))
        ARG['kernel_size'][2] = matdim[2]

    QQ['KSP_processed'] = np.zeros((1, KSP2.shape[0] - ARG['kernel_size'][0]))
    ARG['patch_average'] = 0
    ARG['patch_average_sub'] = ARG['NORDIC_patch_overlap']
    ARG['LLR_scale'] = 1
    ARG['NVR_threshold'] = 0
    ARG['soft_thrs'] = []  # NORDIC (When noise is flat)
    ARG['NVR_threshold'] = 0  # for no reason

    for _ in range(1, 11):
        _, S, _ = np.linalg.svd(np.random.normal(size=(np.prod(ARG['kernel_size']), KSP2.shape[3])))
        ARG['NVR_threshold'] = ARG['NVR_threshold'] + S[1]

    # XXX: This divides by 10, but multiplies by the others. Is that expected?
    if has_complex:
        # sqrt(2) due to complex  1.20 due to understimate of g-factor
        ARG['NVR_threshold'] = ARG['NVR_threshold'] / 10 * np.sqrt(2) * ARG['measured_noise'] * ARG['factor_error']
    else:
        ARG['NVR_threshold'] = ARG['NVR_threshold'] / 10 * ARG['measured_noise'] * ARG['factor_error']

    if algorithm in ['mppca', 'mppca+nordic']:
        ARG['soft_thrs'] = 10

    KSP_weight = np.zeros_like(KSP2[..., 0])
    NOISE = KSP_weight.copy()
    Component_threshold = KSP_weight.copy()
    energy_removed = KSP_weight.copy()
    SNR_weight = KSP_weight.copy()
    QQ['KSP_processed'] = np.zeros((1, KSP2.shape[0] - ARG['kernel_size'][0]))

    if ARG['patch_average'] == 0:
        KSP_processed = QQ['KSP_processed'] * 0
        val = max(1, np.floor(ARG['kernel_size'][0] / ARG['patch_average_sub']))
        for nw1 in range(1, val):
            KSP_processed[0, nw1:val:] = 2
        KSP_processed[-1] = 0
        QQ['KSP_processed'] = KSP_processed

    print('Starting NORDIC')
    QQ['ARG'] = ARG
    for n1 in range(QQ['KSP_processed'].shape[1]):
        (
            KSP_recon,
            _,
            KSP_weight,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        ) = sub_LLR_Processing(
            KSP_recon,
            KSP2,
            ARG,
            n1,
            QQ,
            master_fast,
            KSP_weight,
            NOISE,
            Component_threshold,
            energy_removed,
            SNR_weight,
        )
    # Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
    KSP_recon = KSP_recon / KSP_weight[..., None]
    ARG['NOISE'] = np.sqrt(NOISE / KSP_weight)
    ARG['Component_threshold'] = Component_threshold / KSP_weight
    ARG['energy_removed'] = energy_removed / KSP_weight
    ARG['SNR_weight'] = SNR_weight / KSP_weight
    IMG2 = KSP_recon
    print('Completing NORDIC')

    residual = KSP2 - KSP_recon
    residual_img = nb.Nifti1Image(residual, mag_img.affine, mag_img.header)
    residual_img.to_filename(out_dir / 'residual.nii.gz')

    for i_vol in range(matdim[3]):
        IMG2[:, :, :, i_vol] = IMG2[:, :, :, i_vol] * gfactor

    for i_slice in range(matdim[2])[::-1]:
        for j_vol in range(matdim[3]):
            IMG2[:, :, i_slice, j_vol] = IMG2[:, :, i_slice, j_vol] * np.exp(1j * np.angle(DD_phase[:, :, i_slice]))

    IMG2 = IMG2 * ARG['ABSOLUTE_SCALE']
    IMG2[np.isnan(IMG2)] = 0

    if ARG['make_complex_nii']:
        IMG2_tmp = np.abs(IMG2)  # remove g-factor and noise for DUAL 1
        IMG2_tmp[np.isnan(IMG2_tmp)] = 0
        tmp = np.sort(np.abs(IMG2_tmp.flatten()))
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        gain_level = np.floor(np.log2(32000 / sn_scale))

        if not ARG['full_dynamic_range']:
            gain_level = 0

        IMG2_tmp = np.abs(IMG2_tmp) * (2 ** gain_level)
        IMG2_img = nb.Nifti1Image(IMG2_tmp, mag_img.affine, mag_img.header)
        IMG2_img.to_filename(out_dir / 'magn.nii.gz')

        IMG2_tmp = np.angle(IMG2)
        IMG2_tmp = (IMG2_tmp / (2 * np.pi) + range_center) * range_norm
        IMG2_img = nb.Nifti1Image(IMG2_tmp, mag_img.affine, mag_img.header)
        IMG2_img.to_filename(out_dir / 'phase.nii.gz')
    else:
        IMG2 = np.abs(IMG2)
        IMG2[np.isnan(IMG2)] = 0
        tmp = np.sort(np.abs(IMG2).flatten())
        sn_scale = 2 * tmp[int(np.round(0.99 * len(tmp))) - 1]
        gain_level = np.floor(np.log2(32000 / sn_scale))

        if not ARG['full_dynamic_range']:
            gain_level = 0

        IMG2 = np.abs(IMG2) * (2 ** gain_level)
        IMG2_img = nb.Nifti1Image(IMG2, mag_img.affine, mag_img.header)
        IMG2_img.to_filename(out_dir / 'magn.nii.gz')

"""Tests for the denoise module."""
from pathlib import Path

import pytest

from nordic import denoise


@pytest.mark.parametrize('use_phase', [True, False])
@pytest.mark.parametrize('use_norf', [True, False])
@pytest.mark.parametrize('factor_error', [0.5, 1, 1.5])
@pytest.mark.parametrize('full_dynamic_range', [True, False])
@pytest.mark.parametrize('temporal_phase', [0, 1, 2, 3])
@pytest.mark.parametrize('algorithm', ['nordic', 'mppca', 'gfactor+mppca'])
@pytest.mark.parametrize('patch_overlap_gfactor', [1, 2, 3])
@pytest.mark.parametrize('kernel_size_gfactor', [None, 3, 5, 7])
@pytest.mark.parametrize('patch_overlap_pca', [1, 2, 3])
@pytest.mark.parametrize('kernel_size_pca', [None, 3, 5, 7])
@pytest.mark.parametrize('phase_slice_average_for_kspace_centering', [True, False])
@pytest.mark.parametrize('phase_filter_width', [1, 3, 10])
@pytest.mark.parametrize('save_gfactor_map', [True, False])
@pytest.mark.parametrize('soft_thrs', [None, 'auto', 10])
@pytest.mark.parametrize('debug', [True, False])
@pytest.mark.parametrize('scale_patches', [True, False])
@pytest.mark.parametrize('patch_average', [True, False])
@pytest.mark.parametrize('llr_scale', [0, 1])
def test_run_nordic_smoke(
    use_phase,
    use_norf,
    factor_error,
    full_dynamic_range,
    temporal_phase,
    algorithm,
    patch_overlap_gfactor,
    kernel_size_gfactor,
    patch_overlap_pca,
    kernel_size_pca,
    phase_slice_average_for_kspace_centering,
    phase_filter_width,
    save_gfactor_map,
    soft_thrs,
    debug,
    scale_patches,
    patch_average,
    llr_scale,
    tmp_path,
):
    """Test the run_nordic function.

    This test parameterizes the input arguments to run_nordic, runs the function,
    and checks that the expected files are generated.
    """
    data_path = Path(__file__).parent / 'data'

    # Load test data
    mag_file = data_path / 'mag.nii.gz'
    pha_file = None
    if use_phase:
        pha_file = data_path / 'pha.nii.gz'

    pha_norf_file = None
    mag_norf_file = None
    if use_norf:
        mag_norf_file = data_path / 'mag_norf.nii.gz'
        if use_phase:
            pha_norf_file = data_path / 'pha_norf.nii.gz'

    # Run NORDIC
    denoise.run_nordic(
        mag_file=mag_file,
        pha_file=pha_file,
        mag_norf_file=mag_norf_file,
        pha_norf_file=pha_norf_file,
        out_dir=tmp_path,
        factor_error=factor_error,
        full_dynamic_range=full_dynamic_range,
        temporal_phase=temporal_phase,
        algorithm=algorithm,
        patch_overlap_gfactor=patch_overlap_gfactor,
        kernel_size_gfactor=kernel_size_gfactor,
        patch_overlap_pca=patch_overlap_pca,
        kernel_size_pca=kernel_size_pca,
        phase_slice_average_for_kspace_centering=phase_slice_average_for_kspace_centering,
        phase_filter_width=phase_filter_width,
        save_gfactor_map=save_gfactor_map,
        soft_thrs=soft_thrs,
        debug=debug,
        scale_patches=scale_patches,
        patch_average=patch_average,
        llr_scale=llr_scale,
    )
    assert (tmp_path / 'magn.nii.gz').exists()
    if use_phase:
        assert (tmp_path / 'phase.nii.gz').exists()

    if save_gfactor_map:
        assert (tmp_path / 'gfactor.nii.gz').exists()

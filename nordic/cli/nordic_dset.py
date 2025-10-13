"""Apply NORDIC to BIDS dataset."""

from argparse import ArgumentParser, RawTextHelpFormatter
from functools import partial
from pathlib import Path

from nordic import denoise


def get_parser():
    """Build parser object."""

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f'Path does not exist: <{path}>.')
        return Path(path).absolute()

    def _process_value(value):
        import bids

        if value is None:
            return bids.layout.Query.NONE
        elif value == '*':
            return bids.layout.Query.ANY
        else:
            return value

    def _filter_pybids_none_any(dct):
        d = {}
        for k, v in dct.items():
            if isinstance(v, list):
                d[k] = [_process_value(val) for val in v]
            else:
                d[k] = _process_value(v)
        return d

    def _bids_filter(value, parser):
        from json import JSONDecodeError, loads

        if value:
            if Path(value).exists():
                try:
                    return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
                except JSONDecodeError as e:
                    raise parser.error(f'JSON syntax error in: <{value}>.') from e
            else:
                raise parser.error(f'Path does not exist: <{value}>.')

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    PathExists = partial(_path_exists, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)

    parser.add_argument(
        'bids_dir',
        action='store',
        type=PathExists,
        help=(
            'The root folder of a BIDS valid dataset (sub-XXXXX folders should '
            'be found at the top level in this folder).'
        ),
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=Path,
        help=(
            'The output path for the NORDIC-denoised data. '
            'If the output path is the same as the BIDS directory, '
            'then pseudo-raw denoising is performed. '
            'In pseudo-raw denoising, the raw data will be renamed with `rec-nonordic` '
            'and the denoised data will be written out with the original names.'
        ),
    )
    parser.add_argument(
        'analysis_level',
        choices=['participant'],
        help=(
            'Processing stage to be run, only "participant" in the case of '
            'NORDIC (see BIDS-Apps specification).'
        ),
    )

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument(
        '--participant-label',
        '--participant_label',
        action='store',
        nargs='+',
        type=lambda label: label.removeprefix('sub-'),
        help=(
            'A space delimited list of participant identifiers or a single identifier '
            '(the sub- prefix can be removed)'
        ),
    )
    g_bids.add_argument(
        '-s',
        '--session-id',
        action='store',
        nargs='+',
        type=lambda label: label.removeprefix('ses-'),
        help=(
            'A space delimited list of session identifiers or a single identifier '
            '(the ses- prefix can be removed)'
        ),
    )
    g_bids.add_argument(
        '-r',
        '--run-id',
        action='store',
        nargs='+',
        type=lambda label: label.removeprefix('run-'),
        help=(
            'A space delimited list of run identifiers or a single identifier '
            '(the run- prefix can be removed)'
        ),
    )
    g_bids.add_argument(
        '-t',
        '--task-id',
        action='store',
        nargs='+',
        type=lambda label: label.removeprefix('task-'),
        help=(
            'A space delimited list of task identifiers or a single identifier '
            '(the task- prefix can be removed)'
        ),
    )
    g_bids.add_argument(
        '--bids-filter-file',
        dest='bids_filters',
        action='store',
        type=BIDSFilter,
        metavar='FILE',
        help=(
            'A JSON file describing custom BIDS input filters using PyBIDS. '
            'Supported fields: "bold".'
        ),
    )
    g_bids.add_argument(
        '--ignore',
        required=False,
        action='store',
        nargs='+',
        default=[],
        choices=['phase', 'norf'],
        help=(
            'ignore selected aspects of the input dataset to disable corresponding '
            'parts of the workflow (a space delimited list)'
        ),
    )

    g_nordic = parser.add_argument_group('Options for NORDIC')
    g_nordic.add_argument(
        '--factor-error',
        action='store',
        type=float,
        help=(
            'Error in g-factor estimation. >1 uses a higher noisefloor. '
            '<1 uses a lower noisefloor. Default is 1. '
            'Rather than modifying the gfactor map, this changes nvr_threshold.'
        ),
        default=1,
    )
    g_nordic.add_argument(
        '--full-dynamic-range',
        action='store_true',
        help='Whether to use the full dynamic range. Default is False.',
        default=False,
    )
    g_nordic.add_argument(
        '--temporal-phase',
        action='store',
        type=int,
        help='Temporal phase. Default is 1.',
        default=1,
    )
    g_nordic.add_argument(
        '--algorithm',
        action='store',
        choices=['nordic', 'mppca', 'gfactor+mppca'],
        help='Algorithm to use. Default is "nordic".',
        default='nordic',
    )
    g_nordic.add_argument(
        '--patch-overlap-gfactor',
        action='store',
        type=int,
        help='Patch overlap for g-factor estimation. Default is 2.',
        default=2,
    )
    g_nordic.add_argument(
        '--kernel-size-gfactor',
        action='store',
        type=int,
        help='Kernel size for g-factor estimation. Default is None.',
        default=None,
    )
    g_nordic.add_argument(
        '--patch-overlap-pca',
        action='store',
        type=int,
        help='Patch overlap for PCA. Default is 2.',
        default=2,
    )
    g_nordic.add_argument(
        '--kernel-size-pca',
        action='store',
        type=int,
        help='Kernel size for PCA. Default is None.',
        default=None,
    )
    g_nordic.add_argument(
        '--phase-slice-average-for-kspace-centering',
        action='store_true',
        help='Whether to average the phase slices for k-space centering. Default is False.',
        default=False,
    )
    g_nordic.add_argument(
        '--phase-filter-width',
        action='store',
        type=int,
        help='Width of the phase filter. Default is 3.',
        default=3,
    )
    g_nordic.add_argument(
        '--save-gfactor-map',
        action='store_true',
        help='Whether to save the g-factor map. Default is False.',
        default=False,
    )
    g_nordic.add_argument(
        '--debug',
        action='store_true',
        help='If True, write out intermediate files for debugging. Default is False.',
        default=False,
    )
    g_nordic.add_argument(
        '--scale-patches',
        action='store_true',
        help=(
            'Whether to scale the contributions of patches according to the variance '
            'removed by the patch or not. Default is False.'
        ),
        default=False,
    )
    g_nordic.add_argument(
        '--patch-average',
        action='store_true',
        help='Hardcoded as False in the MATLAB code (ARG.patch_average = 0).',
        default=False,
    )
    g_nordic.add_argument(
        '--llr-scale',
        action='store',
        type=float,
        help=(
            'Local low-rank scaling factor for the denoising step. Default is 1. '
            'Hardcoded as 0 for g-factor estimation and 1 for denoising in the '
            'MATLAB code (ARG.llr_scale).'
        ),
        default=1,
    )
    return parser


def main(args=None):
    """Run NORDIC on a single run."""
    opts = get_parser().parse_args(args)
    kwargs = vars(opts)

    denoise.run_nordic(**kwargs)


if __name__ == '__main__':
    main()

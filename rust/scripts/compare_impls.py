"""Parity + perf rollup for the Python vs Rust NORDIC implementations.

Reads `comparison_results/timings.csv` (written by `compare_impls.sh`) and
walks `comparison_results/<dataset>/<scenario>/{python,rust}/` for the
canonical (iter-1) outputs. Emits:
    - parity.csv  per (dataset, scenario, output) error metrics
    - summary.md  human-readable rollup combining timings + parity

Run:  python scripts/compare_impls.py [comparison_results_dir]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import nibabel as nb
import numpy as np

# Files we always check for. Other `.nii.gz` files in the output dir (the
# `--debug` intermediates) are picked up dynamically per-scenario.
PRIMARY_OUTPUT_FILES = ['magn.nii.gz', 'phase.nii.gz', 'gfactor.nii.gz']


def load_or_none(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return nb.load(str(path)).get_fdata().astype(np.float32)


def parity_metrics(out: np.ndarray | None, ref: np.ndarray | None) -> dict:
    """Compare two same-shape volumes.

    Reported metrics:
        max_abs_err: pointwise max |out - ref|
        mean_abs_err: pointwise mean |out - ref|
        corr: Pearson r over all voxels
        pct_voxels_rel_err_gt_1pct: percentage of "bright" voxels (>5% of max
            |ref|) where |out - ref| / |ref| > 0.01. This is the metric the
            Rust parity tests already use.
    """
    base = {
        'max_abs_err': None,
        'mean_abs_err': None,
        'corr': None,
        'pct_voxels_rel_err_gt_1pct': None,
        'note': '',
    }
    if out is None and ref is None:
        base['note'] = 'both missing'
        return base
    if out is None:
        base['note'] = 'rust missing'
        return base
    if ref is None:
        base['note'] = 'python missing'
        return base
    if out.shape != ref.shape:
        base['note'] = f'shape mismatch (rust={out.shape}, python={ref.shape})'
        return base

    diff = out - ref
    abs_diff = np.abs(diff)
    max_ref = float(np.max(np.abs(ref)))
    bright = np.abs(ref) > 0.05 * max_ref
    pct = 0.0
    if bright.any():
        rel = abs_diff[bright] / np.maximum(np.abs(ref[bright]), 1e-6)
        pct = float(100.0 * np.mean(rel > 0.01))

    # corrcoef errors out on a constant array (variance=0).
    if np.std(out) > 0 and np.std(ref) > 0:
        corr = float(np.corrcoef(out.flatten(), ref.flatten())[0, 1])
    else:
        corr = float('nan')

    return {
        'max_abs_err': float(abs_diff.max()),
        'mean_abs_err': float(abs_diff.mean()),
        'corr': corr,
        'pct_voxels_rel_err_gt_1pct': pct,
        'note': '',
    }


def collect_parity_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    # Layout is comparison_results/<dataset>/<scenario>/{python,rust}/
    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if dataset_dir.name in {'python', 'rust'}:  # safety
            continue
        for scen_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            py_dir = scen_dir / 'python'
            rs_dir = scen_dir / 'rust'
            if not py_dir.exists() and not rs_dir.exists():
                continue
            # Compare every NIfTI present on either side. This picks up the
            # primary outputs (magn, phase, gfactor) plus the `--debug`
            # intermediates (filtered_phase_*, magn_pregfactor_normalized, …).
            files: set[str] = set()
            for d in (py_dir, rs_dir):
                if d.exists():
                    files.update(p.name for p in d.glob('*.nii.gz'))
            for fname in sorted(files):
                ref = load_or_none(py_dir / fname)
                ours = load_or_none(rs_dir / fname)
                metrics = parity_metrics(ours, ref)
                rows.append({
                    'dataset': dataset_dir.name,
                    'scenario': scen_dir.name,
                    'output': fname,
                    'is_primary': fname in PRIMARY_OUTPUT_FILES,
                    **metrics,
                })
    return rows


def collect_perf_rows(timings_csv: Path) -> list[dict]:
    if not timings_csv.exists():
        return []
    grouped: dict[tuple, list[tuple[float, float, float, float]]] = defaultdict(list)
    with open(timings_csv) as f:
        for r in csv.DictReader(f):
            try:
                wall = float(r['wall_seconds'])
                user = float(r['user_seconds'])
                sysc = float(r['sys_seconds'])
                rss = float(r['peak_rss_kb'])
                rc = int(r['exit_code'])
            except (TypeError, ValueError):
                continue
            if rc != 0:
                # Skip failed runs in the rollup (we still keep them in the CSV).
                continue
            key = (
                r['dataset'], r['impl'], r['algorithm'],
                r['temporal_phase'], r['use_phase'],
            )
            grouped[key].append((wall, user, sysc, rss))

    out: list[dict] = []
    for key, vals in grouped.items():
        walls = [v[0] for v in vals]
        users = [v[1] for v in vals]
        syss = [v[2] for v in vals]
        rsses = [v[3] for v in vals]
        # Wall: report median (robust against outlier first-iter cold cache)
        # plus min (best-case for the impl). RSS: report max across iters.
        out.append({
            'dataset': key[0],
            'impl': key[1],
            'algorithm': key[2],
            'temporal_phase': key[3],
            'use_phase': key[4],
            'n_iters': len(vals),
            'wall_min_s': min(walls),
            'wall_median_s': median(walls),
            'wall_max_s': max(walls),
            'cpu_median_s': median([u + s for u, s in zip(users, syss)]),
            'peak_rss_max_mb': max(rsses) / 1024.0,
        })
    return out


def write_parity_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text('# no parity rows\n')
        return
    fields = ['dataset', 'scenario', 'output', 'is_primary',
              'max_abs_err', 'mean_abs_err', 'corr',
              'pct_voxels_rel_err_gt_1pct', 'note']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fields}
            writer.writerow(row)


def fmt_num(v, fmt='.4g'):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def write_summary(perf_rows: list[dict], parity_rows: list[dict], path: Path) -> None:
    with open(path, 'w') as f:
        f.write('# nordic implementation comparison\n\n')
        f.write('Python uses CLI defaults (`n_jobs=1`, single-threaded patch loop).\n')
        f.write('Rust uses CLI defaults (rayon global pool ≈ `os::cpu_count()`).\n')
        f.write('Both timed under WSL via `/usr/bin/time -v`.\n\n')

        # ---- speed ratio summary ----
        f.write('## Performance\n\n')
        f.write('| dataset | algo | tp | phase | impl | iters | min wall (s) | median wall (s) | median CPU (s) | peak RSS (MB) |\n')
        f.write('|---|---|---|---|---|---:|---:|---:|---:|---:|\n')
        perf_rows.sort(key=lambda r: (
            r['dataset'], r['algorithm'], r['temporal_phase'], r['use_phase'], r['impl'],
        ))
        for r in perf_rows:
            f.write(
                f"| {r['dataset']} | {r['algorithm']} | {r['temporal_phase']} | "
                f"{r['use_phase']} | {r['impl']} | {r['n_iters']} | "
                f"{r['wall_min_s']:.2f} | {r['wall_median_s']:.2f} | "
                f"{r['cpu_median_s']:.2f} | {r['peak_rss_max_mb']:.1f} |\n"
            )

        # ---- speed-up table (rust median / python median) ----
        f.write('\n### Rust-vs-Python wall time speedup (Python median ÷ Rust median)\n\n')
        f.write('| dataset | algo | tp | phase | python median (s) | rust median (s) | speedup × |\n')
        f.write('|---|---|---|---|---:|---:|---:|\n')
        keyed = {(r['dataset'], r['impl'], r['algorithm'], r['temporal_phase'], r['use_phase']): r
                 for r in perf_rows}
        scen_keys = sorted({(d, a, tp, up) for (d, _, a, tp, up) in keyed.keys()})
        for d, a, tp, up in scen_keys:
            py = keyed.get((d, 'python', a, tp, up))
            rs = keyed.get((d, 'rust', a, tp, up))
            if py is None or rs is None:
                continue
            speedup = py['wall_median_s'] / rs['wall_median_s'] if rs['wall_median_s'] > 0 else float('inf')
            f.write(
                f"| {d} | {a} | {tp} | {up} | "
                f"{py['wall_median_s']:.2f} | {rs['wall_median_s']:.2f} | "
                f"{speedup:.2f} |\n"
            )

        # ---- parity ----
        parity_rows.sort(key=lambda r: (r['dataset'], r['scenario'], r['output']))
        primary = [r for r in parity_rows if r.get('is_primary')]
        debug = [r for r in parity_rows if not r.get('is_primary')]

        def _emit_table(rows):
            f.write('| dataset | scenario | output | max abs err | mean abs err | corr | %bright voxels rel-err > 1% | note |\n')
            f.write('|---|---|---|---:|---:|---:|---:|---|\n')
            for r in rows:
                f.write(
                    f"| {r['dataset']} | {r['scenario']} | {r['output']} | "
                    f"{fmt_num(r['max_abs_err'])} | {fmt_num(r['mean_abs_err'])} | "
                    f"{fmt_num(r['corr'], '.5f')} | {fmt_num(r['pct_voxels_rel_err_gt_1pct'], '.2f')} | "
                    f"{r.get('note', '')} |\n"
                )

        f.write('\n## Parity — primary outputs (magn / phase / gfactor)\n\n')
        _emit_table(primary)
        if debug:
            f.write('\n## Parity — debug intermediates (`--debug` outputs)\n\n')
            _emit_table(debug)


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1] if len(argv) > 1 else 'comparison_results').resolve()
    if not results_dir.exists():
        print(f'no such directory: {results_dir}', file=sys.stderr)
        return 1

    timings_csv = results_dir / 'timings.csv'
    parity_csv = results_dir / 'parity.csv'
    summary_md = results_dir / 'summary.md'

    perf = collect_perf_rows(timings_csv)
    parity = collect_parity_rows(results_dir)

    write_parity_csv(parity, parity_csv)
    write_summary(perf, parity, summary_md)

    print(f'Wrote {parity_csv}')
    print(f'Wrote {summary_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

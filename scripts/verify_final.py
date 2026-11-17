"""Assert final paper artwork and frozen release inventories without inference."""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def verify():
    figures = rows(ROOT / 'manifests/R2_FIGURES.csv')
    assert len(figures) == 25
    assert len({row['filename'] for row in figures}) == 25
    current_figures = [row for row in figures if row['status'] == 'current']
    historical_figures = [row for row in figures if row['status'] == 'historical']
    assert len(current_figures) == 22
    assert len(historical_figures) == 3
    assert [row['figure'] for row in current_figures[:5]] == [f'Main Figure {i}' for i in range(1, 6)]
    assert [row['figure'] for row in current_figures[5:]] == [f'Appendix Figure {i}' for i in range(1, 18)]
    binding = json.loads((ROOT / 'manifests/PAPER_ARTWORK_BINDING.json').read_text(encoding='utf-8'))
    assert binding['figure_count'] == len(current_figures) == 22
    assert binding['historical_figure_count'] == len(historical_figures) == 3
    for row in figures:
        path = ROOT / row['filename']
        assert path.stat().st_size == int(row['bytes']), row['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['filename']
        assert all((ROOT / source).is_file() for source in row['source_files'].split(';'))
    panels = json.loads((ROOT / 'data/manifests/panel_manifest_all44.json').read_text())['panels']
    assert len(panels) == 44
    assert len({(p['cohort'], p['split'], p['environment']) for p in panels}) == 44
    for panel in panels:
        path = ROOT / panel['path']
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == panel['sha256'], panel['path']
    diagonal = rows(ROOT / 'results/additional/diagonal_comparison.csv')
    assert len(diagonal) == 6
    uniform = next(row for row in diagonal if row['environment'] == 'uniform_partner')
    expected = {
        'tuna_corrected_auroc': 0.7227148621293044,
        'symmetric_frozen_esm2_pair_mlp_auroc': 0.6910325434624119,
        'pooled_ESM_logistic_sum_absdiff_auroc': 0.6427569719077676,
        'dscript_esm2_auroc': 0.5808311097196185,
        'additive_auroc': 0.5697269205122001,
        'rapppid_auroc': 0.5250112487056012,
    }
    for column, value in expected.items():
        assert abs(float(uniform[column]) - value) < 1e-12, column
    pooled = rows(ROOT / 'results/additional/pooled_esm_summary.csv')
    for row in pooled:
        column = row['model'] + '_' + row['metric']
        if column not in uniform:
            continue
        target = next(d for d in diagonal if d['environment'] == row['environment'])
        assert abs(float(target[column]) - float(row['mean'])) < 1e-12, column
    crosswalk = rows(ROOT / 'data/mappings/vidal_orf_crosswalk.csv')
    checkpoints = rows(ROOT / 'manifests/CHECKPOINT_INVENTORY.csv')
    assert len(crosswalk) == 1059
    assert len(checkpoints) == 114
    random = rows(ROOT / 'manifests/FULLY_RANDOM_PREDICTIONS.csv')
    assert len(random) == 72 and all((ROOT / row['path']).is_file() for row in random)
    random_ext = rows(ROOT / 'manifests/FULLY_RANDOM_EXTERNAL_PREDICTIONS.csv')
    assert len(random_ext) == 24 and all((ROOT / row['path']).is_file() for row in random_ext)
    nested_git = [p for p in ROOT.rglob('.git') if p != ROOT / '.git']
    assert not nested_git, nested_git
    git_ancestry = 'ABSENT'
    if (ROOT / '.git').exists():
        import subprocess
        log = subprocess.check_output(
            ['git', '-C', str(ROOT), 'log', '--all', '--format=%H|%P|%an|%ae|%cn|%ce'],
            text=True,
        ).strip().splitlines()
        assert len(log) >= 1, 'expected at least one anonymous commit'
        expected_email = 'anonymous' + chr(64) + 'iclr2027.invalid'
        shas = set()
        roots = 0
        for entry in log:
            sha, parents, an, ae, cn, ce = entry.split('|')
            shas.add(sha)
            assert an == 'Anonymous Authors' and cn == 'Anonymous Authors', entry
            assert ae == expected_email and ce == expected_email, entry
            parent_list = parents.split() if parents else []
            assert len(parent_list) <= 1, f'commit {sha} has merge parents: {parents}'
            if not parent_list:
                roots += 1
        assert roots == 1, f'expected 1 anonymous root commit, got {roots}'
        for entry in log:
            sha, parents, *_ = entry.split('|')
            for parent in (parents.split() if parents else []):
                assert parent in shas, f'commit {sha} has external parent {parent}'
        git_ancestry = (
            'SINGLE_ANONYMOUS_ROOT_COMMIT'
            if len(log) == 1
            else f'LINEAR_ANONYMOUS_HISTORY_{len(log)}_COMMITS'
        )
    assert not any(
        p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors'}
        for p in ROOT.rglob('*')
        if '.git' not in p.parts and '.venv' not in p.parts
    )
    return {'status': 'PASS', 'paper_figures': 22, 'historical_figures': 3, 'actual_stage_panels': 44,
            'figure5_exact_auroc_assertions': expected, 'crosswalk_rows': 1059,
            'checkpoint_inventory_only': 114, 'additional_random_gold_score_files': 72, 'additional_random_external_score_files': 24,
            'git_ancestry': git_ancestry, 'scientific_limit': 'Main Figure 5 (pooled-ESM baseline) inputs table-checked; no new fit or inference'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))

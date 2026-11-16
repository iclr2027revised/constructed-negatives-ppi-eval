"""Frozen deterministic sampler fixture; no real data generation."""
from pathlib import Path
import hashlib, json, sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def fixture():
    sys.path.insert(0, str(ROOT/'src'))
    from ppi_shift import negatives as n, rebuttal_random_pair as r, e002_negative, e003_method_fidelity
    proteins = {f'P{i}': (s, hashlib.sha256(s.encode()).hexdigest())
                for i, s in enumerate(['ACDEFGHIK', 'LMNPQRSTV', 'WYACDEFGH', 'IKLMNPQRS', 'TVWYACDEF', 'GHIKLMNPQ', 'RSTVWYACD', 'EFGHIKLMN'])}
    excluded = {('P0', 'P1')}
    seq_excluded = {tuple(sorted((proteins['P2'][1], proteins['P3'][1])))}
    rows = []
    for seed in (11, 22, 33, 20261106):
        pairs, audit = r.sample_pairs(proteins, excluded, seq_excluded, 10, seed)
        assert pairs == r.sample_pairs(proteins, excluded, seq_excluded, 10, seed)[0]
        assert ('P0', 'P1') not in pairs and ('P2', 'P3') not in pairs
        rows.append({'seed': seed, 'pairs': pairs, 'audit': audit})
    try:
        r.sample_pairs(proteins, excluded, seq_excluded, 27, 11)
        raise AssertionError('shortage accepted')
    except ValueError:
        pass
    anchors = []
    for version in n.SUPPORTED_SCHEMA_VERSIONS:
        for i in range(7):
            p = n.PositivePair('train', i, 'fixture', f'P{i}', f'P{i+1}', proteins[f'P{i}'][0], proteins[f'P{i+1}'][0], version)
            anchors.append({'id': p.positive_row_id, 'anchor': [q.protein_id for q in n.anchor_orientation(p)],
                            'rank': n._hash_rank(11, 'uniform_partner', p.positive_row_id, 'candidate-index', version)})
    value = {'random_pairs': rows, 'anchors': anchors, 'shortage_rejected': True, 'modules_imported': 4}
    return {'status': 'PASS', 'sha256': hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), 'fixture': value}

if __name__ == '__main__':
    actual=fixture()
    expected=json.loads((ROOT/'tests/fixtures/sampler_expected.json').read_text())
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    print(json.dumps({'status':'PASS','fixture_sha256':actual['sha256'],'anchor_contracts':21,'random_pair_seeds':4,'shortage_rejected':True},indent=2))

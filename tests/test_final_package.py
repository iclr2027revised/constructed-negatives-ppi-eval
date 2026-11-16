"""Final additions: exact current artwork and numeric input contracts."""
from pathlib import Path
import importlib.util
import json


def test_final_paper_assets_and_exact_baselines():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('verify_final', root / 'scripts/verify_final.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify()['status'] == 'PASS'


def test_sampler_matches_frozen_serialized_fixture():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('verify_sampler', root / 'scripts/verify_sampler.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((root / 'tests/fixtures/sampler_expected.json').read_text())
    assert json.loads(json.dumps(module.fixture())) == expected


def test_scanner_preserves_email_detection_after_long_sequence():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('validate_release', root / 'scripts/validate_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = 'ACDEFGHIKLMNPQRSTVWY' * 200 + '\n' + 'reviewer' + chr(64) + 'example.invalid'
    findings = module._scan_text('fixture', text, module.load_patterns(None), frozenset())
    assert [(row['kind'], row['line']) for row in findings] == [('email_address', '2')]

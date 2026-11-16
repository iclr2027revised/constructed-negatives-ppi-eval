"""Offline verification for the core archive, without optional prediction data."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import reproduce
from verify_additional import summary_checks

def main():
    manifest=json.loads((ROOT/'manifests/CORE_INTEGRITY.json').read_text())
    for rel,entry in manifest['files'].items():
        path=ROOT/rel
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Core integrity mismatch: '+rel)
    results=[reproduce.check_metric_level(),reproduce.check_release_audit_tables(),reproduce.check_figure_sources()]
    for result in results:
        if result['status']!='PASS':raise ValueError(result)
    print(json.dumps({'core_files_verified':len(manifest['files']),'checks':results,'additional':summary_checks()},indent=2,default=int))

if __name__=='__main__':main()

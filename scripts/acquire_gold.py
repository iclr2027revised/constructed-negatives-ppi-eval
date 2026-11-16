"""Download the pinned public Gold v4 files and verify provider MD5 checksums.

This acquires upstream data, not the historical materialized project panels.
The exact released panels are already supplied in data/frozen_panels.
"""
import argparse,hashlib,json,urllib.request
import sys
from pathlib import Path

sys.dont_write_bytecode = True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=json.loads((Path(__file__).resolve().parents[1]/'data/manifests/gold_acquisition.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for entry in m['files']:
        path=args.output/entry['name']
        if path.exists():
            if hashlib.md5(path.read_bytes()).hexdigest()!=entry['computed_md5']:raise ValueError('Existing file differs: '+str(path))
            continue
        data=urllib.request.urlopen(entry['download_url'],timeout=120).read()
        if len(data)!=entry['size'] or hashlib.md5(data).hexdigest()!=entry['computed_md5']:raise ValueError('Download checksum mismatch: '+entry['name'])
        with path.open('xb') as f:f.write(data)
        print(entry['name'],'verified')

if __name__=='__main__':main()

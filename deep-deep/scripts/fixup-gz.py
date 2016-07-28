#!/usr/bin/env python
"""
Extract all readable JSON lines from a truncated gz file. Usage:

    fixup-gz.py <input.jl.gz> <fixed.jl.gz>

"""
import sys
import gzip
import json
import zlib
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__.strip())
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    assert in_path.exists()

    with gzip.open(str(out_path), 'wt', encoding='utf8', compresslevel=4) as f_out:
        try:
            with gzip.open(str(in_path), 'rt', encoding='utf8') as f_in:
                for line in tqdm(f_in):
                    try:
                        json.loads(line)
                    except Exception:
                        print("Error found, JSON line can't be decoded.")
                        break
                    f_out.write(line)
                else:
                    print("No errors.")
        except (EOFError, zlib.error):
            print("Error found, tuncated archive.")

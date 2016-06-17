#!/usr/bin/env python
"""
usage: crawl-keywords.py <path/to/urls.csv> <path/to/keywords.txt> <results-path> [...spider arguments]
"""
from pathlib import Path
import sys
import json
import time
import subprocess


def crawl():
    ts = str(int(time.time()))
    in_file = Path(sys.argv[1])
    keywords_file = Path(sys.argv[2])

    res_dir = Path(sys.argv[3]).joinpath(in_file.stem + "-" + ts)
    res_dir.mkdir(exist_ok=True)

    log_path = res_dir.joinpath("spider.log")
    stats_path = res_dir.joinpath("stats.jl")
    args = [
        "scrapy", "crawl", "relevant",
        "-a", "seeds_url=%s" % in_file.absolute(),
        "-a", "checkpoint_path=%s" % res_dir.absolute(),
        "-a", "keywords_file=%s" % keywords_file.absolute(),
        "-o", str(stats_path),
        # "--logfile", str(log_path),
        # "-L", "INFO",
        "-s", "CLOSESPIDER_ITEMCOUNT=10000",
    ] + sys.argv[4:]

    res_dir.joinpath("meta.json").write_text(json.dumps({
        "command": " ".join(args),
        "ts": ts,
        "input": str(in_file.absolute())
    }, indent=4))
    print("Results path: %s" % res_dir)
    subprocess.run(args, check=True)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    crawl()

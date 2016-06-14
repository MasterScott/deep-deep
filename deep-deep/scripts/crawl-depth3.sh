#!/usr/bin/env bash
scrapy crawl all \
    -a seeds_url=./$1 \
    -s GRAPH_FILENAME=$1.graph-depth3.pickle \
    --logfile $1.log \
    -L INFO \
    -s DEPTH_LIMIT=3

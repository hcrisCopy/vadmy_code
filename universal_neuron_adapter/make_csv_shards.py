#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

CHUNK_RE = re.compile(r'__(\d+)$')
KNOWN_SUFFIXES = {'.npy', '.npz', '.avi', '.mp4', '.mkv', '.mov', '.webm'}


def base_key(path_or_key: str) -> str:
    value = Path(str(path_or_key)).name
    if Path(value).suffix.lower() in KNOWN_SUFFIXES:
        value = Path(value).stem
    return CHUNK_RE.sub('', value)


def chunk_idx(path_or_key: str) -> int:
    value = Path(str(path_or_key)).name
    if Path(value).suffix.lower() in KNOWN_SUFFIXES:
        value = Path(value).stem
    m = CHUNK_RE.search(value)
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--prefix', default='shard')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError('input csv must contain path,label columns')

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[base_key(row['path'])].append((str(row['path']), str(row['label']), chunk_idx(row['path'])))

    # Greedy balance by number of CSV rows/chunks while keeping every video's rows together.
    shard_rows = [[] for _ in range(args.num_shards)]
    shard_loads = [0 for _ in range(args.num_shards)]
    for key in sorted(grouped):
        sid = min(range(args.num_shards), key=lambda i: shard_loads[i])
        rows = sorted(grouped[key], key=lambda x: x[2])
        shard_rows[sid].extend([[p, lab] for p, lab, _ci in rows])
        shard_loads[sid] += len(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sid, rows in enumerate(shard_rows):
        out = out_dir / f'{args.prefix}_{sid}.csv'
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'label'])
            writer.writerows(rows)
        video_count = len({base_key(r[0]) for r in rows})
        summary.append([sid, str(out), video_count, len(rows)])
        print(f'shard {sid}: videos={video_count}, rows={len(rows)}, path={out}')

    with (out_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['shard', 'csv_path', 'num_videos', 'num_rows'])
        writer.writerows(summary)
    print(f'wrote shards to {out_dir}')


if __name__ == '__main__':
    main()

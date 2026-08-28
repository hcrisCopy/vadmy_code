#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--kind', choices=['manifest', 'group_scores', 'generic'], default='manifest')
    parser.add_argument('--pattern', default=None)
    args = parser.parse_args()

    root = Path(args.input_root)
    if args.pattern:
        paths = sorted(root.glob(args.pattern))
    elif args.kind == 'manifest':
        paths = sorted(root.glob('shard_*/manifest.csv'))
    elif args.kind == 'group_scores':
        paths = sorted(root.glob('shard_*/group_scores.csv'))
    else:
        paths = sorted(root.glob('shard_*/*.csv'))
    if not paths:
        raise RuntimeError(f'no csv files matched under {root}')

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df['source_shard_csv'] = str(p)
        frames.append(df)
    out = pd.concat(frames, axis=0, ignore_index=True)
    key_cols = [c for c in ['key', 'path'] if c in out.columns]
    if key_cols:
        out = out.drop_duplicates(subset=key_cols, keep='first')
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f'merged {len(paths)} csv files, wrote {args.output_csv}: {len(out)} rows')

    # Merge failed.csv if present beside manifests.
    if args.kind == 'manifest':
        failed_paths = sorted(root.glob('shard_*/failed.csv'))
        if failed_paths:
            failed = pd.concat([pd.read_csv(p).assign(source_shard_csv=str(p)) for p in failed_paths], ignore_index=True)
            failed_out = Path(args.output_csv).with_name('failed.csv')
            failed.to_csv(failed_out, index=False)
            print(f'merged failed csvs, wrote {failed_out}: {len(failed)} rows')


if __name__ == '__main__':
    main()

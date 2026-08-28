#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

CHUNK_RE = re.compile(r'__(\d+)$')
VIDEO_SUFFIXES = {'.avi', '.mp4', '.mkv', '.mov', '.webm'}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def scan_videos(root):
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f'video root does not exist: {root_path}')
    return sorted(path for path in root_path.rglob('*') if path.suffix.lower() in VIDEO_SUFFIXES)


def base_key(path_or_key: str) -> str:
    return CHUNK_RE.sub('', Path(str(path_or_key)).stem)


def keys_from_csv(paths):
    if not paths:
        return None
    keys = set()
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        if 'path' not in df.columns:
            raise ValueError(f'{csv_path} must contain a path column')
        keys.update(base_key(p) for p in df['path'])
    return keys


def parse_layers(spec: str, n_layers: int):
    if spec == 'all':
        return list(range(1, n_layers + 1))
    return [int(x) for x in spec.split(',') if x.strip()]


def get_video_len(path: str):
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        return len(vr), float(vr.get_avg_fps())
    except Exception:
        import cv2
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return n, fps


def load_video_frames(path: str, indices):
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert('RGB') for frame in batch]
    except Exception:
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        wanted = set(indices)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame).convert('RGB'))
            idx += 1
        cap.release()
        return frames


def encode_hidden_cls(model, images, layer_ids):
    visual = model.visual
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    outs = []
    wanted = set(layer_ids)
    for idx, block in enumerate(visual.transformer.resblocks, start=1):
        x = block(x)
        if idx in wanted:
            outs.append(x[0].detach().float().cpu())
    return torch.stack(outs, dim=1).numpy()  # [B, L, D]


def existing_row(key, video_path, out_path, stride, layer_ids):
    try:
        z = np.load(out_path)
        n_frames = int(z['num_frames']) if 'num_frames' in z.files else -1
        fps = float(z['fps']) if 'fps' in z.files else -1
        layers = ','.join(str(int(x)) for x in z['layers'].reshape(-1)) if 'layers' in z.files else ','.join(map(str, layer_ids))
        saved_stride = int(z['stride']) if 'stride' in z.files else stride
        return [key, str(video_path), str(out_path), saved_stride, n_frames, fps, layers]
    except Exception:
        return [key, str(video_path), str(out_path), stride, -1, -1, ','.join(map(str, layer_ids))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dsanet-root', default='baseline/DSANet')
    parser.add_argument('--filter-csv', action='append', default=[], help='Only extract videos whose base key appears in this DSANet-style CSV. Can be repeated for train/test CSVs.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Only scan/filter videos and print match counts; do not load CLIP or write features.')
    parser.add_argument('--drop-last-incomplete', action='store_true', help='Use floor(num_frames / stride) snippets and drop the final incomplete <stride-frame tail. This matches DSANet/VadCLIP-style GT alignment better than ceil sampling.')
    args = parser.parse_args()

    filter_keys = keys_from_csv(args.filter_csv)
    if filter_keys is not None:
        print(f'filter enabled: {len(filter_keys)} requested video keys from {len(args.filter_csv)} csv file(s)')

    videos = scan_videos(args.video_root)
    if filter_keys is not None:
        videos = [p for p in videos if p.stem in filter_keys]
        found = {p.stem for p in videos}
        missing = sorted(filter_keys - found)
        print(f'video-root matched {len(found)}/{len(filter_keys)} requested keys')
        if missing[:10]:
            print('first missing keys:', missing[:10])
    else:
        print(f'no filter-csv: matched all {len(videos)} videos under video-root')

    if args.dry_run:
        print('dry-run enabled: not loading CLIP and not writing features')
        print('first videos:', [p.stem for p in videos[:10]])
        return

    dsanet_src = str(Path(args.dsanet_root) / 'src')
    if dsanet_src not in sys.path:
        sys.path.insert(0, dsanet_src)
    from clip import clip

    output = Path(args.out_dir)
    if output.resolve().parent.name != 'vadmy_data' and 'vadmy_data' not in output.resolve().parts:
        raise ValueError('out-dir must be inside the sibling vadmy_data directory')
    ensure_dir(output)
    feat_dir = Path(args.out_dir) / 'features'
    ensure_dir(feat_dir)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    layer_ids = parse_layers(args.layers, len(model.visual.transformer.resblocks))

    rows = []
    failed_rows = []
    for video_path in tqdm(videos, desc='extract CLIP hidden CLS'):
        key = video_path.stem
        out_path = feat_dir / f'{key}.npz'
        if out_path.exists() and not args.overwrite:
            rows.append(existing_row(key, video_path, out_path, args.stride, layer_ids))
            continue
        try:
            n_frames, fps = get_video_len(str(video_path))
            if n_frames <= 0:
                raise RuntimeError(f'non-positive frame count: {n_frames}')
            if args.drop_last_incomplete:
                usable_frames = (n_frames // args.stride) * args.stride
                if usable_frames <= 0:
                    raise RuntimeError(f'not enough frames for one full stride={args.stride} snippet: num_frames={n_frames}')
                indices = list(range(0, usable_frames, args.stride))
            else:
                indices = list(range(0, n_frames, args.stride))
            chunks = []
            for start in range(0, len(indices), args.batch_size):
                batch_idx = indices[start:start + args.batch_size]
                frames = load_video_frames(str(video_path), batch_idx)
                if len(frames) != len(batch_idx):
                    raise RuntimeError(f'failed to load requested frames: got {len(frames)}/{len(batch_idx)}')
                images = torch.stack([preprocess(img) for img in frames], dim=0).to(device)
                with torch.no_grad():
                    chunks.append(encode_hidden_cls(model, images, layer_ids))
            if not chunks:
                raise RuntimeError('no frames/chunks extracted')
            hidden = np.concatenate(chunks, axis=0).astype(np.float16)
            np.savez_compressed(out_path, hidden=hidden, frame_indices=np.asarray(indices, dtype=np.int64), fps=fps, num_frames=n_frames, stride=args.stride, layers=np.asarray(layer_ids, dtype=np.int64), drop_last_incomplete=np.asarray(bool(args.drop_last_incomplete)))
            rows.append([key, str(video_path), str(out_path), args.stride, n_frames, fps, ','.join(map(str, layer_ids))])
        except Exception as exc:
            failed_rows.append([key, str(video_path), repr(exc)])
            print(f'[WARN] skip failed video {key}: {exc}', flush=True)
            continue

    manifest = Path(args.out_dir) / 'manifest.csv'
    with open(manifest, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'video_path', 'hidden_path', 'stride', 'num_frames', 'fps', 'layers'])
        writer.writerows(rows)
    print(f'wrote {manifest} with {len(rows)} rows')
    failed_csv = Path(args.out_dir) / 'failed.csv'
    if failed_rows:
        with open(failed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'video_path', 'error'])
            writer.writerows(failed_rows)
        print(f'wrote {failed_csv} with {len(failed_rows)} failed rows')


if __name__ == '__main__':
    main()

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--selection",required=True);p.add_argument("--out-dir",required=True);a=p.parse_args()
    out=Path(a.out_dir);scores=out/"scores";scores.mkdir(parents=True,exist_ok=True)
    selected=json.loads(Path(a.selection).read_text(encoding="utf-8"))["neurons"]
    layers=np.asarray([x["layer"]-1 for x in selected]);dims=np.asarray([x["dimension"] for x in selected]);rows=[]
    frame=pd.read_csv(a.manifest)
    for row in tqdm(frame.itertuples(index=False),total=len(frame),desc="cache selected CLS neurons"):
        target=scores/f"{row.key}.npy"
        if not target.exists():
            hidden=np.load(str(row.hidden_path))["hidden"];np.save(target,hidden[:,layers,dims].astype(np.float16))
        rows.append({"key":str(row.key),"label":str(row.label),"binary_label":int(row.binary_label),"selected_path":str(target)})
    pd.DataFrame(rows).to_csv(out/"selected_manifest.csv",index=False);print(f"cached {len(rows)} videos with {len(selected)} selected neurons",flush=True)

if __name__=="__main__":main()

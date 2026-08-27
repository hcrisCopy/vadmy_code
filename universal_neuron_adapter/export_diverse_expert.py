from __future__ import annotations
import argparse,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch
from tqdm import tqdm
from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.model import SparseNeuronExpert

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--expert-model",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--device",default="cuda");p.add_argument("--clean",action="store_true");a=p.parse_args();out=Path(a.out_dir)
    if a.clean and out.exists():shutil.rmtree(out)
    scores=out/"scores";scores.mkdir(parents=True,exist_ok=True);device=torch.device(a.device if torch.cuda.is_available() else "cpu");ck=torch.load(a.expert_model,map_location="cpu",weights_only=False);model=SparseNeuronExpert(**ck["config"]);model.load_state_dict(ck["model_state_dict"]);model.to(device).eval();rows=[];frame=pd.read_csv(a.manifest)
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False),total=len(frame),desc="export diverse neuron evidence"):
            target=scores/f"{row.key}.npy"
            if not target.exists():
                hidden=torch.from_numpy(load_hidden_array(str(row.hidden_path))).unsqueeze(0).to(device);length=torch.tensor([hidden.shape[1]],device=device);prob=torch.sigmoid(model(hidden,length))[0,:int(length.item())].cpu().numpy().astype(np.float32);np.save(target,prob)
            rows.append({"key":str(row.key),"expert2_score_path":str(target)})
    pd.DataFrame(rows).to_csv(out/"expert2_scores.csv",index=False);print(f"wrote {len(rows)} diverse expert curves",flush=True)
if __name__=="__main__":main()

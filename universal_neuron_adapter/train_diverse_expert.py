from __future__ import annotations
import argparse,json,random,shutil
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from universal_neuron_adapter.data import HiddenVideoDataset,collate_hidden
from universal_neuron_adapter.model import SparseNeuronExpert,expert_mil_loss,topk_bag

def evaluate(model,loader,device):
    model.eval();targets=[];scores=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc="diverse expert validation",leave=False):
            logits=model(batch["hidden"].to(device),batch["lengths"].to(device));scores.extend(topk_bag(torch.sigmoid(logits),batch["lengths"].to(device)).cpu().tolist());targets.extend(batch["labels"].tolist())
    return .5*(roc_auc_score(targets,scores)+average_precision_score(targets,scores))

def main():
    p=argparse.ArgumentParser()
    for name in ["train-manifest","val-manifest","out-dir"]:p.add_argument("--"+name,required=True)
    p.add_argument("--active-per-layer",type=int,default=64);p.add_argument("--temporal-width",type=int,default=64);p.add_argument("--max-epoch",type=int,default=20);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--sparsity-weight",type=float,default=1e-3);p.add_argument("--maximum-length",type=int,default=256);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--seed",type=int,default=3407);p.add_argument("--device",default="cuda");p.add_argument("--resume",action="store_true");p.add_argument("--clean",action="store_true");a=p.parse_args();out=Path(a.out_dir)
    if a.clean and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    train=DataLoader(HiddenVideoDataset(a.train_manifest,a.maximum_length),batch_size=a.batch_size,shuffle=True,drop_last=True,num_workers=a.num_workers,collate_fn=collate_hidden);val=DataLoader(HiddenVideoDataset(a.val_manifest,a.maximum_length),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=collate_hidden);model=SparseNeuronExpert(a.active_per_layer,a.temporal_width).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.max_epoch);last=out/"checkpoint_last.pth";best_path=out/"expert_best.pth";start=0;best=-1
    if a.resume and last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False);model.load_state_dict(ck["model_state_dict"]);opt.load_state_dict(ck["optimizer_state_dict"]);scheduler.load_state_dict(ck["scheduler_state_dict"]);start=ck["epoch"]+1;best=ck["best_metric"]
    for epoch in range(start,a.max_epoch):
        model.train();total=0
        for batch in tqdm(train,desc=f"diverse expert {epoch+1}/{a.max_epoch}"):
            logits=model(batch["hidden"].to(device),batch["lengths"].to(device));loss=expert_mil_loss(logits,batch["labels"].to(device),batch["lengths"].to(device))+a.sparsity_weight*model.sparsity_loss();opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss)
        scheduler.step();metric=evaluate(model,val,device);payload={"epoch":epoch,"best_metric":max(best,metric),"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"config":{"active_per_layer":a.active_per_layer,"temporal_width":a.temporal_width}};torch.save(payload,last)
        if metric>best:
            best=metric;torch.save(payload,best_path)
            (out/"selected_neurons.json").write_text(json.dumps(model.selection(),indent=2),encoding="utf-8")
        with (out/"history.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps({"epoch":epoch+1,"loss":total/len(train),"validation_metric":metric})+"\n")
        print(json.dumps({"epoch":epoch+1,"loss":total/len(train),"validation_metric":metric}),flush=True)
if __name__=="__main__":main()

from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from universal_neuron_adapter.data import normalize_selected_layers,resample_matrix
from universal_neuron_adapter.model import SemanticNeuronProbe

def tokens(label:str,binary:int,dataset:str)->list[str]:
    if not binary:return []
    return sorted({x for x in label.split("-") if x not in {"0","A"}}) if dataset=="xd" else [label]

class Videos(Dataset):
    def __init__(self,manifest:str,keys:str,dataset:str,classes:list[str],limit:int):
        f=pd.read_csv(manifest);keep=set(pd.read_csv(keys).key.astype(str));self.f=f[f.key.astype(str).isin(keep)].reset_index(drop=True);self.dataset=dataset;self.classes=classes;self.limit=limit
    def __len__(self):return len(self.f)
    def __getitem__(self,i):
        r=self.f.iloc[i];x=np.load(str(r.selected_path)).astype(np.float32);x=normalize_selected_layers(resample_matrix(x,min(len(x),self.limit)));y=np.zeros(len(self.classes),np.float32)
        for c in tokens(str(r.label),int(r.binary_label),self.dataset):y[self.classes.index(c)]=1
        return x,y,float(r.binary_label)

def collate(batch):
    lengths=torch.tensor([len(x[0]) for x in batch]);steps=int(lengths.max());width=batch[0][0].shape[1];x=torch.zeros(len(batch),steps,width)
    for i,(v,_,_) in enumerate(batch):x[i,:len(v)]=torch.from_numpy(v)
    return x,torch.from_numpy(np.stack([r[1] for r in batch])),torch.tensor([r[2] for r in batch]),lengths

def loss_fn(logits,target,binary,lengths,pos_weight):
    bags=[]
    for row,n in zip(logits,lengths):
        n=int(n);k=max(1,n//16+1);bags.append(torch.sigmoid(row[:n]).topk(k,dim=0).values.mean(0))
    bag=torch.stack(bags);weight=1+target*(pos_weight-1);bag_loss=F.binary_cross_entropy(bag,target,weight=weight)
    normal=0.0;count=0
    for row,n,label in zip(logits,lengths,binary):
        if label<.5:normal=normal+F.binary_cross_entropy_with_logits(row[:int(n)],torch.zeros_like(row[:int(n)]));count+=1
    return bag_loss+.5*normal/max(1,count)

def main():
    p=argparse.ArgumentParser();
    for name in ["selected-manifest","train-keys","val-keys","dataset","out-dir"]:p.add_argument("--"+name,required=True)
    p.add_argument("--max-epoch",type=int,default=20);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--maximum-length",type=int,default=256);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--seed",type=int,default=3407);p.add_argument("--device",default="cuda");p.add_argument("--resume",action="store_true");a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);frame=pd.read_csv(a.selected_manifest);classes=sorted({c for r in frame.itertuples() for c in tokens(str(r.label),int(r.binary_label),a.dataset)});train=Videos(a.selected_manifest,a.train_keys,a.dataset,classes,a.maximum_length);val=Videos(a.selected_manifest,a.val_keys,a.dataset,classes,a.maximum_length)
    counts=np.zeros(len(classes));
    for r in train.f.itertuples():
        for c in tokens(str(r.label),int(r.binary_label),a.dataset):counts[classes.index(c)]+=1
    pos=torch.from_numpy(np.clip((len(train)-counts)/np.maximum(counts,1),1,20).astype(np.float32));dev=torch.device(a.device if torch.cuda.is_available() else "cpu");model=SemanticNeuronProbe(train[0][0].shape[1],len(classes)).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);last=out/"checkpoint_last.pth";best_path=out/"model_best.pth";start=0;best=1e9
    if a.resume and last.exists():
        ck=torch.load(last,map_location="cpu",weights_only=False);model.load_state_dict(ck["model_state_dict"]);opt.load_state_dict(ck["optimizer_state_dict"]);start=ck["epoch"]+1;best=ck["best_metric"]
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=collate,generator=torch.Generator().manual_seed(a.seed));vl=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=collate);pos=pos.to(dev)
    for epoch in range(start,a.max_epoch):
        model.train();total=0
        for x,y,b,n in tqdm(loader,desc=f"semantic probe {a.dataset} {epoch+1}/{a.max_epoch}"):
            x,y,b,n=x.to(dev),y.to(dev),b.to(dev),n.to(dev);loss=loss_fn(model(x),y,b,n,pos);opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss)
        model.eval();v=0
        with torch.no_grad():
            for x,y,b,n in vl:x,y,b,n=x.to(dev),y.to(dev),b.to(dev),n.to(dev);v+=float(loss_fn(model(x),y,b,n,pos))
        v/=max(1,len(vl));payload={"epoch":epoch,"best_metric":min(best,v),"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"classes":classes,"input_dim":train[0][0].shape[1],"dataset":a.dataset};torch.save(payload,last)
        if v<best:best=v;torch.save(payload,best_path)
        with (out/"history.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps({"epoch":epoch+1,"loss":total/len(loader),"validation_loss":v})+"\n")
        print(json.dumps({"epoch":epoch+1,"loss":total/len(loader),"validation_loss":v}),flush=True)

if __name__=="__main__":main()

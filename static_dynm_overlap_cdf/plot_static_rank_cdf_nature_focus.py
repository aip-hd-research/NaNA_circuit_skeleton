"""Focused, journal-style Static-rank CDF with a full-range inset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

MODELS=("openai-community/gpt2","Qwen/Qwen3-0.6B","meta-llama/Llama-3.2-1B","mistralai/Mistral-7B-v0.3")
LABELS={"openai-community/gpt2":"GPT-2","Qwen/Qwen3-0.6B":"Qwen3-0.6B","meta-llama/Llama-3.2-1B":"LLaMA3-1B","mistralai/Mistral-7B-v0.3":"Mistral v0.3"}
COLORS=("#355F8D","#2A9D8F","#D99A16","#C84A32")

def args():
 p=argparse.ArgumentParser(); p.add_argument("--curves_json",required=True); p.add_argument("--output_dir",required=True)
 p.add_argument("--models",nargs="+",default=list(MODELS)); p.add_argument("--dynamic_percents",nargs="+",type=float,default=[.001,.01,.1,1]); p.add_argument("--dpi",type=int,default=600); return p.parse_args()

def setup():
 plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"axes.labelsize":8,"axes.titlesize":8.5,"axes.linewidth":.65,"xtick.labelsize":7,"ytick.labelsize":7,"xtick.direction":"out","ytick.direction":"out","legend.fontsize":7,"legend.frameon":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

def data(entry, percents):
 n=int(entry["n_subspaces"]); x=np.r_[0.,np.arange(1,n+1)/n*100]; out=[]
 for p in percents:
  k=next((k for k,v in entry["curves"].items() if np.isclose(float(v["dynamic_k_percent"]),p)),None)
  if k is None: raise ValueError(f"Dynamic Top {p:g}% missing")
  out.append((p,np.r_[0.,np.asarray(entry["curves"][k]["observed_cdf"],float)]))
 return x,out

def panel(ax, model, entry, percents, add_inset=True):
 x,curves=data(entry,percents); handles=[]
 for i,(p,y) in enumerate(curves):
  h,=ax.step(x,y,where="post",lw=1.35,color=COLORS[i],label=f"DC Top {p:g}%"); handles.append(h)
 null,=ax.plot([0,25],[0,.25],color="#777",lw=.8,ls=(0,(3,2)),label="Uniform null"); handles.append(null)
 ax.set(xlim=(0,25),ylim=(0,1.015),xticks=[0,5,10,15,20,25],yticks=[0,.25,.5,.75,1]); ax.set_title(LABELS.get(model,model.rsplit('/',1)[-1]),pad=4,fontweight="semibold")
 ax.spines[["top","right"]].set_visible(False); ax.set_xlabel("Static Top-k threshold (%)")
 ax.grid(axis="y",color="#E8E8E8",lw=.4,zorder=0)
 if add_inset:
  ia=inset_axes(ax,width="34%",height="31%",loc="lower right",borderpad=1.0)
  for i,(_,y) in enumerate(curves): ia.step(x,y,where="post",lw=.75,color=COLORS[i])
  ia.plot([0,100],[0,1],color="#777",lw=.6,ls=(0,(3,2))); ia.set(xlim=(0,100),ylim=(0,1.02),xticks=[0,50,100],yticks=[0,1]); ia.tick_params(labelsize=5,length=2,pad=1)
  ia.spines[["top","right"]].set_visible(False); ia.set_title("Full range",fontsize=5.5,pad=1.5)
 return handles

def save(fig,stem,dpi):
 for s in ("pdf","png","svg"): fig.savefig(Path(f"{stem}.{s}"),dpi=dpi,bbox_inches="tight")
 plt.close(fig)

def main():
 a=args(); setup(); models=json.load(open(a.curves_json,encoding="utf-8"))["models"]; out=Path(a.output_dir); (out/"individual").mkdir(parents=True,exist_ok=True); mm=1/25.4
 for m in a.models:
  fig,ax=plt.subplots(figsize=(89*mm,72*mm),constrained_layout=True); h=panel(ax,m,models[m],a.dynamic_percents); ax.set_ylabel("Cumulative fraction"); fig.legend(handles=h,loc="outside upper center",ncol=3,handlelength=1.8,columnspacing=1.0)
  save(fig,out/"individual"/(m.replace('/','_')+"_cdf_focus"),a.dpi)
 fig,axs=plt.subplots(2,2,figsize=(180*mm,125*mm),sharex=True,sharey=True,constrained_layout=True)
 for ax,m in zip(axs.ravel(),a.models): h=panel(ax,m,models[m],a.dynamic_percents)
 axs[0,0].set_ylabel("Cumulative fraction"); axs[1,0].set_ylabel("Cumulative fraction"); fig.legend(handles=h,loc="outside upper center",ncol=5,handlelength=2,columnspacing=1.4)
 save(fig,out/"static_rank_cdf_four_models_focus",a.dpi); print(f"[done] wrote {out}")
if __name__=="__main__": main()

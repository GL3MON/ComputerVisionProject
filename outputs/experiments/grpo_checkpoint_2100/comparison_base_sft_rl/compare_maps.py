import os, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import medmnist
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

torch.set_num_threads(2)
ROOT=Path('/home/harishr/Med/outputs/experiments/grpo_checkpoint_2100')
OUT=ROOT/'comparison_base_sft_rl'
MAPS=OUT/'maps'
RAW=OUT/'raw'
MAPS.mkdir(exist_ok=True); RAW.mkdir(exist_ok=True)
CHECKPOINT='/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100'
SFT='GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged'
MODELS={'base':'Qwen/Qwen3-VL-2B-Instruct','sft':SFT,'rl':SFT}
CLASSIFIERS={'base':OUT/'base/classifier/linear_classifier_best.pt','sft':OUT/'sft/classifier/linear_classifier_best.pt','rl':ROOT/'classifier/linear_classifier_best.pt'}
SAMPLES=json.loads((ROOT/'attribution/summary.json').read_text())['samples']
processor=AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
dataset=medmnist.PneumoniaMNIST(split='val',size=224,download=False,as_rgb=True)
assert torch.cuda.is_available()
device='cuda'; dtype=torch.bfloat16
labels=['Normal','Pneumonia']
records=[]
for name,model_id in MODELS.items():
    print('Loading', name,flush=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_id,dtype=dtype,attn_implementation='sdpa',local_files_only=True)
    if name=='rl':
        model=PeftModel.from_pretrained(model,CHECKPOINT,is_trainable=False).merge_and_unload()
    model=model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    saved=torch.load(CLASSIFIERS[name],map_location='cpu',weights_only=True)
    classifier=nn.Linear(saved['input_dim'],saved['num_classes']).to(device).eval()
    classifier.load_state_dict({k.removeprefix('classifier.'):v for k,v in saved['model_state_dict'].items()})
    for p in classifier.parameters(): p.requires_grad_(False)
    for idx in SAMPLES:
        image,label=dataset[idx]; label=int(np.asarray(label).squeeze()); image=image.convert('RGB')
        inputs=processor.image_processor(images=[image],return_tensors='pt').to(device)
        pixel=inputs['pixel_values'].detach().clone().requires_grad_(True)
        features=model.get_image_features(pixel_values=pixel,image_grid_thw=inputs['image_grid_thw']).pooler_output[0]
        logits=classifier(features.mean(0).float().unsqueeze(0))
        probabilities=logits.softmax(-1).detach().cpu().numpy()[0]
        prediction=int(probabilities.argmax())
        gradient=torch.autograd.grad(logits[0,label],pixel)[0]
        values=(gradient*pixel).abs().mean(1)
        t,h,w=map(int,inputs['image_grid_thw'][0].tolist()); merge=processor.image_processor.merge_size
        values=values.reshape(t,h//merge,w//merge,merge,merge).permute(0,1,3,2,4).reshape(t,h,w).mean(0)
        raw=values.detach().float().cpu().numpy()
        assert np.isfinite(raw).all() and raw.sum()>0
        np.save(RAW/f'{name}_sample_{idx:04d}.npy',raw)
        mass=raw.flatten()/raw.sum()
        entropy=float(-(mass*np.log(mass+1e-12)).sum()/np.log(len(mass)))
        n=max(1,int(len(mass)*.1))
        records.append({'model':name,'sample':idx,'true_label':label,'prediction':prediction,'correct':int(label==prediction),
            'predicted_confidence':float(probabilities[prediction]),'target_class_confidence':float(probabilities[label]),
            'attribution_entropy':entropy,'top_10_percent_mass':float(np.sort(mass)[-n:].sum()),'grid_h':h,'grid_w':w})
        print(name,'sample',idx,'complete',flush=True)
        del inputs,pixel,features,logits,gradient,values
    del model,classifier
    gc.collect(); torch.cuda.empty_cache()
rows=pd.DataFrame(records)
rows.to_csv(OUT/'attribution_metrics.csv',index=False)

def norm(x): return x/max(float(x.max()),1e-12)
def resized(x,size):
    return np.asarray(Image.fromarray(x.astype(np.float32),mode='F').resize(size,Image.Resampling.BILINEAR))
def panels(axes,idx):
    image,label=dataset[idx]; image=image.convert('RGB'); label=int(np.asarray(label).squeeze())
    axes[0].imshow(image); axes[0].set_title(f'Sample {idx} | True: {labels[label]}')
    maps={n:np.load(RAW/f'{n}_sample_{idx:04d}.npy') for n in MODELS}
    assert len({v.shape for v in maps.values()})==1
    for ax,name,title in zip(axes[1:4],MODELS,['Base','Medical SFT','GRPO checkpoint-2100']):
        row=rows[(rows.model==name)&(rows['sample']==idx)].iloc[0]
        ax.imshow(image)
        ax.imshow(resized(norm(maps[name]),image.size),cmap='inferno',alpha=.55,vmin=0,vmax=1)
        ax.set_title(f'{title}\nPred: {labels[int(row.prediction)]} ({row.predicted_confidence:.3f})')
    delta=resized(norm(maps['rl'])-norm(maps['sft']),image.size)
    im=axes[4].imshow(delta,cmap='RdBu_r',vmin=-1,vmax=1)
    axes[4].set_title('RL minus SFT\nRed: RL | Blue: SFT')
    for ax in axes: ax.axis('off')
    return im
for idx in SAMPLES:
    fig,axes=plt.subplots(1,5,figsize=(20,4.7))
    im=panels(axes,idx)
    fig.colorbar(im,ax=axes[4],fraction=.046,pad=.02)
    _,label=dataset[idx]; label=int(np.asarray(label).squeeze())
    fig.suptitle(f'Attribution target for all models: {labels[label]} | Absolute gradient × input',fontsize=13)
    fig.text(.5,.015,'Overlays normalized to each map’s peak (0–1); differences show relative spatial patterns.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.01,right=.985,bottom=.085,top=.82,wspace=.08)
    fig.savefig(MAPS/f'comparison_sample_{idx:04d}.png',dpi=150)
    plt.close(fig)
    print('Rendered',idx,flush=True)
# Include a prediction disagreement and representative correct predictions in the overview.
predictions=rows.pivot(index='sample',columns='model',values='prediction')
disagreements=[i for i in SAMPLES if predictions.loc[i].nunique()>1]
selected=(disagreements+[i for i in SAMPLES if i not in disagreements])[:5]
fig,axes=plt.subplots(5,5,figsize=(20,20))
for ax,idx in zip(axes,selected): panels(ax,idx)
fig.suptitle('Base / Medical SFT / GRPO checkpoint-2100 | Same-class attribution comparison',fontsize=18)
fig.text(.5,.012,'Each row targets the true class. Heatmaps: per-map peak normalization. Difference: RL minus SFT.',ha='center',fontsize=13)
fig.subplots_adjust(left=.01,right=.99,bottom=.04,top=.95,hspace=.28,wspace=.08)
fig.savefig(OUT/'comparison_overview.png',dpi=120); plt.close(fig)
performance={n:json.loads((OUT/n/'classifier/metrics.json').read_text()) for n in ['base','sft']}
performance['rl']=json.loads((ROOT/'classifier/metrics.json').read_text())
summary={'models':MODELS,'rl_checkpoint':CHECKPOINT,'processor':CHECKPOINT,'classifiers':{n:str(p) for n,p in CLASSIFIERS.items()},
    'samples':SAMPLES,'split':'val','device':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'attribution_target':'true class label for all three models','method':'absolute gradient times normalized input; mean patch elements; restored Qwen spatial patch order',
    'visualization':'zero-to-peak normalization independently per map; differences compare relative spatial patterns',
    'test_metrics':{n:{k:m[k] for k in ['best_epoch','best_val_accuracy','test_accuracy','test_weighted_f1','test_auroc']} for n,m in performance.items()},
    'sample_metrics':rows.groupby('model')[['correct','predicted_confidence','attribution_entropy','top_10_percent_mass']].mean().to_dict('index'),
    'overview_samples':selected,'prediction_disagreement_samples':disagreements}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
report=['# Base / SFT / RL comparison','',
    '25 matched PneumoniaMNIST validation images. Same checkpoint processor, mean vision-token pooling, seed 3407, and 20-epoch frozen-encoder linear-probe training recipe for all three models. RL uses the previously trained classifier; base and SFT probes were freshly trained with matching settings.',
    '', 'Attribution targets the true class on every image for all models. Titles report each model’s predicted class and softmax confidence. Patches are restored to spatial order. Each heatmap is scaled from zero to its own peak; the RL-minus-SFT panel compares relative spatial patterns, not absolute model sensitivity.',
    '', '| Model | Validation accuracy | Test accuracy | Weighted F1 | AUROC |','|---|---:|---:|---:|---:|']
for n,m in performance.items():
    report.append(f"| {n.upper()} | {m['best_val_accuracy']:.2%} | {m['test_accuracy']:.2%} | {m['test_weighted_f1']:.4f} | {m['test_auroc']:.4f} |")
report+=['','![Overview](comparison_overview.png)','',
    'Individual five-panel figures are in `maps/`; raw maps for all 75 model/image pairs are in `raw/`. Full per-image metrics are in `attribution_metrics.csv`; settings and metrics are in `summary.json`.',
    '', 'These are classifier attribution maps, not attention weights or explanations of generated answers. Each model has its own trained linear probe, so differences reflect both encoder representation and classifier weights. The 25-image subset was selected before these comparisons and is not the full validation set.']
(OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
assert len(list(MAPS.glob('comparison_sample_*.png')))==25
assert len(list(RAW.glob('*.npy')))==75
print(json.dumps(summary,indent=2),flush=True)

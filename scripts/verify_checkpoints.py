"""Re-evaluate saved classifier and run a seeded counterfactual replication."""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.networks.schedulers import DDIMScheduler
R=Path(__file__).resolve().parents[1];OUT=R / "verification_run";OUT.mkdir(parents=True,exist_ok=True)
os.chdir(R);sys.path.insert(0,str(R/'src'))
from data import OrganSlices,ORGANS
from counterfactual import load_models,generate,metrics
from train_classifier import evaluate
from evaluate import nearest_train_distance

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4);dev='cuda' if torch.cuda.is_available() else 'cpu';unet,clf,total=load_models(dev);ds=OrganSlices('test');dl=DataLoader(ds,batch_size=256,shuffle=False,num_workers=0)
acc,perclass=evaluate(clf,dl,dev)
result={'classifier_recomputed':{'test_acc':acc,'per_class_test_acc':perclass},'parameters':{'classifier':sum(x.numel() for x in clf.parameters()),'diffusion_unet':sum(x.numel() for x in unet.parameters())},'checkpoint_sha256':{p.name:digest(p) for p in (R/'weights').glob('*.pt')},'source_sha256':{p.name:digest(p) for p in (R/'src').glob('*.py')}}
print('Classifier',result['classifier_recomputed'],flush=True)
rng=np.random.default_rng(0);idx=rng.choice(len(ds),size=256,replace=False);x=torch.stack([ds[i][0] for i in idx]).to(dev);source=torch.tensor([ds[i][1] for i in idx],device=dev)
target=(source+1+torch.arange(len(source),device=dev)%(len(ORGANS)-1))%len(ORGANS);target=torch.where(target==source,(target+1)%len(ORGANS),target)
with torch.inference_mode():pred=clf(x).argmax(1)
result['subset']={'n':len(idx),'selection_seed':0,'indices':idx.tolist(),'correct_original_predictions':int((pred==source).sum()),'already_target_before_generation':int((pred==target).sum())}
sched=DDIMScheduler(num_train_timesteps=total);sched.set_timesteps(100)
torch.manual_seed(2026);result['replication']={'torch_seed':2026,'ddim_grid_steps':100,'guidance_scale':6.,'sweep':[]};arrays={'test_indices':idx,'source':source.cpu().numpy(),'target':target.cpu().numpy(),'original_pred':pred.cpu().numpy()}
for t in [50,100,150,200,300,400,600]:
 cf=generate(unet,clf,x,target,sched,t,6.,dev)
 with torch.inference_mode():cp=clf(cf).argmax(1)
 m=metrics(x,cf,clf,source,target);m.update({'start_t':t,'executed_steps':sum(int(v)<=t for v in sched.timesteps),'target_success_count':int((cp==target).sum()),'actual_target_flip_count':int(((cp==target)&(pred!=target)).sum()),'correct_original_target_success_count':int(((cp==target)&(pred==source)).sum())})
 result['replication']['sweep'].append(m);arrays[f'counterfactual_pred_t{t}']=cp.cpu().numpy();arrays[f'l1_t{t}']=(cf-x).flatten(1).abs().mean(1).cpu().numpy()
 if t==200:
  torch.save({'x0':x[:8].cpu(),'xcf':cf[:8].cpu(),'source':source[:8].cpu(),'target':target[:8].cpu(),'test_indices':idx[:8]},OUT/'replication_examples.pt')
 print('SWEEP',m,flush=True)
np.savez_compressed(OUT/'replication_per_sample.npz',**arrays)
# Repeat the initial 64-image check; verify_sample_quality.py covers all 1,024.
if (R/'results/generated_samples.npz').exists():
 gen=torch.from_numpy(np.load(R/'results/generated_samples.npz')['images'][:64]);train=OrganSlices('train');tr=torch.from_numpy(train.imgs).unsqueeze(1)
 dg=nearest_train_distance(gen,tr,dev);dr=nearest_train_distance(x.cpu()[:64],tr,dev)
 result['saved_samples_full_training_nn_check']={'n_generated_saved':len(gen),'n_reference_train':len(train),'n_real_comparison':64,'generated_min':float(dg.min()),'generated_median':float(np.median(dg)),'real_min':float(dr.min()),'real_median':float(np.median(dr)),'scope':'initial 64-image pixel-distance check; separate sample-quality verification covers all 1,024; not a patient-level privacy test'}
else:
 result['saved_samples_full_training_nn_check']={'status':'skipped: regenerate samples with src/evaluate.py or use the saved verification record'}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',result['saved_samples_full_training_nn_check'],flush=True)

import torch,numpy as np,json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));OUT=R/'verification_run';OUT.mkdir(exist_ok=True)
from data import OrganSlices
from train_classifier import OrganCNN
from evaluate import features,frechet,nearest_train_distance
torch.set_num_threads(4);dev='cuda' if torch.cuda.is_available() else 'cpu'
gen=torch.from_numpy(np.load(R/'results/generated_samples.npz')['images']);n=len(gen)
clf=OrganCNN().to(dev);clf.load_state_dict(torch.load(R/'weights/classifier.pt',weights_only=True)['state_dict']);clf.eval()
tr=OrganSlices('train');te=OrganSlices('test');rng=np.random.default_rng(0);ti=rng.choice(len(tr),2048,replace=False);ei=rng.choice(len(te),1024,replace=False);a=torch.stack([tr[i][0] for i in ti]);b=torch.stack([te[i][0] for i in ei]);fa=features(clf,a,dev)
out={'sample_n':n,'original_protocol_recomputed':{'frechet_generated':frechet(features(clf,gen,dev),fa),'frechet_real':frechet(features(clf,b,dev),fa)},'full_training_pool':{}}
for mode,refs in [('original_protocol_recomputed',a),('full_training_pool',torch.from_numpy(tr.imgs).unsqueeze(1))]:
 out[mode]['reference_n']=len(refs)
 for key,q in [('generated',gen),('real',b)]:
  dd=nearest_train_distance(q,refs,dev);out[mode][key]={'n':len(q),'median':float(np.median(dd)),'mean':float(dd.mean()),'p5':float(np.percentile(dd,5)),'min':float(dd.min())}
(OUT/'sample_quality_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

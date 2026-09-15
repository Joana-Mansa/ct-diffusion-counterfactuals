"""Render code-derived architecture diagrams and recorded learning curves."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
FIGS=ROOT/'docs/figures';FIGS.mkdir(parents=True,exist_ok=True)
BLUE='#215A86'; GREEN='#237568'; ORANGE='#A75E20'; INK='#203448'; MUTED='#566777'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'architecture-v1','axes.spines.top':False,'axes.spines.right':False})
def canvas(h):
 f,a=plt.subplots(figsize=(13,h));a.set(xlim=(0,13),ylim=(0,h));a.axis('off');f.subplots_adjust(0,0,1,1);return f,a

def box(a,x,y,w,h,title,detail='',color=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.08',lw=1.2,edgecolor=color,facecolor={'#215A86':'#EEF5FB','#237568':'#EDF7F3','#A75E20':'#FFF5E9'}.get(color,'#F5F7FA')))
 a.text(x+w/2,y+h*.68,title,ha='center',va='center',color=color,fontsize=11,weight='bold')
 if detail:a.text(x+w/2,y+h*.29,detail,ha='center',va='center',color=INK,fontsize=10,linespacing=1.4)

def arrow(a,p,q,color=MUTED,style='-'):
 a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=color,linestyle=style))

def save(f,name):
 for ext in ['svg','png']:
  target=FIGS/f'{name}.{ext}'
  f.savefig(target,dpi=170,facecolor='white',metadata={'Date':None} if ext=='svg' else {})
  if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
 plt.close(f)

def diffusion():
 f,a=canvas(8.4)
 a.text(.25,8.0,'Two separately trained models',fontsize=18,weight='bold',color=INK)
 a.text(.25,7.62,'OrganCNN predicts the organ; the diffusion U-Net predicts added noise.',fontsize=12,color=MUTED)
 xs=[.25,2.4,4.55,6.7,8.85,11.0];w=1.72;y=6.3
 labels=[('CT crop','1 × 64 × 64'),('Stage 1','32 × 32 × 32'),('Stage 2','64 × 16 × 16'),('Stage 3','128 × 8 × 8'),('Stage 4','256 × 4 × 4'),('Head','pool → dropout\nlinear → 11 logits')]
 for i,(title,detail) in enumerate(labels):
  box(a,xs[i],y,w,.9,title,detail)
  if i:arrow(a,(xs[i-1]+w+.04,y+.45),(xs[i]-.04,y+.45))
 a.text(.25,5.98,'Each stage: [3 × 3 Conv → BatchNorm → SiLU] twice, then 2 × 2 max pooling.',fontsize=10.5,color=MUTED)
 a.text(.25,5.49,'Diffusion U-Net  |  10,395,905 parameters',fontsize=15,weight='bold',color=GREEN)
 # Encoder/decoder levels show feature resolution before the next scale change.
 left=.65;right=8.05;w=2.2;h=.75
 for yy,c,res in [(4.24,64,64),(2.79,128,32),(1.34,192,16)]:
  box(a,left,yy,w,h,f'Encoder: {c} channels',f'{res} × {res} spatial',GREEN)
  box(a,right,yy,w,h,f'Decoder: {c} channels',f'{res} × {res} spatial',GREEN)
  arrow(a,(left+w+.08,yy+h*.55),(right-.08,yy+h*.55),GREEN,'--')
  a.text(5.4,yy+h*.55+.1,'skip features (concatenated)',ha='center',fontsize=9,color=GREEN)
 for yy in [4.24,2.79]:arrow(a,(left+w/2,yy-.04),(left+w/2,yy-.66),GREEN)
 for yy in [1.34,2.79]:arrow(a,(right+w/2,yy+h+.04),(right+w/2,yy+1.40),GREEN)
 # A separate bottom path represents the bottleneck; skips enter decoder blocks at each scale.
 box(a,4.25,.38,2.7,.58,'Middle: 192 channels','16 × 16; residual + attention',GREEN)
 arrow(a,(left+w/2,1.29),(4.2,.67),GREEN);arrow(a,(7.0,.67),(right+w/2,1.29),GREEN)
 a.text(1.75,5.13,'Input: noisy CT x_t',ha='center',fontsize=10.5,color=INK)
 a.text(5.4,5.13,'Timestep t → embedding',ha='center',fontsize=10.5,color=INK)
 arrow(a,(1.75,5.09),(1.75,5.01),GREEN)
 a.text(5.4,3.78,'Timestep embedding enters\nthe residual blocks',ha='center',fontsize=10.5,color=INK)
 a.text(5.4,1.04,'Self-attention at 16 × 16 resolution',ha='center',fontsize=10,color=GREEN)
 box(a,10.8,4.24,1.85,.75,'Noise estimate','1 × 64 × 64',ORANGE)
 arrow(a,(10.32,4.62),(10.73,4.62),ORANGE)
 a.text(11.65,3.75,'Training loss:\nnoise MSE',ha='center',va='top',fontsize=11,color=ORANGE)
 a.text(.25,.07,'Shapes omit batch size. Encoder: 2 residual blocks/level; decoder: 3/level. Schematic groups skip tensors by resolution.',fontsize=8.8,color=MUTED)
 save(f,'architecture')
 f,a=canvas(3.35);a.text(.25,2.95,'How a counterfactual is generated',fontsize=17,weight='bold',color=INK)
 labels=[('Real CT crop','Add noise to start_t'),('Frozen U-Net','Estimate noise'),('Clean estimate','Predict x_0 from x_t'),('Frozen classifier','Target-class gradient'),('DDIM update','Adjust noise estimate')]
 for i,(t,d) in enumerate(labels):
  x=.25+i*2.55;box(a,x,1.43,2.13,.85,t,d,GREEN if i!=3 else BLUE)
  if i:arrow(a,(x-.37,1.85),(x-.06,1.85))
 a.plot([11.5,11.5,3.85,3.85],[1.37,.98,.98,1.36],color=GREEN,lw=1.2);arrow(a,(3.85,1.09),(3.85,1.37),GREEN)
 a.text(7.5,.68,'Repeat over selected timesteps, then clip output to [-1, 1].',ha='center',fontsize=11,color=INK)
 a.text(.25,.22,'Weights stay fixed. Guidance changes the sampled image; it is not a new training loss.',fontsize=11,color=MUTED)
 save(f,'sampling_workflow')

def ctfm():
 f,a=canvas(6.8);a.text(.25,6.4,'CT-FM encoder + nodule classification head',fontsize=18,weight='bold',color=INK)
 a.text(.25,6.02,'77,763,042 trainable parameters; the same architecture is used for the scratch control.',fontsize=11.5,color=MUTED)
 labels=[('CT patch','1 × 64³'),('Stage 0','32 × 64³\n1 residual block'),('Stage 1','64 × 32³\n2 residual blocks'),('Stage 2','128 × 16³\n2 residual blocks'),('Stage 3','256 × 8³\n4 residual blocks'),('Stage 4','512 × 4³\n4 residual blocks')]
 xs=[.25,2.4,4.55,6.7,8.85,11.0];w=1.72
 for i,(t,d) in enumerate(labels):
  box(a,xs[i],4.68,w,1.02,t,d,GREEN)
  if i:arrow(a,(xs[i-1]+w+.04,5.18),(xs[i]-.04,5.18),GREEN)
 a.text(.25,4.29,'Stem: 3³ convolution. Residual blocks: BatchNorm, ReLU and 3³ convolutions. Downsampling: stride-2 convolutions.',fontsize=10.5,color=MUTED)
 # Continue from the deepest features to the head, without implying use of the original decoder.
 a.plot([11.86,11.86,1.24,1.24],[4.63,3.92,3.92,3.60],color=GREEN,lw=1.2);arrow(a,(1.24,3.85),(1.24,3.59),GREEN)
 labs=[('Global pooling','512 features'),('LayerNorm','512 features'),('Dropout','p = 0.3'),('Linear layer','512 → 2 logits'),('Prediction','softmax → 2 probabilities')]
 for i,(t,d) in enumerate(labs):
  x=.25+2.55*i;box(a,x,2.63,2.13,.91,t,d)
  if i:arrow(a,(x-.37,3.085),(x-.06,3.085))
 box(a,.35,.97,3.55,1.02,'Training: weighted cross-entropy','Computed from logits + rating labels',ORANGE)
 box(a,4.55,.97,3.55,1.02,'Pretrained vs scratch','Same architecture and label subset',GREEN)
 box(a,8.75,.97,3.85,1.02,'After training','Grad-CAM / Zennit maps; head dropout',BLUE)
 a.text(.25,.52,'All encoder and head weights are optimised. Original CT-FM decoder is discarded.',fontsize=11,color=INK)
 a.text(.25,.18,'Shapes omit batch size. Labels come from malignancy ratings; post-training maps remain exploratory.',fontsize=10.5,color=MUTED)
 save(f,'architecture')

def curves():
 diffusion_project='diffusion' in ROOT.name
 if diffusion_project:
  d=json.loads((ROOT/'results/diffusion_training.json').read_text())['history'];c=json.loads((ROOT/'results/classifier.json').read_text())['history'];f,axs=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
  for ax,h,key,title,col in [(axs[0],d,'loss','DDPM: training noise MSE',GREEN),(axs[1],c,'loss','OrganCNN: training cross-entropy',BLUE),(axs[2],c,'val_acc','OrganCNN: validation accuracy',BLUE)]:
   ax.plot([x['epoch'] for x in h],[x[key] for x in h],color=col,lw=2);ax.set_title(title,fontsize=11);ax.set_xlabel('Epoch');ax.grid(alpha=.18)
  axs[0].set_ylabel('Mean squared error');axs[1].set_ylabel('Cross-entropy');axs[2].set_ylabel('Accuracy')
  selected=max(c,key=lambda x:x['val_acc']);axs[2].scatter(selected['epoch'],selected['val_acc'],color=ORANGE,zorder=3,label=f"Selected epoch {selected['epoch']}");axs[2].legend(fontsize=9)
 else:
  runs=[x for x in json.loads((ROOT/'results/ablation.json').read_text())['runs'] if x['fraction']==1.0];f,axs=plt.subplots(1,2,figsize=(10,3.2),layout='constrained')
  for run in runs:
   h=run['history'];label='CT-FM' if run['pretrained'] else 'Scratch';color=BLUE if run['pretrained'] else ORANGE
   for ax,key in [(axs[0],'loss'),(axs[1],'auc')]:ax.plot([x['epoch'] for x in h],[x[key] for x in h],label=label,color=color,lw=1.8)
   best=max(h,key=lambda x:x['auc']);axs[1].scatter(best['epoch'],best['auc'],color=color,zorder=4,s=40)
  axs[0].set(title='Training: class-weighted cross-entropy',ylabel='Logged mean batch loss');axs[1].set(title='Validation: ROC AUC (dots select checkpoints)',ylabel='ROC AUC')
  for ax in axs:ax.set_xlabel('Epoch');ax.legend(fontsize=9);ax.grid(alpha=.18)
 save(f,'learning_curves')

if __name__=='__main__':
 if 'diffusion' in ROOT.name:diffusion()
 else:ctfm()
 curves()

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parents[1]
a=json.loads((D/'results/counterfactual_sweep.json').read_text())['sweep'];b=json.loads((D/'results/verification.json').read_text())['replication']['sweep']
fig,axs=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
for s,label in [(a,'Original run'),(b,'Repeat, seed 2026')]:
 axs[0].plot([x['start_t'] for x in s],[100*x['validity'] for x in s],'o-',label=label)
 axs[1].plot([x['start_t'] for x in s],[x['l1'] for x in s],'o-',label=label)
for ax in axs:ax.set_xlabel('Starting noise level');ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
axs[0].set_ylabel('Target success (%)');axs[0].legend(frameon=False);axs[1].set_ylabel('Mean absolute pixel change');fig.savefig(D/'docs/figures/replication.png',dpi=170);plt.close(fig)

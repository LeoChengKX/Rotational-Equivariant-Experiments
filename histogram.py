import torch
import matplotlib.pyplot as plt
import os.path as path
import seaborn as sns
import matplotlib.colors as mcolors

blocks = 8
PATH = f'histograms/blocks{blocks}'

prefix = 'checkpoint_'
ckpt_name = ['0', '100', '200', '400', '600', '1000', '2000', '3000']
ckpts = []

ckpt_weights = []
cpkt_cp = []

for name in ckpt_name:
    p = path.join(PATH, prefix + name + '.ckpt')
    ckpt = torch.load(p)['model']
    weights = []
    cp_trans = []
    for k in ckpt:
        if 'MLP_blocks' in k:
            weights.append(ckpt[k])
        if 'cparams_transform' in k:
            cp_trans.append(ckpt[k])
    weights = torch.stack(weights).flatten().detach().cpu()
    cp_trans = torch.stack(cp_trans).flatten().detach().cpu()
    ckpt_weights.append(weights)
    cpkt_cp.append(cp_trans)
    

cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["lightblue", "darkred"])

for i in range(len(ckpt_name)):
    sns.kdeplot(ckpt_weights[i], fill=True, clip=(-0.005, 0.005), bw_adjust=0.01, label=ckpt_name[i], alpha=.2, color=cmap(i / len(ckpt_name)))

# fig.savefig("histograms/test.png")
plt.xticks([])  # Remove x-axis tick labels
plt.yticks([])  # Remove y-axis tick labels

# Disable x and y axis labels
plt.xlabel('')  # Remove x-axis label
plt.ylabel('')  # Remove y-axis label
plt.legend()
plt.legend(fontsize=14)
plt.savefig(f"histograms/blocks{blocks}.png")

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn

from mlp2cnn_v6 import MLP
from ml_collections import ConfigDict

def cnn_dim_out(in_size, ker, stride, padding):
    return math.floor((in_size - ker + 2 * padding)/stride)+1

path = 'mlpmixer_so2.ckpt'
cfg = ConfigDict()
cfg.name = 'mlp2cnn_v6'
cfg.root_dir = "/home/marko/steerable_equivariant_experiments"
cfg.log_dir = f"{cfg.root_dir}/logs/"
cfg.seed = 12315
cfg.timesteps = 200
cfg.max_iters = 80000
cfg.grad_norm_clip = 1.0
cfg.lr = 2e-4
cfg.bs = 2000
cfg.img_dim = sz = 7
cfg.stride = sd = 1
cfg.kernel_size = ks = 5

cfg.in_ch = ic = 4
cfg.out_ch = oc = 4

num_bands = 4
cfg.out_dim = od = cnn_dim_out(sz, ks, sd, 0)
cfg.input_fkwargs = {
    "index_dims": (sz,sz), 
    "max_resolution": (sz,sz),
    "concat_pos": True,
    "num_bands": num_bands,
    "sine_only": False,
}
cfg.query_fkwargs = {
    "index_dims": (od,od), 
    "max_resolution": (od,od),
    "concat_pos": True,
    "num_bands": num_bands,
    "sine_only": False,
}
num_dims = len(cfg.input_fkwargs.max_resolution)
cfg.f_size = (cfg.input_fkwargs.num_bands * num_dims) * 2 + num_dims

cfg.hidden_size = hs = 24
cfg.channels = ch = cfg.f_size+ic

cfg.mixer1_cfg = mixer1 = ConfigDict()

mixer1.image_size=(sz**2+od**2+ks//2+1,1)
mixer1.channels=ch
mixer1.patch_size=1
mixer1.dim=hs
mixer1.depth=2

cfg.mixer2_cfg = mixer2 = ConfigDict()
mixer2.image_size=(sz**2+od**2+ks//2+1,1)
mixer2.channels=ch+ic
mixer2.patch_size=1
mixer2.dim=hs
mixer2.depth=2

model = MLP(cfg).cuda()
ckpt = torch.load("mlpmixer_so2.ckpt")
model.load_state_dict(ckpt['model'], strict=False)

char = torch.tensor([[0, 0, 0.6, 1, 1, 1, 0.5],
                          [0, 0, 0.6, 1, 1, 1, 0.8],
                          [0, 0, 0.5, 1, 0, 0, 0],
                          [0, 0, 0.5, 1, 1, 1, 0.2],
                          [0, 0, 0.4, 1, 0, 0, 0],
                          [0, 0, 0.4, 1, 0, 0, 0],
                          [0, 0, 0.3, 1, 0, 0, 0], ])

img = torch.stack([char, char, char, char])
img = img.unsqueeze(0).cuda()

mlp = nn.Linear(7*7, 3*3, bias=False).cuda()

# plt.imshow(img[0][0].cpu(), cmap='gray')
# plt.savefig("test.png")
k = torch.rand(48).cuda()
mlp_out, cnn_out = model.visualization(img, k)

mlp_out = mlp_out.squeeze().T.reshape(4, 3, 3)
cnn_out = cnn_out.squeeze()
f, axarr = plt.subplots(3, 4)
axarr[0, 0].imshow(char, cmap='gray')
axarr[0, 0].axis(False)
axarr[0, 0].set_title("Original")

axarr[0, 1].imshow(cnn_out[0].detach().cpu(), cmap='gray')
axarr[0, 1].axis(False)
axarr[0, 1].set_title("S-CNN")

axarr[0, 2].imshow(mlp_out[0].detach().cpu(), cmap='gray')
axarr[0, 2].axis(False)
axarr[0, 2].set_title("mlpmixer2scnn")

axarr[0, 3].imshow(mlp(img[0][0].flatten()).reshape(3, 3).detach().cpu(), cmap='gray')
axarr[0, 3].axis(False)
axarr[0, 3].set_title("Random MLP")

img = F.rotate(img, 90).cuda()
mlp_out, cnn_out = model.visualization(img, k)

mlp_out = mlp_out.squeeze().T.reshape(4, 3, 3)
cnn_out = cnn_out.squeeze()

axarr[1, 0].imshow(img[0][0].detach().cpu(), cmap='gray')
axarr[1, 0].axis(False)

axarr[1, 1].imshow(cnn_out[0].detach().cpu(), cmap='gray')
axarr[1, 1].axis(False)

axarr[1, 2].imshow(mlp_out[0].detach().cpu(), cmap='gray')
axarr[1, 2].axis(False)

axarr[1, 3].imshow(mlp(img[0][0].flatten()).reshape(3, 3).detach().cpu(), cmap='gray')
axarr[1, 3].axis(False)

img = F.rotate(img, 90).cuda()
mlp_out, cnn_out = model.visualization(img, k)

mlp_out = mlp_out.squeeze().T.reshape(4, 3, 3)
cnn_out = cnn_out.squeeze()

axarr[2, 0].imshow(img[0][0].detach().cpu(), cmap='gray')
axarr[2, 0].axis(False)

axarr[2, 1].imshow(cnn_out[0].detach().cpu(), cmap='gray')
axarr[2, 1].axis(False)

axarr[2, 2].imshow(mlp_out[0].detach().cpu(), cmap='gray')
axarr[2, 2].axis(False)

axarr[2, 3].imshow(mlp(img[0][0].flatten()).reshape(3, 3).detach().cpu(), cmap='gray')
axarr[2, 3].axis(False)

plt.subplots_adjust(hspace=0.1, wspace=0.1)
f.savefig("MixerVisualization.png")

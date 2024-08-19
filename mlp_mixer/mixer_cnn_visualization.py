import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn

from mlp2cnn_v5v2 import MLP, cfg
from ml_collections import ConfigDict

def cnn_dim_out(in_size, ker, stride, padding):
    return math.floor((in_size - ker + 2 * padding)/stride)+1


model = MLP(cfg).cuda()
ckpt = torch.load("mlpmixer2cnn_checkpoint.ckpt")
model.load_state_dict(ckpt['model'], strict=False)

char = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0.3, 0.6, 1, 0, 0, 0],
                          [0, 1, 0.5, 1, 0, 0, 0],
                          [0, 1, 0.7, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], ])

img = torch.stack([char, char, char, char])
img = img.unsqueeze(0).cuda()

mlp = nn.Linear(7*7, 5*5, bias=False).cuda()

# plt.imshow(img[0][0].cpu(), cmap='gray')
# plt.savefig("test.png")
k = torch.rand(cfg.out_ch, cfg.in_ch, 3, 3).cuda()
mlp_out, cnn_out = model.visualization(img, k)
print(mlp_out.shape, cnn_out.shape)

mlp_out = mlp_out.squeeze().T.reshape(4, 5, 5)
cnn_out = cnn_out.squeeze()
f, axarr = plt.subplots(3, 4)
axarr[0, 0].imshow(char, cmap='gray')
axarr[0, 0].axis(False)
axarr[0, 0].set_title("Original")

axarr[0, 1].imshow(cnn_out[0].detach().cpu(), cmap='gray')
axarr[0, 1].axis(False)
axarr[0, 1].set_title("CNN")

axarr[0, 2].imshow(mlp_out[0].detach().cpu(), cmap='gray')
axarr[0, 2].axis(False)
axarr[0, 2].set_title("mlpmixer2cnn")

axarr[0, 3].imshow(mlp(img[0][0].flatten()).reshape(5, 5).detach().cpu(), cmap='gray')
axarr[0, 3].axis(False)
axarr[0, 3].set_title("Random MLP")

img = F.affine(img, angle=0, translate=(0, 2), scale=1, shear=0).cuda()
mlp_out, cnn_out = model.visualization(img, k)

mlp_out = mlp_out.squeeze().T.reshape(4, 5, 5)
cnn_out = cnn_out.squeeze()

axarr[1, 0].imshow(img[0][0].detach().cpu(), cmap='gray')
axarr[1, 0].axis(False)

axarr[1, 1].imshow(cnn_out[0].detach().cpu(), cmap='gray')
axarr[1, 1].axis(False)

axarr[1, 2].imshow(mlp_out[0].detach().cpu(), cmap='gray')
axarr[1, 2].axis(False)

axarr[1, 3].imshow(mlp(img[0][0].flatten()).reshape(5, 5).detach().cpu(), cmap='gray')
axarr[1, 3].axis(False)

img = F.affine(img, angle=0, translate=(2, 0), scale=1, shear=0).cuda()
mlp_out, cnn_out = model.visualization(img, k)

mlp_out = mlp_out.squeeze().T.reshape(4, 5, 5)
cnn_out = cnn_out.squeeze()

axarr[2, 0].imshow(img[0][0].detach().cpu(), cmap='gray')
axarr[2, 0].axis(False)

axarr[2, 1].imshow(cnn_out[0].detach().cpu(), cmap='gray')
axarr[2, 1].axis(False)

axarr[2, 2].imshow(mlp_out[0].detach().cpu(), cmap='gray')
axarr[2, 2].axis(False)

axarr[2, 3].imshow(mlp(img[0][0].flatten()).reshape(5, 5).detach().cpu(), cmap='gray')
axarr[2, 3].axis(False)

plt.subplots_adjust(hspace=0.15, wspace=0.1)
f.savefig("MixerCNNVisualization.png")

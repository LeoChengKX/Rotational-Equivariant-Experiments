"""
Version 5 experiments:
- mlpmixer 2 multi-channel conv layer
"""

import os
os.environ["WANDB_API_KEY"] = '63b4682f41e1cca160972dfea265edf551361fe0'

import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import lightning as L
import wandb

from datetime import datetime
from ml_collections import ConfigDict
from matplotlib import pyplot as plt
from nnutils import MLPMixer, mean_flat, timestep_embedding, cnn_dim_out, generate_fourier_features

def cnn_dim_out(in_size, ker, stride, padding):
    return math.floor((in_size - ker + 2 * padding)/stride)+1

cfg = ConfigDict()
cfg.name = 'mlp2cnn_v5'
cfg.root_dir = "/scratch/a/aspuru/huang651/eee_experiments"
cfg.log_dir = f"{cfg.root_dir}/logs/"
cfg.seed = 12315
cfg.timesteps = 200
cfg.max_iters = 12000
cfg.grad_norm_clip = 1.0
cfg.lr = 1e-3
cfg.bs = 200
cfg.img_dim = sz = 7
cfg.stride = sd = 1
cfg.kernel_size = ks = 3

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

mixer1.image_size=(sz**2+od**2+ks**2,1)
mixer1.channels=ch
mixer1.patch_size=1
mixer1.dim=hs
mixer1.depth=2

cfg.mixer2_cfg = mixer2 = ConfigDict()
mixer2.image_size=(sz**2+od**2+ks**2,1)
mixer2.channels=ch+ic
mixer2.patch_size=1
mixer2.dim=hs
mixer2.depth=2



class MLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.bs = cfg.bs
        self.mixer1 = MLPMixer(**cfg.mixer1_cfg) # encoder
        self.W_enc = nn.Linear(cfg.hidden_size, cfg.in_ch)
        self.mixer2 = MLPMixer(**cfg.mixer2_cfg) # decoder
        self.W_dec = nn.Linear(cfg.hidden_size, cfg.out_ch)
        self.register_buffer('ipos_emb', generate_fourier_features(cfg.bs, **cfg.input_fkwargs))
        self.register_buffer('qpos_emb', generate_fourier_features(cfg.bs, **cfg.query_fkwargs))
        self.in_ch, self.out_ch = self.cfg.in_ch, self.cfg.out_ch
        self.context_hidden_state = nn.Parameter(torch.randn(cfg.out_dim**2,cfg.channels))
        self.query_hidden_state = nn.Parameter(torch.randn(cfg.img_dim**2+cfg.kernel_size**2,cfg.channels))
        self.f_size = cfg.f_size
        self.k_embed = nn.Linear(self.in_ch*self.out_ch, cfg.channels)
        
    def input_pos_emb(self, bs):
        if bs == self.bs:
            return self.ipos_emb
        return generate_fourier_features(bs, **self.cfg.input_fkwargs).to(self.ipos_emb.device)
        
    def query_pos_emb(self, bs):
        if bs == self.bs:
            return self.qpos_emb
        return generate_fourier_features(bs, **self.cfg.query_fkwargs).to(self.qpos_emb.device)
        
    def forward(self, flat_img, flat_cout, flat_k, bs=None):
        if bs is None:
            bs = self.bs
        x = torch.cat((flat_img,self.input_pos_emb(bs)),dim=-1) # TODO
        x = torch.cat((x,self.k_embed(flat_k).expand(bs,-1,-1)),dim=1)
        x = torch.cat((x,self.context_hidden_state.expand(bs,-1,-1)),dim=1)
        query = torch.cat((
            self.query_hidden_state.expand(bs,-1,-1),
            torch.cat((torch.ones_like(flat_cout),self.query_pos_emb(bs)),dim=-1)
        ),dim=1)
        
        x = x[:,:,None,:].permute(0,3,1,2) # -> b, c, h, w
        x = self.W_enc(self.mixer1(x))
        x = torch.cat((x,query),dim=-1)
        x = x[:,:,None,:].permute(0,3,1,2) # -> b, c, h, w
        x = self.W_dec(self.mixer2(x))
        return x[:,-self.cfg.out_dim**2:]
    
    def train_step(self):
        ks = self.cfg.kernel_size
        img_dim = self.cfg.img_dim
        
        loss = 0 #nn.Conv2d(16, 32, 3, 1)
        in_ch, out_ch = self.in_ch, self.out_ch
        conv = nn.Conv2d(in_ch,out_ch,ks,1,bias=False).cuda()
        conv_bs = 50
        for _ in range(conv_bs):
            k = torch.randn(in_ch,out_ch,ks,ks).cuda()
            conv.weight = nn.Parameter(k, requires_grad=False) # cnn weight
            img = torch.randn(self.bs,in_ch,img_dim,img_dim).cuda()
            cnn_out = conv(img)
            flat_cout = cnn_out.flatten(start_dim=2).permute(0,2,1)
            model_output = self(
                img.flatten(start_dim=2).permute(0,2,1), 
                flat_cout,
                k.flatten(0,1).flatten(1,2).permute(1,0),
            )
            loss += mean_flat((flat_cout - model_output) ** 2).mean()
        return loss/conv_bs
    
    def visualization(self):
        ks = self.cfg.kernel_size
        img_dim = self.cfg.img_dim
        stride = self.cfg.stride
        
        # Conv Output
        in_ch, out_ch = self.in_ch, self.out_ch
        conv = nn.Conv2d(in_ch,out_ch,ks,1,bias=False).cuda()
        k = torch.randn(in_ch,out_ch,ks,ks).cuda()
        conv.weight = nn.Parameter(k,requires_grad=False) # cnn weight

        img = torch.randn(1,in_ch,img_dim,img_dim).cuda()
        # cnn_out = conv(img).squeeze().detach().cpu().numpy()
        cnn_out = conv(img)

        # MLP Output
        reshape_size = cnn_dim_out(img_dim, ks, stride, 0)
        mlp_out = self(
            img.flatten(start_dim=2).permute(0,2,1),
            cnn_out.flatten(start_dim=2).permute(0,2,1),
            k.flatten(0,1).flatten(1,2).permute(1,0),
            bs=1,
        ).detach().cpu().numpy().reshape(self.out_ch,reshape_size,reshape_size)
        cnn_out = cnn_out.squeeze().detach().cpu().numpy()
        # out_imgs = np.concatenate((cnn_out, mlp_out))
        # f, axarr = plt.subplots(2,self.out_ch,sharex=True, sharey=True)
        # for out_img, ax in zip(out_imgs, axarr.flatten()):
        #     ax.imshow(out_img)
        #     ax.axis(False)
        return mlp_out, cnn_out


if __name__ == "__main__":
    cfg.run_name = f'mlpmixer_'+datetime.now().strftime("run_%m%d_%H_%M")
    cfg.save_dir = f"{cfg.root_dir}/{cfg.name}_checkpoints/{cfg.run_name}/"
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    logger = WandbLogger(
        save_dir=cfg.log_dir, 
        project=cfg.name, 
        name=cfg.run_name, 
        log_model=True
    )
    
    torch.set_float32_matmul_precision('medium')
    model = MLP(cfg)
    fabric = L.Fabric(accelerator="cuda", loggers=[logger])
    fabric.seed_everything(cfg.seed)
    fabric.launch()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer = fabric.setup(model, optimizer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable params: {pytorch_total_params}")
    fig = model.visualization()
    fig.savefig(cfg.save_dir+"start_visualization.png")
    model.train()
    optimizer.zero_grad()
    for iteration in range(cfg.max_iters):
        is_accumulating = iteration % 25 != 0
        loss = model.train_step()
        fabric.log_dict({"train/loss": loss})
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()

    fig = model.visualization()
    fig.savefig(cfg.save_dir+"finish_visualization.png")
    wandb.log({"visualization": fig})
    state = { "model": model, "optimizer": optimizer }
    fabric.save(cfg.save_dir+f"checkpoint.ckpt", state)
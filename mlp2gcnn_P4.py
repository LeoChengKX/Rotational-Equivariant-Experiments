"""
Learn only P4ConvP4. 
"""
import os
os.environ["WANDB_API_KEY"] = '05f974a64215c03b7fc204d65f79b91c1bb3e369'

import torch
import math
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import lightning as L
import wandb

from datetime import datetime
from ml_collections import ConfigDict
from matplotlib import pyplot as plt

from gcov2d_for_learn import *

cfg = ConfigDict()
cfg.name = 'mlp2gcnn_P4'
cfg.root_dir = "/project/a/aspuru/chengl43/rot_equiv_exp"
cfg.log_dir = f"{cfg.root_dir}/logs/"
# cfg.seed = 42 
cfg.seed = 10000
cfg.max_iters = 10000
cfg.grad_norm_clip = 1.0
cfg.lr = 1e-2
cfg.img_dim = 10
cfg.stride = 1
cfg.kernel_size = 3
cfg.ss_gamma = 2.0
cfg.bs = 128


import math
import torch
from torch import nn

# Small constant used to ensure numerical stability.
EPSILON = 1e-6

def prepare_dataset():
    train_split = MNIST("dataset", train=True, download=True, transform=ToTensor())
    test_split = MNIST("dataset", train=False, transform=ToTensor())
    train_loader = DataLoader(train_split, batch_size=cfg.bs, shuffle=True)
    test_loader = DataLoader(test_split, batch_size=cfg.bs, shuffle=True)
    return train_loader, test_loader


class SmoothStep(nn.Module):
    """A smooth-step function in PyTorch.

    For a scalar x, the smooth-step function is defined as follows:
      0                                           if x <= -gamma/2
      1                                           if x >= gamma/2
      3*x/(2*gamma) - 2*x*x*x/(gamma**3) + 0.5    otherwise

    See https://arxiv.org/abs/2002.07772 for more details on this function.
    """

    def __init__(self, gamma=1.0):
        """Initializes the layer.

        Args:
          gamma: Scaling parameter controlling the width of the polynomial region.
        """
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2
        self._upper_bound = gamma / 2
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def forward(self, inputs):
        return torch.where(
            inputs <= self._lower_bound, torch.zeros_like(inputs),
            torch.where(inputs >= self._upper_bound, torch.ones_like(inputs),
                        self._a3 * (inputs**3) + self._a1 * inputs + self._a0))


def cnn_dim_out(in_size, ker, stride, padding):
    return math.floor((in_size - ker + 2 * padding)/stride)+1

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        in_dim, out_dim, hidden_dim, blocks, layers, block_out_dim, merge_dim, activation = (
            cfg.in_dim, cfg.out_dim, cfg.hidden_dim, cfg.blocks, cfg.layers, cfg.block_output_dim, cfg.merge_dim, cfg.activation
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.layers = layers
        self.block_out_dim = block_out_dim
        self.merge_dim = merge_dim
        self.activation = activation

        self.MLP_blocks = nn.ModuleList()
        self.f_out = nn.ModuleList()
        self.f_merge = nn.ParameterList()
        self.act_fn = SmoothStep(cfg.ss_gamma)
        for _ in range(blocks):
            self.MLP_blocks.append(MLP_block(in_dim, block_out_dim, hidden_dim, layers))
        self.mtx1 = nn.Parameter(torch.rand(cfg.kernel_size ** 2, self.merge_dim)) # Projection to embbeding space
        self.mtx2 = nn.Parameter(torch.rand(cfg.kernel_size ** 2 * self.merge_dim, self.blocks)) # Projection back to normal space

    def forward(self, img, cparams):
        img = img.flatten(start_dim=1)
        cparams = cparams.flatten()
        ks = cparams.shape[-1]
        if self.activation == 0:
            MLP_result = [self.MLP_blocks[i](img) for i in range(self.blocks)]
        else:
            MLP_result = [self.act_fn(self.MLP_blocks[i](img)) for i in range(self.blocks)]
        result = []

        h = cparams.view(cfg.kernel_size ** 2, 1) * self.mtx1 # Same shape as mtx1
        h = h.flatten()
        h = h @ self.mtx2 # self.blocks x 1
        result = [MLP_result[i] * h[i] for i in range(len(MLP_result))]

        return sum(result)
    
    def return_toeplitz(self, cparams):
        return sum([cparams[i]*next(iter(self.MLP_blocks[i].parameters())).T for i in range(cfg.kernel_size**2)]).detach().cpu().numpy()
    

class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, layers):
        super(MLP_block, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        hidden_layers = nn.ModuleList()
        for _ in range(layers - 2):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            hidden_layers.append(nn.ReLU())

        if self.layers >= 2:
            self.linears = nn.ModuleList(nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False), nn.ReLU()]) + 
                                        hidden_layers + nn.ModuleList([nn.Linear(hidden_dim, out_dim, bias=False)]))
        elif self.layers == 1:
            self.linears = nn.Linear(in_dim, out_dim, bias=False)
            with torch.no_grad():
                self.linears.weight.copy_(torch.zeros((out_dim, in_dim))) # Initialize as zero
        else:
            assert "Layers number not correct!"

    def forward(self, x):
        if self.layers == 1:
            return self.linears(x)
        else:
            for layer in self.linears:
                x = layer(x)
            return x

class fakeCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.models = nn.ModuleList([MLP(cfg) for _ in range(4)]) # 4 channels
        self.model = MLP(cfg)
        self.loss_rc = nn.MSELoss()
        self.loss_kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # self.cparams_transform_1 = nn.ParameterList([
        #     nn.Parameter(torch.rand(cfg.kernel_size ** 2, cfg.kernel_size ** 2)).cuda() for _ in range(4)])
        # self.cparams_transform_2 = nn.ParameterList([
        #     nn.Parameter(torch.rand(cfg.kernel_size ** 2, cfg.kernel_size ** 2)).cuda() for _ in range(4)])
        # self.cparams_mtx_up = nn.ParameterList([nn.Parameter(torch.rand(cfg.kernel_size ** 2, cfg.emb_dim)) for _ in range(4)])
        # self.cparams_mtxs_down = nn.ParameterList([nn.Parameter(torch.rand(cfg.kernel_size ** 2 * cfg.emb_dim, 
        #                                                                   cfg.kernel_size ** 2)) for _ in range(4)])
        self.cparams_transform = nn.ParameterList([nn.Parameter(torch.rand(4, cfg.kernel_size ** 2, cfg.kernel_size ** 2)) for _ in range(16)])

    def train_step(self, dataloader):
        ks = self.cfg.kernel_size
        img_dim = self.cfg.img_dim
        
        loss = 0
        conv = P4ConvP4()
        conv_bs = 128
        for _ in range(conv_bs):
            cparams = torch.randn(4 * ks**2).cuda()
            conv.weight = nn.Parameter(cparams.view(1,1,4,ks,ks), requires_grad=False) # cnn weight

            # Prepare dataset
            if cfg.dataset == 0:
                img_bs = cfg.bs
                img = torch.randn(img_bs,4,img_dim,img_dim).cuda()
            else:
                img, _ = next(iter(dataloader))
                img = torch.Tensor(img).cuda()
            
            cnn_out = conv(img)  # bs, 4, cnn_out_dim, cnn_out_dim

            # Architecture
            if cfg.provide_filter == 0:
                cp = []
                for i in range(16): 
                    cp.append(torch.sum(torch.einsum("bij,bj->bi", self.cparams_transform[i], cparams.view(4, ks ** 2)), axis=0))
                cp = torch.stack(cp).view(4, 4, ks ** 2)
                mlp_out = []
                for i in range(4):
                    layer_out = []
                    for j in range(4):
                        img_in = img.transpose(0, 1)[j].view(img_bs, -1)
                        layer_out.append(self.model(img_in, cp[i][j]))
                    mlp_out.append(sum(layer_out))
                mlp_out = torch.stack(mlp_out).transpose(0, 1)
            
            elif cfg.provide_filter == 1:
                # Test converging using provided filter
                mlp_out = []
                tw = trans_filter(cparams.view(1,1,4,ks,ks), self.make_transformation_indices())
                tw_shape = (4, 4, 3, 3)
                tw = tw.view(tw_shape)
                for i in range(4):
                    layer_out = []
                    for j in range(4):
                        cp = tw[i][j]
                        img_in = img.transpose(0, 1)[j].view(img_bs, -1)
                        layer_out.append(self.model(img_in, cp))
                    mlp_out.append(sum(layer_out))
                mlp_out = torch.stack(mlp_out).transpose(0, 1)

            # -------------------------------------------

            if cfg.loss_type == 0: # kl loss
                loss += self.loss_kl(
                    F.log_softmax(mlp_out.flatten(start_dim=1), dim=1),
                    F.log_softmax(cnn_out.flatten(start_dim=1), dim=1)
                )
            elif cfg.loss_type == 1: # rc loss
                loss += self.loss_rc(mlp_out, cnn_out.flatten(start_dim=2)).mean()
            else: # Both rc and kl loss combined
                loss += self.loss_kl(
                    F.log_softmax(mlp_out.flatten(start_dim=1), dim=1),
                    F.log_softmax(cnn_out.flatten(start_dim=1), dim=1)
                )
                loss += self.loss_rc(mlp_out.flatten(start_dim=1), cnn_out.flatten(start_dim=1)).mean()
        return loss/conv_bs

    def transform_cparms(cparams):
        pass

    def make_transformation_indices(self):
        return make_indices_functions[(4, 4)](cfg.kernel_size)
    
    def visualization(self, dataloader):
        ks = self.cfg.kernel_size
        img_dim = self.cfg.img_dim
        stride = self.cfg.stride
        
        # Conv Output
        conv = P4ConvZ2()
        cparams = torch.randn(ks**2).cuda()
        conv.weight = nn.Parameter(cparams.reshape(1,1,1,ks,ks),requires_grad=False) # cnn weight

        if cfg.dataset == 0:
            img = torch.randn(1,img_dim,img_dim).cuda()
        else:
            img = next(iter(dataloader))
            img = torch.Tensor(img[0][0]).cuda()
            
        cnn_out = conv(img).squeeze().detach().cpu().numpy()[0]

        # MLP Output
        reshape_size = cnn_dim_out(img_dim, ks, stride, 0)
        cp = self.cparams_transform[0] @ cparams
        mlp_out = self.model(img.view(1,-1), cp).detach().cpu().numpy().reshape(reshape_size,reshape_size)

        vmin = min(cnn_out.min(), mlp_out.min())
        vmax = max(cnn_out.max(), mlp_out.max())
        f, axarr = plt.subplots(1,2, figsize=(16, 8))
        axarr[0].imshow(cnn_out, vmin=vmin, vmax=vmax)
        axarr[0].axis(False)
        axarr[1].imshow(mlp_out, vmin=vmin, vmax=vmax)
        axarr[1].axis(False)

        plt.subplots_adjust(left=0.05, right=0.8, wspace=0.2)
        return f
    
    def forward(self, img, cparams):
        """Used for visualization solely. """
        ks = 3
        if self.cfg.provide_filter == 0:
            cp = []
            for i in range(16): 
                cp.append(torch.sum(torch.einsum("bij,bj->bi", self.cparams_transform[i], cparams.view(4, ks ** 2)), axis=0))
            cp = torch.stack(cp).view(4, 4, ks ** 2)
            mlp_out = []
            for i in range(4):
                layer_out = []
                for j in range(4):
                    img_in = img.transpose(0, 1)[j].view(1, -1)
                    layer_out.append(self.model(img_in, cp[i][j]))
                mlp_out.append(sum(layer_out))
            mlp_out = torch.stack(mlp_out).transpose(0, 1)
        
        elif self.cfg.provide_filter == 1:
            # Test converging using provided filter
            mlp_out = []
            tw = trans_filter(cparams.view(1,1,4,ks,ks), self.make_transformation_indices())
            tw_shape = (4, 4, 3, 3)
            tw = tw.view(tw_shape)
            for i in range(4):
                layer_out = []
                for j in range(4):
                    cp = tw[i][j]
                    img_in = img.transpose(0, 1)[j].view(img.shape[0], -1)
                    layer_out.append(self.model(img_in, cp))
                mlp_out.append(sum(layer_out))
            mlp_out = torch.stack(mlp_out).transpose(0, 1)

        return mlp_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=int, default=0)
    parser.add_argument("--loss_type", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1000) # 5000
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--ss_gamma", type=float)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dataset", type=int, default=0) # 0 or 1
    parser.add_argument("--hidden_dim", type=int, default=64) # Hidden dimension
    parser.add_argument("--block_output_dim", type=int, default=64) # MLP Block out dimension
    parser.add_argument("--img_dim", type=int, default=10)
    parser.add_argument("--merge_dim", type=int, default=10) # determine the dimension of the embbeding space for merging blocks
    parser.add_argument("--emb_dim", type=int, default=20) # The dimension of the embbeding space for cparams
    parser.add_argument("--provide_filter", type=int, choices=[0, 1], default=0) # Whether provide transformed filters
    args = parser.parse_args()
    cfg.update(**{k:v for k,v in vars(args).items() if v is not None})
    
    cfg.run_name = f'blocks{cfg.blocks}_layers{cfg.layers}_img{cfg.img_dim}_filter{cfg.provide_filter}_emb_dim{cfg.emb_dim}_img_dim{cfg.img_dim}_' + \
            datetime.now().strftime("run_%m%d_%H_%M")
    cfg.save_dir = f"{cfg.root_dir}/checkpoints/{cfg.name}_checkpoints/{cfg.run_name}/"
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    cfg.block_output_dim = cnn_dim_out(cfg.img_dim, 3, 1, 0) ** 2 # Force output dim same as cnn output 

    logger = WandbLogger(
        save_dir=cfg.log_dir, 
        project=cfg.name, 
        name=cfg.run_name, 
        log_model=True,
        dir="$PROJECT/"
    )
    torch.set_float32_matmul_precision('medium')
    cfg.in_dim = cfg.img_dim**2
    cfg.out_hw = cnn_dim_out(cfg.img_dim, cfg.kernel_size, cfg.stride, 0)
    cfg.out_dim = cfg.out_hw**2
    model = fakeCNN(cfg)
    
    train_data, test_data = prepare_dataset()
    fabric = L.Fabric(accelerator="cuda", loggers=[logger])
    fabric.seed_everything(cfg.seed)
    fabric.launch()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer = fabric.setup(model, optimizer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total trainable params: {pytorch_total_params}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    model.train()
    optimizer.zero_grad()
    for iteration in range(cfg.max_iters):
        is_accumulating = iteration % 5 != 0
        loss = model.train_step(train_data)
        if iteration % 100 == 0:
            fabric.log_dict({"train/loss": loss})
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
        if iteration % 500 == 0: # for visualization
        #     fig = model.visualization(test_data)
        #     fig.savefig(cfg.save_dir+f"visualization_{iteration}.png")
            state = { "model": model, "optimizer": optimizer }
            fabric.save(cfg.save_dir+f"checkpoint_{iteration}.ckpt", state)

    fig = model.visualization(test_data)
    fig.savefig(cfg.save_dir+"visualization.png")
    wandb.log({"visualization": fig})
    state = { "model": model, "optimizer": optimizer }
    fabric.save(cfg.save_dir+f"checkpoint.ckpt", state)

import os
os.environ["WANDB_API_KEY"] = '05f974a64215c03b7fc204d65f79b91c1bb3e369'

import torch
import math
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomRotation, Compose
from torch.utils.data import DataLoader
import lightning as L
import wandb
from torchsummary import summary

from datetime import datetime
from ml_collections import ConfigDict
from matplotlib import pyplot as plt

from groupy.gconv.pytorch_gconv import *
from pooling import MaxPoolRotation2D, MaxPoolSpatial2D

def prepare_dataset():
    transform = Compose([
        # RandomRotation(degrees=90),
        ToTensor()
    ])

    mnist_train = MNIST(root='data', train=True, download=True, transform=transform)
    mnist_val = MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=20, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=20, shuffle=False)
    return train_loader, val_loader

class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3), # 26 x 26
            nn.ReLU(),
            nn.MaxPool2d(2), # 13 x 13
            nn.Conv2d(16, 32, 3), # 11 x 11
            nn.ReLU(),
            nn.MaxPool2d(2), # 5 x 5
            nn.Conv2d(32, 10, 3), # 3 x 3
            nn.ReLU(),
            )
        
    
    def forward(self, x, return_logits=True):
        x = self.model(x)
        x = x.reshape(x.shape[0], 10, -1).max(dim=-1)[0]
        if return_logits:
            x = F.log_softmax(x, dim=1)
        return x, 

class P4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            P4ConvZ2(1, 10, 3), 
            nn.ReLU(),
            MaxPoolSpatial2D(2, stride=1),
            P4ConvP4(10, 10, 3),
            nn.ReLU(),
            MaxPoolSpatial2D(2, stride=1),
            P4ConvP4(10, 10, 3),
            nn.ReLU(),
            MaxPoolSpatial2D(2, stride=1),
            P4ConvP4(10, 10, 3),
            nn.ReLU(),
            MaxPoolSpatial2D(2, stride=1),
            P4ConvP4(10, 10, 3),
            nn.ReLU(),
            MaxPoolSpatial2D(2, stride=1),
            P4ConvP4(10, 10, 3),
            nn.ReLU(),
            MaxPoolSpatial2D(2, stride=1),
            P4ConvP4(10, 10, 4),
            MaxPoolRotation2D()
            )
        
    
    def forward(self, x, return_logits=True):
        x = self.model(x)
        x = x.reshape(x.shape[0], 10, -1).max(dim=-1)[0]
        if return_logits:
            x = F.log_softmax(x, dim=1)
        return x, 


class MNISTClassifier(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits, *_loss = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        self.log("train/loss", loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits, *_loss = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        
        _, predicted = torch.max(logits, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        self.log("val/accuracy", correct/total, on_step = True)
        print(f'val/accuracy: {correct*100/total:.2f}%')
        self.log("val/loss", loss, on_step=True, sync_dist=True)
        return {"val_loss": loss, "correct": correct, "total": total}
    
    # def on_validation_end(self):
    #     accuracy =self.validation_step_outputs[0]/self.validation_step_outputs[1]
    #     self.log("val/accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    cfg = ConfigDict()
    cfg.lr = 1e-2
    cfg.name = 'RotMNIST_P4CNN_benchmark'
    cfg.run_name = datetime.now().strftime("run_%m%d_%H_%M")
    cfg.seed = 10


    save_dir = f'RotMNIST_benchmarking/'
    model_path = 'models'

    Wandb_logger = WandbLogger(
        save_dir=save_dir,
        project=cfg.name, 
        name=cfg.run_name,
        log_model=True,
    )

    CSV_logger = CSVLogger('MNIST_benchmarking', name=save_dir, version = datetime.now().strftime("run_%m%d_%H_%M"))
    early_stopping = EarlyStopping('val/loss', mode='min', patience=5)
    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss',
        every_n_epochs=1,
    )

    # fabric = L.Fabric(accelerator='cuda', loggers=[Wandb_logger, CSV_logger])
    # fabric.seed_everything(cfg.seed)
    # fabric.launch()

    model = CNNBaseline()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # model, optimizer = fabric.setup(model, optimizer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_params}")
    # summary(model, (1, 28, 28))
    model.train()
    # optimizer.zero_grad()

    wrapper = MNISTClassifier(model)
    trainer = Trainer(
        max_epochs=50,
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=model_path, 
        log_every_n_steps=50,
        enable_progress_bar=False,
        logger=[CSV_logger]
    )
    
    train_loader, val_loader = prepare_dataset()
    trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)

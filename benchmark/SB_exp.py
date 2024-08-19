import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from torch.nn import functional as F
from pytorch_lightning import LightningModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from ml_collections import ConfigDict

from datetime import datetime
import math
import random
from copy import deepcopy
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from MLP_gconv import MLPGConv_P4, MLPGConv_Z2, Fast_MLPGConv_P4

torch.manual_seed(10)
random.seed(10)

project_path = '/project/a/aspuru/chengl43/rot_equiv_exp/benchmark/'
ckpt_28 = project_path + 'checkpoints/9_28_Z2_checkpoint.ckpt'
ckpt_26 = project_path + 'checkpoints/9_26_P4_checkpoint.ckpt'
ckpt_12 = project_path + 'checkpoints/9_12_P4_checkpoint.ckpt'
ckpt_10 = project_path + 'checkpoints/9_10_P4_checkpoint.ckpt'

def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0):
    xs = x.size()
    x = x.reshape(xs[0], xs[1] * xs[2], xs[3], xs[4])
    x = F.max_pool2d(input=x, kernel_size=ksize, stride=stride, padding=pad)
    x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 20)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.log_softmax(self.fc7(x), dim=1)
        return x,


class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(4*20*4, 50)
        self.fc2 = nn.Linear(50, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 20, kernel_size=3)
        self.conv3 = P4ConvP4(20, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        self.projection = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 26, 26
        x = F.relu(self.conv2(x)) # 24, 24
        x = plane_group_spatial_max_pooling(x, 2, 2) # 12, 12
        x = F.relu(self.conv3(x)) # 10, 10
        x = F.relu(self.conv4(x)) # 8, 8
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = self.projection(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
class MLPGConvNet(nn.Module):
    def __init__(self):
        super(MLPGConvNet, self).__init__()
        self.conv1 = MLPGConv_Z2(ckpt_28, 28, 1, 10, ks=3)
        self.conv2 = Fast_MLPGConv_P4(ckpt_26, 26, 10, 10, ks=3)
        self.conv3 = Fast_MLPGConv_P4(ckpt_12, 12, 10, 20, ks=3)
        self.conv4 = Fast_MLPGConv_P4(ckpt_10, 10, 20, 20, ks=3)
        self.projection = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = self.projection(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class MLPGConvNet_Unfreeze(nn.Module):
    def __init__(self):
        super(MLPGConvNet_Unfreeze, self).__init__()
        self.conv1 = MLPGConv_Z2(ckpt_28, 28, 1, 10, ks=3, freeze=False)
        self.conv2 = Fast_MLPGConv_P4(ckpt_26, 26, 10, 10, ks=3, freeze=False)
        self.conv3 = Fast_MLPGConv_P4(ckpt_12, 12, 10, 20, ks=3, freeze=False)
        self.conv4 = Fast_MLPGConv_P4(ckpt_10, 10, 20, 20, ks=3, freeze=False)
        self.projection = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 20)

        self.weights_initial = {}
        self.KLLoss = nn.KLDivLoss()
        for name, p in self.named_parameters():
            p_c = deepcopy(p).cuda()
            p_c = p_c.detach()
            self.weights_initial[name] = p_c

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = self.projection(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def get_KLLoss(self):
        params_new = self.named_parameters()
        new_data = dict(params_new)
        loss = 0
        for k in self.weights_initial.keys():
            if any(e in k for e in ['mtx1', 'mtx2', 'weights', 'cparams_transform']): # Original weights
                loss += self.KLLoss(F.log_softmax(self.weights_initial[k]), F.softmax(new_data[k]))
        return loss
    

# class RotatedMNIST(Dataset):
#     def __init__(self, root, train=True, transform=transforms.ToTensor()):
#         self.mnist = datasets.MNIST(root=root, train=train, download=False, transform=None)
#         self.transform = transform

#     def __len__(self):
#         return len(self.mnist)

#     def __getitem__(self, idx):
#         img, label = self.mnist[idx]
        
#         # Randomly translate the image
#         direction = random.choice([0, 180])
#         if direction == 0:
#             rotation = random.randint(30, 120)
#             # rotation = random.randint(30, 180)
#             translated_img = transforms.functional.affine(img, angle=rotation, translate=(0, 0), scale=1.0, shear=0)
#             new_label = label
#         else:
#             rotation = -random.randint(30, 120)
#             # rotation = -random.randint(30, 180)
#             translated_img = transforms.functional.affine(img, angle=rotation, translate=(0, 0), scale=1.0, shear=0)
#             new_label = label + 10  # Add 10 to the label to indicate translation to the right
        
#         if self.transform is not None:
#             translated_img = self.transform(translated_img)

#         return translated_img, new_label

class RotatedMNIST(Dataset):
    def __init__(self, root, train=True, transform=transforms.ToTensor()):
        self.mnist = datasets.MNIST(root=root, train=train, download=False, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # Randomly rotate the image
        direction = random.choice([0, 90, 180, 270])
        if direction == 0 or direction == 90:
            translated_img = transforms.functional.affine(img, angle=direction, translate=(0, 0), scale=1.0, shear=0)
            new_label = label
        else:
            translated_img = transforms.functional.affine(img, angle=direction, translate=(0, 0), scale=1.0, shear=0)
            new_label = label + 10  # Add 10 to the label to indicate translation to the right
        
        if self.transform is not None:
            translated_img = self.transform(translated_img)

        return translated_img, new_label

def setup_experiment(exp_type, train_aug=False, val_aug=False):
    trainset = RotatedMNIST("data", train=True)
    valset = RotatedMNIST("data", train=False)

    # trainset = datasets.MNIST("data", train=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    # ]))

    # valset = datasets.MNIST("data", train=False, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.RandomAffine(degrees=180, translate=(0, 0))
    #     transforms.RandomRotation(degrees=180)
    # ]))

    if exp_type == 'gconv':
        model = Net()
    elif exp_type == 'faux-gconv':
        model = MLPGConvNet()
    elif exp_type == 'faux-unfreeze':
        model = MLPGConvNet_Unfreeze()
    elif exp_type == 'cnn':
        model = CNNBaseline()
    elif exp_type == 'mlp':
        model = MLPBaseline()
    
    tloader = DataLoader(trainset, batch_size=32, num_workers=20, shuffle=True)
    vloader = DataLoader(valset, batch_size=32, num_workers=20, shuffle=False)
    return model, tloader, vloader


class MNISTClassifier(LightningModule):  
    def __init__(self, model, exp_type, beta=5e10):
        super().__init__()
        self.exp_type = exp_type
        self.model = model
        self.beta = beta
        self.weights_initial = []
        self.KLLoss = nn.KLDivLoss()
        for p in self.model.parameters():
            p_c = deepcopy(p).cuda()
            p_c = p_c.detach()
            self.weights_initial.append(p_c)

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        if self.exp_type == 'mlp':
            logits = logits[0]
        loss = self.cross_entropy_loss(logits, y)

        if self.exp_type == 'faux-unfreeze':
            loss += self.get_KLLoss() * self.beta
        self.log("train/loss", loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        if self.exp_type == 'mlp':
            logits = logits[0]
        loss = self.cross_entropy_loss(logits, y)
        
        _, predicted = torch.max(logits, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        self.log("val/accuracy", correct/total, on_step = True)
        # print(f'val/accuracy: {correct*100/total:.2f}%')
        self.log("val/loss", loss, on_step=True, sync_dist=True)
        return {"val_loss": loss, "correct": correct, "total": total}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.001)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

    def get_KLLoss(self):
        params_new = self.model.parameters()
        new_data = [p for p in params_new]
        loss = 0
        for i in range(len(self.weights_initial)):
            if self.weights_initial[i].shape[0] == 9:
                loss += self.KLLoss(F.log_softmax(self.weights_initial[i]), F.softmax(new_data[i]))
        return loss
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type",type=str , choices=['gconv', 'faux-gconv', 'faux-unfreeze', 'cnn', 'mlp'], default='faux-unfreeze')
    parser.add_argument("--blocks", type=int, default=9)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    # EXP_NAME = 'mnist-baseline'
    cfg = ConfigDict()
    # cfg.exp_type = 'faux-cnn' # mlp, mlp-promax, mlp-pro, cnn, faux-cnn
    cfg.train_aug = False
    cfg.val_aug = True
    cfg.update(**{k:v for k,v in vars(args).items() if v is not None})
    save_dir = f"{cfg.exp_type}_symmetry_breaking"

    model, tloader, vloader = setup_experiment(cfg.exp_type, cfg.train_aug, cfg.val_aug)
    print('running experiment:', cfg['exp_type'])
    
    model_path = 'models/'

    wrapper = MNISTClassifier(model, cfg['exp_type'])

    logger = CSVLogger('MNIST_benchmarking', name=save_dir, version = datetime.now().strftime("run_%m%d_%H_%M"))
    early_stopping = EarlyStopping('val/loss', mode='min', patience=5)

    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss',
        every_n_epochs=1,
    )
        
    trainer = Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        default_root_dir=model_path,
        log_every_n_steps=50,
        enable_progress_bar=True,
        logger = logger
    )
    trainer.fit(wrapper, tloader, vloader)


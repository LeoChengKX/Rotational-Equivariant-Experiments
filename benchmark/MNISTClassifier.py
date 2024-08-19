from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
import torchsummary
from datetime import datetime
from MLP_gconv import MLPGConv_P4, MLPGConv_Z2, Fast_MLPGConv_P4

from copy import deepcopy
import random

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--blocks", type=int, default=9, choices=[7,8,9])
parser.add_argument("--exp_type", type=str, default='MLP_GCNN', choices=['CNN', 'GCNN', "MLP_GCNN", "MLP_GCNN_Unfreeze", "MLP"])
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

project_path = '/project/a/aspuru/chengl43/rot_equiv_exp/benchmark/'
ckpt_28 = project_path + f'checkpoints/{args.blocks}_28_Z2_checkpoint.ckpt'
ckpt_26 = project_path + f'checkpoints/{args.blocks}_26_P4_checkpoint.ckpt'
ckpt_12 = project_path + f'checkpoints/{args.blocks}_12_P4_checkpoint.ckpt'
ckpt_10 = project_path + f'checkpoints/{args.blocks}_10_P4_checkpoint.ckpt'



class RotatedMNIST(Dataset):
    def __init__(self, root, train=True, transform=transforms.ToTensor()):
        self.mnist = datasets.MNIST(root=root, train=train, download=False, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # Randomly translate the image
        direction = random.choice([0, 90, 180, 270])
        translated_img = transforms.functional.affine(img, angle=direction, translate=(0, 0), scale=1.0, shear=0)
        
        if self.transform is not None:
            translated_img = self.transform(translated_img)

        return translated_img, label

kwargs = {'num_workers': 31, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(RotatedMNIST("data", train=False), batch_size=args.test_batch_size, shuffle=True, **kwargs)


# kwargs = {'num_workers': 20, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.RandomRotation(180),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=True, **kwargs)


def plane_group_spatial_max_pooling(x, ksize, stride=None, pad=0):
    xs = x.size()
    x = x.reshape(xs[0], xs[1] * xs[2], xs[3], xs[4])
    x = F.max_pool2d(input=x, kernel_size=ksize, stride=stride, padding=pad)
    x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x


class MNISTBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 10)

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 20, kernel_size=3)
        self.conv3 = P4ConvP4(20, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        self.projection = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)

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
    
class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(4*20*4, 50)
        self.fc2 = nn.Linear(50, 10)

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

class MLPGConvNet(nn.Module):
    def __init__(self):
        super(MLPGConvNet, self).__init__()
        self.conv1 = MLPGConv_Z2(ckpt_28, 28, 1, 10, ks=3)
        self.conv2 = Fast_MLPGConv_P4(ckpt_26, 26, 10, 10, ks=3)
        self.conv3 = Fast_MLPGConv_P4(ckpt_12, 12, 10, 20, ks=3)
        self.conv4 = Fast_MLPGConv_P4(ckpt_10, 10, 20, 20, ks=3)
        self.projection = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)

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

class MLPGConvNet_Freeze(nn.Module):
    def __init__(self):
        super(MLPGConvNet_Freeze, self).__init__()
        self.conv1 = MLPGConv_Z2(ckpt_28, 28, 1, 10, ks=3,freeze=False)
        self.conv2 = Fast_MLPGConv_P4(ckpt_26, 26, 10, 10, ks=3, freeze=False)
        self.conv3 = Fast_MLPGConv_P4(ckpt_12, 12, 10, 20, ks=3, freeze=False)
        self.conv4 = Fast_MLPGConv_P4(ckpt_10, 10, 20, 20, ks=3, freeze=False)
        self.projection = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)
        
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

class MNISTClassifier(LightningModule):
    def __init__(self, model, exp_type, beta=1e15):
        super().__init__()
        self.model = model
        self.exp_type = exp_type
        self.beta = beta
        self.KLLoss = nn.KLDivLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        if self.exp_type == 'MLP':
            logits = logits[0]
        loss = self.cross_entropy_loss(logits, y)
        
        if isinstance(model, MLPGConvNet_Freeze):
            loss += self.model.get_KLLoss() * self.beta
        self.log("train/loss", loss, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        if self.exp_type == 'MLP':
            logits = logits[0]
        loss = self.cross_entropy_loss(logits, y)
        
        _, predicted = torch.max(logits, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        self.log("val/accuracy", correct/total, on_step = True)
        # print(f'val/accuracy: {correct*100/total:.2f}%')
        self.log("val/loss", loss, on_step=True, sync_dist=True)
        # print(self.get_KLLoss())
        return {"val_loss": loss, "correct": correct, "total": total}
    
    # def on_validation_end(self):
    #     accuracy =self.validation_step_outputs[0]/self.validation_step_outputs[1]
    #     self.log("val/accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=[0.92, 0.999])
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]


# def train(model, epoch, beta=1e10):  # 2.03e5
    
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         if isinstance(model, MLPGConvNet_Freeze):
#             loss += model.get_KLLoss() * beta
#         # loss /= 1e10
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data))


# def test(model):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).data # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=['GCNN', 'CNN', 'MLP_GCNN', 'MLP_GCNN_Unfreeze'], default='MLP_GCNN_Unfreeze')
    # args = parser.parse_args()

    model_type = args.exp_type
    # model_type = 'MLP'

    if model_type == 'GCNN':
        model = Net()
    elif model_type == 'CNN':
        model = CNNBaseline()
    elif model_type == 'MLP_GCNN':
        model = MLPGConvNet()
    elif model_type == 'MLP_GCNN_Unfreeze':
        model = MLPGConvNet_Freeze()
    elif model_type == 'MLP':
        model = MNISTBaseline()
    else:
        raise NotImplementedError
    
    if args.cuda:
        model.cuda()

    # optimizer = optim.AdamW(model.parameters(), lr=0.002)
    # # torchsummary.summary(model, input_size=(1, 1, 28, 28))
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"total trainable params: {pytorch_total_params}")

    # for epoch in range(1, args.epochs + 1):
    #     train(model, epoch)
    #     test(model)
    
    torch.set_float32_matmul_precision('medium')
    model_path = 'models/'
    save_dir = f'{model_type}'

    wrapper = MNISTClassifier(model, model_type)

    logger = CSVLogger('MNIST_benchmarking', name=save_dir, version = f"run_blocks{args.blocks}_" + datetime.now().strftime("%m%d_%H_%M"))
    early_stopping = EarlyStopping('val/loss', mode='min', patience=10)

    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss',
        every_n_epochs=1,
    )
        
    trainer = Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        default_root_dir=model_path,
        log_every_n_steps=100,
        enable_progress_bar=True,
        logger = logger,
        # val_check_interval=10
    )
    trainer.fit(wrapper, train_loader, test_loader)

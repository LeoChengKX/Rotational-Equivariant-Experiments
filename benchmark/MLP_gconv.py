import torch
import math
import torch.nn as nn

def cnn_dim_out(in_size, ker, stride, padding):
    return math.floor((in_size - ker + 2 * padding)/stride)+1

def load_mlp_weights(ckpt_path):
    x = torch.load(ckpt_path)
    if 'model' in x.keys():
        x_model = x['model']
    # return torch.stack([*x.values()]).cpu().detach()
    weights = []
    cp_trans = []
    for key in x_model.keys():
        if key[:7] == 'cparams':
            cp_trans.append(x_model[key])
        elif key != 'model.mtx1' and key != 'model.mtx2':
            weights.append(x_model[key])
    return x_model['model.mtx1'], x_model['model.mtx2'], torch.stack(weights), torch.stack(cp_trans)

class MLPGConv_Z2(nn.Module):
    def __init__(self, ckpt_path, img_dim, ch_in, ch_out, ks, cnn_weight=None, cnn_bias=None):
        super().__init__()
        self.ks = ks
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.img_dim = img_dim
        self.num_pixels = img_dim**2
        self.cnn_out_dim = cnn_dim_out(img_dim, ks, 1, 0)
        
        if cnn_weight is None:
            cnn_weight = torch.empty(ch_out, ch_in, ks, ks)
            nn.init.kaiming_uniform_(cnn_weight, a=math.sqrt(5))
        if cnn_bias is None:
            cnn_bias = torch.empty(ch_out)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(cnn_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(cnn_bias, -bound, bound)

        self.conv_w = nn.ParameterList([
            nn.Parameter(cnn_weight[i], requires_grad=True)
            for i in range(ch_out)
        ])
        self.conv_b = nn.Parameter(cnn_bias, requires_grad=True)

        self.mtx1, self.mtx2, self.weights, self.cparams_transform = load_mlp_weights(ckpt_path)
        self.blocks = self.weights.shape[0]

    
    def forward(self, img):
        # Input (bs, ch_in, img_size, img_size)
        img = img.flatten(start_dim=2) # (bs, ch_in, img_size ** 2)

        # For non-matrix representation: 
        result = []
        for i in range(self.blocks):
            result.append(img @ self.weights[i].T) # weights.T: img_size ** 2, bout_dim

        result = torch.stack(result).to("cuda") # (blocks, bs, ch_in, bout_dim)
       
        out = []
        per_result = result.permute(2, 1, 3, 0) # (ch_in, bs, bout_dim, blocks)
        for i in range(self.ch_out):
            out_temp = []
            for j in range(self.ch_in):
                mlp_out = []
                conv_ij = self.conv_w[i][j] # ch_in, ks, ks
                for k in range(4):
                    cp = self.cparams_transform[k] @ conv_ij.flatten()
                    mtx1_temp = (self.mtx1 * cp.view(self.ks ** 2, 1)).flatten() # ks ** 2 * merge_dim
                    mtx2_temp = mtx1_temp @ self.mtx2 # (blocks,)
                    result_temp = torch.sum(mtx2_temp * per_result[j], axis=2) + self.conv_b[i]
                    mlp_out.append(result_temp)
                mlp_out = torch.stack(mlp_out).transpose(0, 1) # bs, 4, out
                out_temp.append(mlp_out)
            out.append(sum(out_temp))

        out = torch.stack(out)

        return out.transpose(0, 1).reshape(-1, self.ch_out, 4, self.cnn_out_dim, self.cnn_out_dim)


class MLPGConv_P4(nn.Module):
    def __init__(self, ckpt_path, img_dim, ch_in, ch_out, ks, cnn_weight=None, cnn_bias=None):
        super().__init__()
        self.ks = ks
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.img_dim = img_dim
        self.num_pixels = img_dim**2
        self.cnn_out_dim = cnn_dim_out(img_dim, ks, 1, 0)
        
        if cnn_weight is None:
            cnn_weight = torch.empty(ch_out, ch_in, 4, ks, ks)
            nn.init.kaiming_uniform_(cnn_weight, a=math.sqrt(5))
        if cnn_bias is None:
            cnn_bias = torch.empty(ch_out)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(cnn_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(cnn_bias, -bound, bound)

        self.conv_w = nn.ParameterList([
            nn.Parameter(cnn_weight[i], requires_grad=True)
            for i in range(ch_out)
        ])
        self.conv_b = nn.Parameter(cnn_bias, requires_grad=True)

        self.mtx1, self.mtx2, self.weights, self.cparams_transform = load_mlp_weights(ckpt_path)
        self.blocks = self.weights.shape[0]

    
    def forward(self, img):
        # Input (bs, ch_in, in_stab, img_size, img_size)
        img = img.flatten(start_dim=3) # (bs, ch_in, in_stab, img_size ** 2)

        # For non-matrix representation: 
        result = []
        for i in range(self.blocks):
            result.append(img @ self.weights[i].T) # weights.T: img_size ** 2, bout_dim

        result = torch.stack(result).to("cuda") # (blocks, bs, ch_in, in_stab, bout_dim)

        out = []
        per_result = result.permute(2, 3, 1, 4, 0) # (ch_in, stab_in, bs, bout_dim, blocks)
        for i in range(self.ch_out):
            out_temp = []
            for j in range(self.ch_in):
                mlp_out = []
                conv_ij = self.conv_w[i][j] # ch_in, ks, ks
                cp = []
                for ii in range(16): 
                    cp.append(torch.sum(torch.einsum("bij,bj->bi", self.cparams_transform[ii], conv_ij.view(4, self.ks ** 2)), axis=0))
                cp = torch.stack(cp).view(4, 4, self.ks ** 2)
                k_out = []
                for k in range(4):
                    m_out = []
                    for m in range(4):
                        mtx1_temp = (self.mtx1 * cp[k][m].view(self.ks ** 2, 1)).flatten() # ks ** 2 * merge_dim
                        mtx2_temp = mtx1_temp @ self.mtx2 # (blocks,)
                        result_temp = torch.sum(mtx2_temp * per_result[j][m], axis=2) + self.conv_b[i]
                        m_out.append(result_temp)
                    k_out.append(sum(m_out))
                mlp_out = torch.stack(k_out).transpose(0, 1) # bs, 4, out
                out_temp.append(mlp_out)
            out.append(sum(out_temp))

        out = torch.stack(out)

        return out.transpose(0, 1).reshape(-1, self.ch_out, 4, self.cnn_out_dim, self.cnn_out_dim)

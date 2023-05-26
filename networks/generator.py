from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis
import torch
import math
from torch.nn import functional as F


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class Generator(nn.Module):
    def __init__(self, size, style_dim=256, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec_pose = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)
        self.dec_exp = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

        # motion network pose
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        fc.append(EqualLinear(style_dim, motion_dim))
        self.mlp = nn.Sequential(*fc)

        self.dir = Direction(motion_dim)

        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(2):
            fc.append(EqualLinear(style_dim, style_dim))
        self.mlp_pose = nn.Sequential(*fc)

        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(2):
            fc.append(EqualLinear(style_dim, style_dim))
        self.mlp_exp = nn.Sequential(*fc)

    def forward(self, img_source, img_drive,stage='train'):
        if stage == 'train':
            final_output = {}
            
            wa, wa_t, feats, feats_t = self.enc(img_source, img_drive, h_start)
            alpha_D = self.mlp(wa_t)
            directions_D = self.dir(alpha_D)
            alpha_S = self.mlp(wa)
            directions_S = self.dir(alpha_S)
            directions_poseD = self.mlp_pose(directions_D)
            directions_poseS = self.mlp_pose(directions_S)
            directions_expD = self.mlp_exp(directions_D)
            directions_expS = self.mlp_exp(directions_S)

            # forward
            latent_poseD = wa + directions_poseD  # wa + direction
            img_recon = self.dec_pose(latent_poseD, None, feats)
            final_output['fake_poseB2A'] = img_recon
            wa1, _, feats1, _ = self.enc(final_output['fake_poseB2A'], None, h_start)
            latent_poseexpD = wa1 + directions_expD  # wa + direction
            img_recon = self.dec_exp(latent_poseexpD, None, feats1)
            final_output['fake_pose_expB2A'] = img_recon


            # backward
            latent_expS = wa_t + directions_expS  # wa + direction
            img_recon = self.dec_exp(latent_expS, None, feats_t)
            final_output['fake_expA2B'] = img_recon
            wa2, _, feats2,_ = self.enc(final_output['fake_expA2B'], None, h_start)
            latent_expposeS = wa2 + directions_poseS  # wa + direction
            img_recon = self.dec_pose(latent_expposeS, None, feats2)
            final_output['fake_exp_poseA2B'] = img_recon

            # self rec
            # pose
            latent_selfpose = wa + directions_poseS  # wa + direction
            img_recon = self.dec_pose(latent_selfpose, None, feats)
            final_output['fake_selfpose'] = img_recon

            # exp
            latent_selfexp = wa + directions_expS  # wa + direction
            img_recon = self.dec_exp(latent_selfexp, None, feats)
            final_output['fake_selfexp'] = img_recon

            return final_output
            
        elif stage=='pose':
            wa, wa_t, feats, feats_t = self.enc(img_source, img_drive, None)

            alpha_D = self.mlp(wa_t)
            directions_D = self.dir(alpha_D)

            directions_expD = self.mlp_exp(directions_D)

            latent_expD = wa + directions_expD  # wa + direction

            img_recon = self.dec_exp(latent_expD, None, feats)

        elif stage=='both':
            wa, wa_t, feats, feats_t = self.enc(img_source, img_drive, None)

            alpha_D = self.mlp(wa_t)
            directions_D = self.dir(alpha_D)

            directions_poseD = self.mlp_pose(directions_D)
            directions_expD = self.mlp_exp(directions_D)

            latent_poseD = wa + directions_poseD 
            img_recon = self.dec_pose(latent_poseD, None, feats)

            wa1, _, feats1, _ = self.enc(img_recon, None, None)
            latent_poseexpD = wa1 + directions_expD 
            img_recon = self.dec_exp(latent_poseexpD, None, feats1)
        elif stage == 'exp':
            wa, wa_t, feats, feats_t = self.enc(img_source, img_drive, None)

            alpha_D = self.mlp(wa_t)
            directions_D = self.dir(alpha_D)

            directions_poseD = self.mlp_pose(directions_D)

            latent_poseD = wa + directions_poseD  # wa + direction

            img_recon = self.dec_pose(latent_poseD, None, feats)
        else:
            print("---------------------------ERROR------------------------")
            exit(0)
        return img_recon

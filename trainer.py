import torch
from networks.discriminator import Discriminator
from networks.generator import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).to(
            device)
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)
        # self.dis_dir = Discriminator_dir().to(device)

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)
        # self.dis_dir = DDP(self.dis_dir, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        d_dir_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )


        self.criterion_vgg = VGGLoss().to(rank)

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_source, img_target):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        final_output = self.gen(img_source, img_target)
        fake_poseB2A = final_output['fake_poseB2A']
        fake_pose_expB2A = final_output['fake_pose_expB2A']
        fake_expA2B = final_output['fake_expA2B']
        fake_exp_poseA2B = final_output['fake_exp_poseA2B']

        fake_selfpose = final_output['fake_selfpose']
        fake_selfexp = final_output['fake_selfexp']

        img_recon_B2A = self.dis(fake_pose_expB2A)
        img_recon_A2B = self.dis(fake_exp_poseA2B)

        vgg_loss = self.criterion_vgg(fake_pose_expB2A, img_target).mean()
        vgg_loss += self.criterion_vgg(fake_exp_poseA2B, img_source).mean()
        vgg_loss_mid = self.criterion_vgg(fake_poseB2A, fake_expA2B).mean()*2

        rec_loss = self.criterion_vgg(fake_selfpose, img_source).mean()*2
        rec_loss += self.criterion_vgg(fake_selfexp, img_source).mean()*2

        l1_loss = F.l1_loss(fake_pose_expB2A, img_target)*2
        l1_loss += F.l1_loss(fake_exp_poseA2B, img_source)*2 + F.l1_loss(fake_poseB2A, fake_expA2B)

        rec_loss += F.l1_loss(fake_selfpose, img_source)
        rec_loss += F.l1_loss(fake_selfexp, img_source) 


        gan_g_loss = self.g_nonsaturating_loss(img_recon_B2A) + self.g_nonsaturating_loss(img_recon_A2B)

        g_loss = vgg_loss + l1_loss + gan_g_loss + vgg_loss_mid + rec_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, vgg_loss_mid, rec_loss, fake_poseB2A, fake_pose_expB2A, fake_expA2B, fake_exp_poseA2B

    def dis_update(self, img_target, img_source, fake_pose_expB2A, fake_exp_poseA2B):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        # d_loss = d_dir_loss
        real_img_pred = self.dis(img_target)
        recon_img_pred = self.dis(fake_pose_expB2A.detach())
        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

        real_img_pred = self.dis(img_source)
        recon_img_pred = self.dis(fake_exp_poseA2B.detach())

        d_loss += self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

        d_loss = d_loss*10
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target):
        with torch.no_grad():
            self.gen.eval()
            final_output = self.gen(img_source, img_target, 'both')
            final_output1 = final_output
        return final_output, final_output1

    def resume(self, resume_ckpt, mo='no'):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = os.path.splitext(ckpt_name)[0]
        if start_iter == 'vox':
            start_iter = 800000
        else:
            start_iter = int(start_iter)
        if start_iter == 800000:
            a = ckpt["gen"]
            dic = {}
            for k,v in ckpt["gen"].items():
                if 'enc.net_app' in k:
                    dic[k[12:]] = v
            self.gen.module.enc.net_app.load_state_dict(dic)


            dic = {}
            for k,v in ckpt["gen"].items():
                if 'enc.fc' in k:
                    dic[k[7:]] = v
            self.gen.module.mlp.load_state_dict(dic)

            dic = {}
            for k,v in ckpt["gen"].items():
                if 'dec.direction' in k:
                    dic[k[14:]] = v
            self.gen.module.dir.load_state_dict(dic)
        else:
            self.gen.module.load_state_dict(ckpt["gen"])
            self.dis.module.load_state_dict(ckpt["dis"])
            self.g_optim.load_state_dict(ckpt["g_optim"])
            self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "dis": self.dis.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )

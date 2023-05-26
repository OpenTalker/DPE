import argparse
import os
import torch
from torch.utils import data
from dataset import VoxDataset
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import os.path

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)

def write_loss(i, vgg_loss, l1_loss, g_loss, vgg_loss_mid, rec_loss, d_loss, writer):
    writer.add_scalar('vgg_loss', vgg_loss.item(), i)
    writer.add_scalar('l1_loss', l1_loss.item(), i)
    writer.add_scalar('mid_loss', vgg_loss_mid.item(), i)
    writer.add_scalar('rec_loss', rec_loss.item(), i)
    writer.add_scalar('gen_loss', g_loss.item(), i)
    writer.add_scalar('dis_loss', d_loss.item(), i)
    # writer.add_scalar('cyc_loss', cyc_loss.item(), i)

    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    import os
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    dataset = VoxDataset(args.data_root, is_inference=False)
    dataset_test = VoxDataset(args.data_root, is_inference=True)

    from torch.utils import data
    loader = data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=True,
    )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=8,
        batch_size=min(8,args.batch_size // world_size),
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        drop_last=True,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = range(args.iter)
    for idx in pbar:
        i = idx + args.start_iter

        data = next(loader)
        img_source = data['source_image']
        img_target = data['target_image']
        img_source = img_source.to(rank, non_blocking=True)
        img_target = img_target.to(rank, non_blocking=True)

        # update generator
        vgg_loss, l1_loss, gan_g_loss, vgg_loss_mid, rec_loss, fake_poseB2A, fake_expA2B = trainer.gen_update(img_source, img_target)

        # update discriminator
        gan_d_loss = trainer.dis_update(img_target, img_source, fake_poseB2A, fake_expA2B)

        if rank == 0:
            write_loss(idx, vgg_loss, l1_loss, gan_g_loss, vgg_loss_mid,rec_loss, gan_d_loss, writer)

        # display
        if i % args.display_freq == 0 and rank == 0:
            print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [mid loss: %f] [g loss: %f] [d loss: %f] [rec loss: %f]"
                  % (i, args.iter, vgg_loss.item(), l1_loss.item(), vgg_loss_mid.item(), gan_g_loss.item(), gan_d_loss.item(), rec_loss.item()))

            if rank == 0:
                data = next(loader_test)
                img_test_source = data['source_image'] 
                img_test_target = data['target_image']
                img_test_source = img_test_source.to(rank, non_blocking=True)
                img_test_target = img_test_target.to(rank, non_blocking=True)
                final_output, _ = trainer.sample(img_test_source, img_test_target)
                fake_poseB2A = final_output['fake_poseB2A']
                fake_expA2B = final_output['fake_expA2B']
                display_img(i, img_test_source, 'source', writer)
                display_img(i, fake_poseB2A, 'fake_poseB2A', writer)
                display_img(i, fake_expA2B, 'fake_expA2B', writer)
                display_img(i, img_test_target, 'target', writer)
                writer.flush()

        # save model
        if i % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)

    return


import numpy as np
from PIL import Image
def tensor2pil(tensor):
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1600000)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=3000)
    parser.add_argument("--save_freq", type=int, default=3000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)

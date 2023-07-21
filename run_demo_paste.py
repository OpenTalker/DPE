
import os
import cv2 
import lmdb
import math
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import collections
import seg_model_2
from torch.nn import functional as F
from torchvision import transforms
from morphology import dilation
from torchvision.transforms.functional import to_tensor
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from GFPGAN.gfpgan import GFPGANer

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0

def load_image1(filename, size):
    img = filename.convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0

def img_preprocessing(img_path, size):
    img = load_image1(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out
    
def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)


def create_masks(border_pixels, mask, inner_dilation=0, outer_dilation=0, whole_image_border=True):
    image_size = mask.shape[2]
    grid = torch.cartesian_prod(torch.arange(image_size), torch.arange(image_size)).view(image_size, image_size,
                                                                                         2).cuda()
    image_border_mask = logical_or_reduce(
        grid[:, :, 0] < border_pixels,
        grid[:, :, 1] < border_pixels,
        grid[:, :, 0] >= image_size - border_pixels,
        grid[:, :, 1] >= image_size - border_pixels
    )[None, None].expand_as(mask)

    temp = mask
    if inner_dilation != 0:
        temp = dilation(temp, torch.ones(2 * inner_dilation + 1, 2 * inner_dilation + 1, device=mask.device),
                        engine='convolution')

    content = temp.clone().squeeze(0)
    content = content.squeeze(0)*255
    content = content.cpu().numpy()
    content = np.array(content,np.uint8)
    temp = FillHole(content)
    temp = temp/255
    temp = torch.from_numpy(temp)
    temp = temp.unsqueeze(0)
    temp = temp.unsqueeze(0)
    temp = temp.type(torch.FloatTensor).cuda()
    mask = temp.clone()
    border_mask = torch.min(image_border_mask, temp)
    full_mask = dilation(temp, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=mask.device),
                         engine='convolution')
    if whole_image_border:
        border_mask_2 = 1 - temp
    else:
        border_mask_2 = full_mask - temp
    border_mask = torch.maximum(border_mask, border_mask_2)

    border_mask = border_mask.clip(0, 1)
    content_mask = (mask - border_mask).clip(0, 1)

    return content_mask, border_mask, full_mask


def calc_masks(inversion, segmentation_model, border_pixels, inner_mask_dilation, outer_mask_dilation,
               whole_image_border):
    background_classes = [0, 18, 16]
    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = logical_and_reduce(*[segmentation != cls for cls in background_classes])
    foreground_mask = is_foreground.float()
    content_mask, border_mask, full_mask = create_masks(border_pixels // 2, foreground_mask, inner_mask_dilation // 2,
                                                        outer_mask_dilation // 2, whole_image_border)
    size = 256
    content_mask = F.interpolate(content_mask, (size, size), mode='bilinear', align_corners=True)
    border_mask = F.interpolate(border_mask, (size, size), mode='bilinear', align_corners=True)
    full_mask = F.interpolate(full_mask, (size, size), mode='bilinear', align_corners=True)
    return content_mask, border_mask, full_mask

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def tensor2pil_mask(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).mul(255).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def paste_image_mask(             quad,    image,          dst_image,       mask,       radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    ori = dst_image.copy()

    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        mask = blurred_mask.copy()
        image_masked.putalpha(mask)
    else:
        image_masked.putalpha(mask)
    x1, y1, x2, y2 = int(quad.split(' ')[0]), int(quad.split(' ')[1]), int(quad.split(' ')[2]), int(quad.split(' ')[3])

    pasted_image = np.asarray(pasted_image)   
    other = pasted_image[y1:y2, x1:x2]     

    other = Image.fromarray(np.uint8(other))
    other = other.resize((256,256),Image.ANTIALIAS)
    mask = (1-to_tensor(mask)[None]).mul(2).sub(1).cuda()
    mask = tensor2pil(mask)
    other.putalpha(mask)     
    other.alpha_composite(image_masked)

    other = other.resize((x2 - x1,y2 - y1),Image.ANTIALIAS)
    other = other.convert("RGB")
    ori = np.array(ori)
    ori.flags.writeable = True
    ori[y1:y2, x1:x2] = other
    return ori

def video2imgs(videoPath):
    cap = cv2.VideoCapture(videoPath)    
    judge = cap.isOpened()                
    fps = cap.get(cv2.CAP_PROP_FPS)     

    frames = 1                           
    count = 1                           
    img = []
    while judge:
        flag, frame = cap.read()         
        if not flag:
            break
        else:
           img.append(frame) 
    cap.release()

    return img



def GFP(img,restorer):
    input_img = img
    _, _, restored_img = restorer.enhance(
        input_img, has_aligned=False, only_center_face=True, paste_back=True)
    return cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        model_path = args.model_path
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        seg_model_path = './checkpoints/79999_iter.pth'
        self.segmentation_model = seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
        self.segmentation_model.load_state_dict(torch.load(seg_model_path))

        model_en = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model_en,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode

        model_name = 'GFPGANv1.3'
        model_path = os.path.join('./checkpoints', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('realesrgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {model_name} does not exist.')

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)

        print('==> loading data')
        self.save_path = args.output_folder
        os.makedirs(self.save_path, exist_ok=True)


        s_img = video2imgs(args.s_path)
        d_img = video2imgs(args.d_path)

        s = []
        for i in s_img:
            img = Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
            s.append(img_preprocessing(img,256).cuda())

        d = []
        for i in d_img:
            img = Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
            d.append(img_preprocessing(img,256).cuda())


        pa_box = args.box_path
        with open(pa_box, 'r') as f:
            hw = f.readline().strip()
            four = f.readline()

        self.s_img = s
        self.d_img = d
        self.full_path = args.full_path
        self.four = four

        self.run()


    def run(self):
    
        output_dir = self.save_path

        crop_vi = os.path.join(output_dir, 'edit.mp4')
        out_edit = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        crop_vi = os.path.join(output_dir, 's.mp4')
        out_s = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        crop_vi = os.path.join(output_dir, 'd.mp4')
        out_d = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))


        hw = Image.open(os.path.join(self.full_path,'0.jpg')).size
        crop_vi = os.path.join(output_dir, 'paste.mp4')
        out_edit_paste = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, hw)

        print('==> running')
        with torch.no_grad():
            l = min(len(self.d_img),len(self.s_img))
            for i in tqdm(range(l)):
                img_target = self.d_img[i]
                img_source = self.s_img[i]
                full_img = Image.open(os.path.join(self.full_path,str(i)+'.jpg'))
                output_dict = self.gen(img_source, img_target, 'exp')
                fake = output_dict
                fake = fake.cpu().clamp(-1, 1)

                video_numpy = fake[:,:3,:,:].clone().cpu().float().detach().numpy()
                video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                video_numpy = video_numpy.astype(np.uint8)[0]
                video_numpy = cv2.cvtColor(video_numpy, cv2.COLOR_RGB2BGR)
                out_edit.write(video_numpy)

                if self.args.EN:
                    fake = GFP(video_numpy,self.restorer)
                    fake = self.transform(fake).unsqueeze(0)
                # print(fake.shape)
                # print(torch.min(fake))
                # exit(0)

                border_pixels = 50
                inner_mask_dilation = 0
                outer_mask_dilation = 50
                whole_image_border = False
                content_mask, border_mask, full_mask = calc_masks(fake.clone().cuda(), self.segmentation_model, border_pixels,
                                                                    inner_mask_dilation, outer_mask_dilation,
                                                                    whole_image_border)
                orig_img =  full_img
                full_mask_image = tensor2pil(full_mask.mul(2).sub(1))
                oup_paste = paste_image_mask(self.four, tensor2pil(fake.clone()), orig_img.copy(), full_mask_image, radius=50)
                oup_paste = cv2.cvtColor(oup_paste, cv2.COLOR_RGB2BGR)
                out_edit_paste.write(oup_paste)

                video_numpy = img_source[:,:3,:,:].clone().cpu().float().detach().numpy()
                video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                video_numpy = video_numpy.astype(np.uint8)[0]
                video_numpy = cv2.cvtColor(video_numpy, cv2.COLOR_RGB2BGR)
                out_s.write(video_numpy)

                video_numpy = img_target[:,:3,:,:].clone().cpu().float().detach().numpy()
                video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                video_numpy = video_numpy.astype(np.uint8)[0]
                video_numpy = cv2.cvtColor(video_numpy, cv2.COLOR_RGB2BGR)
                out_d.write(video_numpy)

            out_edit.release()
            out_s.release()
            out_d.release()
            out_edit_paste.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--s_path", type=str, default='./data/crop_video/video6.mp4')
    parser.add_argument("--full_path", type=str, default='./data/full_img/video6/')
    parser.add_argument("--d_path", type=str, default='./data/d.mp4')
    parser.add_argument("--box_path", type=str, default='./data/crop_video6.txt')
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--face", type=str, default='exp')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--output_folder", type=str, default='')
    parser.add_argument("--EN", action="store_true", help="can enhance the result") 
    args = parser.parse_args()

    # demo
    demo = Demo(args)

import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class VoxDataset(Dataset):
    def __init__(self, data_root, is_inference):
        # path = opt.path
        self.data_root = data_root
        self.env = None
        self.video_items =[] 
        self.person_ids = []
        self.resolution = 256
        self.is_inference = is_inference
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        if is_inference:
            self.l = 1200
        else:
            self.l = 41400
        # self.open_lmdb()

    def open_lmdb(self):
        path = self.data_root
        self.env = lmdb.open(
            os.path.join(path, str(256)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_list.txt" if self.is_inference else "train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.person_ids = self.person_ids * 100
        # print(len(self.person_ids))

        self.txn = self.env.begin(buffers=True)

    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))

        return video_items, person_ids

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict

    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            # print(key)
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = min(length, 500)
        return video_item
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def __len__(self):
        return self.l
        # return len(self.person_ids)

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        txn = self.txn
        data = {}
        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        num_frame = video_item['num_frame']

        frame_source, frame_target = self.random_select_frames(video_item)

        
        key = format_for_lmdb(video_item['video_name'], frame_source)
        img_bytes_1 = txn.get(key)
        key = format_for_lmdb(video_item['video_name'], frame_target)
        img_bytes_2 = txn.get(key)


        img1 = Image.open(BytesIO(img_bytes_1))
        data['source_image'] = self.transform(img1)

        img2 = Image.open(BytesIO(img_bytes_2))
        data['target_image'] = self.transform(img2)
        return data

import numpy as np
from PIL import Image
def tensor2pil(tensor):
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

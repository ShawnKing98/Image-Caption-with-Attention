import os

import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from typing import Iterator, List, Dict
import random

class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None, val=False):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.val = val    # whether or not to enable validation
        self.length_map = self.collect_length_map()
        self.cap_lengths = torch.tensor([x for x in self.length_map.keys()], dtype=torch.float)
        # self.min_batch_size = min([len(x) for x in self.length_map.values()])

    def collect_length_map(self) -> Dict:
        D = dict()
        for i, ann_id in enumerate(self.ids):
            caption = self.coco.anns[ann_id]['caption']
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            cap_length = len(tokens)
            if cap_length in D:
                D[cap_length].append(i)
            else:
                D[cap_length] = [i]
        return D

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        if not self.val:
            return image, target
        else:
            all_caps_ids = coco.getAnnIds(imgIds=img_id)
            all_caps = []
            for all_caps_id in all_caps_ids:
                caption = coco.anns[all_caps_id]['caption']
                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                all_caps.append(torch.Tensor(caption))
            all_caps_tensor = torch.zeros(len(all_caps), max(len(cap) for cap in all_caps)).long()
            for i, cap in enumerate(all_caps):
                end = len(cap)
                all_caps_tensor[i, :end] = cap[:end]
            return image, target, all_caps_tensor

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def collate_fn_val(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions, allcaps = zip(*data)

    images = torch.stack(images, 0)
    allcaps_lengths = [allcap.shape[1] for allcap in allcaps]
    allcaps_stack = torch.zeros(len(allcaps), 5, max(allcaps_lengths))
    for i, allcap in enumerate(allcaps):
        for j in range(5):
            allcaps_stack[i, j, :allcaps_lengths[i]] = allcap[j]
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, allcaps_stack

class FixLengthSampler(data.sampler.Sampler[int]):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        # if batch_size > self.data_source.min_batch_size:
        #    raise Exception("Batch size too large to sample from captions!")
        self.length = len(self.data_source) // self.batch_size    # drop the last batch
    
    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterator[List[int]]:
        for i in range(self.length):
            cap_lengths = self.data_source.cap_lengths
            pool_idx = torch.multinomial(cap_lengths, 1)
            cap_pool = torch.tensor(list(self.data_source.length_map.values())[pool_idx])
            idx = torch.multinomial(torch.ones(len(cap_pool)), self.batch_size, replacement=True)
            yield cap_pool[idx].tolist()

def get_loader(root, json, vocab, transform, batch_size, num_workers, val=False):
    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform, val=val)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    fix_length_sampler = FixLengthSampler(coco, batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              #batch_sampler=fix_length_sampler,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn if not val else collate_fn_val)
    return data_loader


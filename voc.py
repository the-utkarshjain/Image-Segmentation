import torch
from torch.utils import data
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms


num_classes = 21
ignore_label = 255
root = './data'
img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def class_dict():
    return {
        0  : 'background', 1  : 'aeroplane', 2  : 'bicycle', 3  : 'bird',
        4  : 'boat', 5  : 'bottle', 6  : 'bus', 7  : 'car', 
        8  : 'cat', 9  : 'hair', 10 : 'cow', 11 : 'diningtable',
        12 : 'dog', 13 : 'horse', 14 : 'motorbike', 15 : 'person',
        16 : 'potted_plant', 17 : 'sheep', 18 : 'sofa', 19 : 'train', 
        20 : 'tv_monitor'
    }

''' Collate function for dataloader '''
def collate_fn(batch):
    images = []
    masks = []
    
    for image, mask in batch:
        images.append(image)
        masks.append(mask)

    images = torch.cat(images)
    masks = torch.cat(masks)

    return images.to(device), masks.to(device)

def _make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    data_list = [l.strip('\n') for l in open(os.path.join(
        root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', f'{mode}.txt')).readlines()]
    for it in data_list:
        item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        items.append(item)
    return items

def get_train_val_test_loader(input_transform, target_transform, TF_transform= None, 
        device='cpu', bz=16, collate_fn=None):

    train_dataset =VOC('train', transform=input_transform, target_transform=target_transform, TF_transform=TF_transform, device = device)
    val_dataset = VOC('val', transform=input_transform, target_transform=target_transform, TF_transform = None, device = device)
    test_dataset = VOC('test', transform=input_transform, target_transform=target_transform, TF_transform = None, device = device)

    if TF_transform is not None:
        train_loader = DataLoader(dataset=train_dataset, batch_size= bz, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_dataset, batch_size= bz, shuffle=False, collate_fn= None)
        test_loader = DataLoader(dataset=test_dataset, batch_size= bz, shuffle=False, collate_fn= None)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size= bz, shuffle=True, collate_fn=None)
        val_loader = DataLoader(dataset=val_dataset, batch_size= bz, shuffle=False, collate_fn=None)
        test_loader = DataLoader(dataset=test_dataset, batch_size= bz, shuffle=False, collate_fn=None)
    return train_loader, val_loader, test_loader


def getClassWeights(input_transform, target_transform, TF_transform):
    """
    Uses the concept of weighing infrequent classes more in the training dataset.

    Input:
    1. dataset: the dataset of type data.Dataset
    2. useWeights: a boolean value indicating whether to compute the weights or not.

    Return:
    1. weights: a 1-d tensor of shape [21] containing weights for each class.

    The logic for computing weights: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    """
    dataset = VOC(mode='train', transform=input_transform, target_transform=target_transform, TF_transform=TF_transform)

    all_masks = []

    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        all_masks.append(torch.flatten(mask))
    
    all_masks = torch.flatten(torch.cat(all_masks))
    bincount = torch.bincount(all_masks, minlength=21)

    weights = 1 - bincount/bincount.sum()

    return weights.to(device)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class VOC(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None, TF_transform=None, device = torch.device('cpu')):
        self.imgs = _make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode

        self.transform = transform
        self.target_transform = target_transform
        self.TF_transform = TF_transform
        self.width = 224
        self.height = 224
        self.device = device

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = np.array(Image.open(img_path).convert('RGB').resize((self.width, self.height)))
        mask = np.array(Image.open(mask_path).resize((self.width, self.height)))

        if self.transform is not None:
            img = self.transform(img)
            img = img.to(self.device)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            mask = mask.to(self.device)
        if self.TF_transform is not None:
            img = self.TF_transform(img)
            mask = self.TF_transform(mask)
            img = torch.stack(img)
            mask = torch.stack(mask)

        mask[mask==ignore_label]=0


        return img, mask

    def __len__(self):
        return len(self.imgs)



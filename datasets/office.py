"""Dataset setting and data loader for Office."""
import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import cv2


class CaffeTransform(torch.utils.data.Dataset):

    def __init__(self, dataset, train=True):
        super(CaffeTransform, self).__init__()
        self.dataset = dataset
        self.samples = dataset.samples
        self.mean_color = [104.0069879317889, 116.66876761696767, 122.6789143406786]  # BGR
        self.train = train
        self.output_size = [227, 227]
        if self.train:
            self.horizontal_flip = True
            self.multi_scale = [256, 257]
        else:
            self.horizontal_flip = False
            self.multi_scale = [256, 257]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        #image, target = self.dataset.__getitem__(idx)
        path, target = self.samples[idx]
        img = cv2.imread(path)

        # Flip image at random if flag is selected
        if self.horizontal_flip and np.random.random() < 0.5:
            img = cv2.flip(img, 1)

        if self.multi_scale is None:
            # Resize the image for output
            img = cv2.resize(img, (self.output_size[0], self.output_size[0]))
            img = img.astype(np.float32)
        elif isinstance(self.multi_scale, list):
            # Resize to random scale
            new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]

            img = cv2.resize(img, (new_size, new_size))
            img = img.astype(np.float32)
            if new_size != self.output_size[0]:
                if self.train:
                    # random crop at output size
                    diff_size = new_size - self.output_size[0]
                    random_offset_x = np.random.randint(0, diff_size, 1)[0]
                    random_offset_y = np.random.randint(0, diff_size, 1)[0]
                    img = img[random_offset_x:(random_offset_x + self.output_size[0]), random_offset_y:(
                        random_offset_y + self.output_size[0])]
                else:
                    y, x, _ = img.shape
                    startx = x // 2 - self.output_size[0] // 2
                    starty = y // 2 - self.output_size[1] // 2
                    img = img[starty:starty + self.output_size[0], startx:startx + self.output_size[1]]
        img -= np.array(self.mean_color)
        img = torch.from_numpy(img)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img, target


def get_office(dataset_root, batch_size, category, train=True):
    """Get Office datasets loader."""
    # image pre-processing
    if train:
        pre_process = transforms.Compose([
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # Resize -> RandomResizedCrop 精度大幅降低，0.51 -> 0.44
    else:
        pre_process = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    # datasets and data_loader
    office_dataset_ = datasets.ImageFolder(os.path.join(dataset_root, 'office', category, 'images'))

    office_dataset = CaffeTransform(office_dataset_, train)

    office_dataloader = torch.utils.data.DataLoader(
        dataset=office_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    return office_dataloader
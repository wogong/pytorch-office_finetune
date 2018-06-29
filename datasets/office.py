"""Dataset setting and data loader for Office."""
import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data


def get_office(dataset_root, batch_size, category, train=True):
    """Get Office datasets loader."""
    # image pre-processing
    if train:
        pre_process = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                             mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)
                                         )])
        # Resize -> RandomResizedCrop 精度大幅降低，0.51 -> 0.44
    else:
        pre_process = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                             mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)
                                         )])

    # datasets and data_loader
    office_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, 'office', category, 'images'),
        transform=pre_process)

    office_dataloader = torch.utils.data.DataLoader(
        dataset=office_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)

    return office_dataloader
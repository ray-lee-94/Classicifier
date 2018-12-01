import torch
import json
import imgaug.augmenters as iaa
from torchvision import transforms, datasets

mean_path = './mean.json'
with open(mean_path, 'r') as R:
    mean = json.load(R)

data_transform = {'train': transforms.Compose([



    transforms.Resize(230),
    transforms.RandomCrop(224),
    transforms.ToTensor(),

    transforms.Normalize(mean=[mean['R'], mean['G'], mean['B']], std=[128, 128, 128])
]), 'val': transforms.Compose([

    transforms.Resize((224,224)),
    # transforms.CenterCrop(128),
    transforms.ToTensor(),
    # transforms.CenterCrop(512),
    transforms.Normalize(mean=[mean['R'], mean['G'], mean['B']], std=[128, 128, 128])
])
}

dataset = {'train': datasets.ImageFolder(root='../biger/train_roi', transform=data_transform['train']),
           'val': datasets.ImageFolder(root='../biger/test_roi',transform=data_transform['val'])}
dataset_loader = {'train': torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True, num_workers=8),
                  'val': torch.utils.data.DataLoader(dataset['val'], batch_size=64,
                                                     shuffle=True, num_workers=8)}

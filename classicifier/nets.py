import torch
import torch.nn as nn
import torchvision
from torchvision import models,transforms



choices=['resnet','alexnet','vgg','squeezenet','densenet','inception']


def set_parameter_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad=False

def initialize_model(model_name,num_classes,feature_extract,use_pretrainded=True):
    model_ft=None
    input_size=0
    if model_name=="resnet":
        model_ft=models.resnet18(pretrained=use_pretrainded)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=nn.Linear(num_ftrs,num_classes)
        input_size=224
    elif model_name=='alexnet':
        model_ft=models.alexnet(pretrained=use_pretrainded)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier[6].in_features
        model_ft.classifier[6]=nn.Linear(num_ftrs,num_classes)
        input_size=224
    elif model_name=="vgg":
        model_ft=models.vgg11_bn(pretrained=use_pretrainded)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier[6].in_features
        model_ft.classifier[6]==nn.Linear(num_ftrs,num_classes)
        input_size=224
    elif model_name==" squeezenet":
        model_ft=models.squeezenet1_0(pretrained=use_pretrainded)
        set_parameter_requires_grad(model_ft,feature_extract)
        model_ft.classifier[1]=nn.Conv2d(512,num_classes,kernel_size=(1,1),stride=(1,1))
        model_ft.num_classes=num_classes
        input_size=224
    elif model_name=="densenet":
        model_ft=models.densenet121(pretrained=use_pretrainded)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier.in_features
        model_ft.classifier=nn.Linear(num_ftrs,num_classes)
        input_size=224
    elif model_name=="inception":
        model_ft=models.inception_v3(pretrained=use_pretrainded)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc=nn.Linear(num_ftrs,num_classes)
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=nn.Linear(num_ftrs,num_classes)
        input_size=229
    else:
        print("invalid model name,exiting...")
        exit()
    return model_ft,input_size

model_ft,input_size=initialize_model(model_name='resnet',num_classes=2,feature_extract=True,use_pretrainded=True)
print(model_ft)
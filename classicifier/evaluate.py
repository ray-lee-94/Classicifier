import time
import copy
import torch
import os
import sys
sys.path.append(os.path.abspath('.'))
from  data_set import dataset_loader,dataset
# import data
from net import ResNet
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
DEVICE=torch.device('cuda:0')


def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(9):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['AE1', 'AE2', 'AE3', 'CE1', 'CE2', 'CE3', 'CE4', 'CE5', 'CL'], loc = 'upper right')
    plt.xlim(xmin=-14,xmax=14)
    plt.ylim(ymin=-14,ymax=14)
    plt.text(-13.8,7.3,"val epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)

def evaluate(model):
    for phase in ['val']:
        model.eval()
        running_corrects=0
        ip1_loader=[]
        idx_loader=[]
        for inputs,targets in dataset_loader[phase]:
            inputs=inputs.to(DEVICE)
            targets=targets.to(DEVICE)
            with torch.no_grad():
                ip1, outputs=model(inputs)
                _,preds=torch.max(outputs,1)
                ip1_loader.append(ip1)
                idx_loader.append((targets))


            running_corrects+=torch.sum(preds==targets.data)
        feat=torch.cat(ip1_loader,0)
        labels=torch.cat(idx_loader,0)
        # visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),120)
        epoch_acc= running_corrects.double()/len(dataset[phase])

    print("acc {:.4f}".format(epoch_acc.item()))

model_=ResNet.resnet50()
# CenterLoss
model_.load_state_dict(torch.load('best.pth'))
model_=model_.to(DEVICE)
model= evaluate(model=model_)

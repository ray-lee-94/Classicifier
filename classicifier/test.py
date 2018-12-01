import time
import copy
import torch
import os
import numpy as np
import sys
import torch.optim as optim
import torch.nn as nn
from centerloss import centerloss
sys.path.append(os.path.abspath('.'))
from  data_set import dataset_loader,dataset
# import data
from torch.optim import sgd

from net import resnet
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

os.environ['cuda_visible_devices'] = '7'
device=torch.device('cuda:0')


def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(9):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['ae1', 'ae2', 'ae3', 'ce1', 'ce2', 'ce3', 'ce4', 'ce5', 'cl'], loc = 'upper right')
    plt.xlim(xmin=-14,xmax=14)
    plt.ylim(ymin=-14,ymax=14)
    plt.text(-13.8,7.3,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)

def train_model(model, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_'*20)
        for phase in ['train','val']:
            if phase =='train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss=0.0
            running_corrects=0
            ip1_loader=[]
            idx_loader=[]
            for inputs,targets in dataset_loader[phase]:
                inputs=inputs.to(device)
                targets=targets.to(device)
                optimizer.zero_grad()
                optimzer4center.zero_grad()
                with torch.set_grad_enabled(true):

                    ip1, outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss = nllloss(outputs,targets) + loss_weight * centerloss(targets, ip1)
                    if phase =='train':
                        loss.backward()
                        optimizer.step()
                        optimzer4center.step()
                        ip1_loader.append(ip1)
                        idx_loader.append((targets))


                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==targets.data)
            # feat=torch.cat(ip1_loader,0)
            # labels=torch.cat(idx_loader,0)
            # visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)
            epoch_loss=running_loss/len(dataset[phase])
            epoch_acc= running_corrects.double()/len(dataset[phase])
            print('{} loss:{:.4f} acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
    best_model=copy.deepcopy(model.state_dict())
    time_elapsed=time.time()-since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    print('best val acc:{:.4f}'.format(best_acc))
    model.load_state_dict(best_model)
    return model

def visualize_model(model,num_images=6):
    was_training=model.training
    model.eval()
    images_so_far=0

    fig=plt.figure()
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(dataset_loader['val']):
            inputs=inputs.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs=model(inputs)
            _,preds=torch.max(outputs,0)
            for j in range(inputs.size[0]):
                images_so_far+=1
                ax=plt.subplot(num_images//2,2,images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(dataset['train'].classes[preds[j]]))
                # imshow(inputs.cpu().data[j])
                if images_so_far==num_images:
                    model.training(model==was_training)
                    return
        model.training(model=was_training)



model_=ResNet.resnet50()
# model_.load_state_dict(torch.load('resnet50.pth'),strict=True)
# num_ftrs=model_.fc.in_features
# model_.maxpool=torch.nn.AdaptiveMaxPool2d(2048)
# model_.fc=torch.nn.Linear(num_ftrs,2)
nllloss = nn.NLLLoss().to(DEVICE) #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
loss_weight = 0.01
centerloss = CenterLoss(9, 2).to(DEVICE)
model_=model_.to(DEVICE)
optimizer=SGD(model_.parameters(),lr=1e-2,weight_decay=1e-4,momentum=0.9)
optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)
learn_sheduler=lr_scheduler.StepLR(optimizer,step_size=30)
model= train_model(model=model_,optimizer=optimizer,scheduler= learn_sheduler,num_epochs=100)
# torch.save(model.state_dict(),'./best.pth')

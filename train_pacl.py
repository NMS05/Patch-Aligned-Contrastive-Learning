import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.image_caption_data import CocoDataset
from model.pacl import open_clip_pacl, ClipLoss

import time
import numpy as np


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    epoch_loss = []
   
    model.train()

    ###Iterating over data loader
    for i, (images, caps) in enumerate(train_data_loader):
        
        #Loading data and labels to device
        images = images.to(device)
        caps = caps.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        visual_features, text_features = model(images, caps)
        #Calculating Loss
        _loss = loss_fn(visual_features, text_features)
        epoch_loss.append(_loss.item())      
        #Backward
        _loss.backward()
        optimizer.step()
    
        if i%10 == 0: print("train_loss = ",_loss.item())

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss

def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (images, caps) in enumerate(val_data_loader):
        
            #Loading data and labels to device
            images = images.to(device)
            caps = caps.to(device)

            #Forward
            visual_features, text_features = model(images, caps)
            #Calculating Loss
            _loss = loss_fn(visual_features, text_features)
            epoch_loss.append(_loss.item())

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss


def train_clip(batch_size, epochs):
    """
    DataLoader
    """
    # Define the paths to the dataset and annotations
    train_dir = "/home/Dataset/Visual_Recognition/MSCOCO/train2017/"
    train_anno = "/home/Dataset/Visual_Recognition/MSCOCO/annotations/captions_train2017.json"
    val_dir = "/home/Dataset/Visual_Recognition/MSCOCO/val2017/"
    val_anno = "/home/Dataset/Visual_Recognition/MSCOCO/annotations/captions_val2017.json"
    # Create the dataset and dataloader
    train_dataset = CocoDataset(train_dir, train_anno, apply_transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataset = CocoDataset(val_dir, val_anno, apply_transform=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
  
    """
    Model and Loss
    """
    model = open_clip_pacl()
    device = torch.device("cuda:0")
    model = nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7])
    model.to(device)
    print("\n\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    """
    Train
    """
    loss_fn = ClipLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    print("\n\t Started Training\n")

    for epoch in range(epochs):

        begin = time.time()

        ###Training
        loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        val_loss = val_one_epoch(val_loader, model, loss_fn, device)

        print('\n\n\t Epoch....', epoch + 1)
        print("\t Training loss ......",round(loss,4))
        print("\t Val loss ......", round(val_loss,4))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')

    torch.save(model.state_dict(),'pacl_ft.pth')

if __name__=="__main__":
    train_clip(batch_size=1024, epochs=10)
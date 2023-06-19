import torch
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from tqdm import tqdm

import gc
import time
import numpy as np

import voc
from util import *
from model import *

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases


def train(model, train_loader, val_loader, criterion, scheduler, optimizer, config):
    epochs, early_stop, remark = config['epochs'], config['early_stop'], config['remark']
    print(f'Training model [{remark}] on device [{device}]')

    best_iou_score = 0.0
    best_accuracy = 0.0
    min_validation_loss = float("inf")
    patience = early_stop
    
    training_loss_history = []
    validation_loss_history = []
    early_stop_epoch = 0
    
    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        training_loss_per_epoch = 0
        
        for inputs, labels in train_loader:
            outputs =  model(inputs)
            loss = criterion(outputs, labels)
            training_loss_per_epoch += loss.item()

            # pbar.set_postfix_str(f'Train loss: {round(loss.item(), 2)}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if scheduler != None:
            scheduler.step()
            
        training_loss_per_epoch /= len(train_loader)
        
        # Evaluation on Validation    
        validation_loss, validation_iou_score, validation_accuracy = val(model, val_loader, criterion)

        # Printing and storing information
        pbar.set_description_str(f"Epoch {epoch+1}")
        pbar.set_postfix_str(
            f"Train Loss: {round(training_loss_per_epoch, 2)} Val Loss: {round(validation_loss, 2)} | " +
            f"Val IoU: {round(validation_iou_score, 2)}. Val acc: {round(validation_accuracy, 2)}")
        
        training_loss_history.append(training_loss_per_epoch)
        validation_loss_history.append(validation_loss)
        
        # Early stop mechanism
        if(validation_loss < min_validation_loss and patience > 0):
            min_validation_loss = validation_loss
            best_iou_score = validation_iou_score
            best_accuracy = validation_accuracy
            patience = early_stop
            early_stop_epoch = epoch
            save_model(model, remark + str(epoch+1))
        else:
            patience -= 1
            if (patience == 0):
                print(f"\nEarly stop at epoch {early_stop_epoch}.\n")
    
    return best_iou_score, best_accuracy, min_validation_loss, training_loss_history, validation_loss_history, early_stop_epoch


def val(model, val_loader, criterion):
    model.eval()
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs =  model(inputs)
            losses.append(criterion(outputs, labels))
            
            argmaxed_outputs = torch.argmax(outputs, dim=1)
            mean_iou_scores.append(compute_iou(argmaxed_outputs, labels))
            accuracy.append(compute_pixel_acc(argmaxed_outputs, labels))
            
    model.train()
    return torch.mean(torch.tensor(losses)).item(), torch.mean(torch.tensor(mean_iou_scores)).item(), torch.mean(torch.tensor(accuracy)).item()


def modelTest(model, test_loader, criterion, epoch, remark):
    model = load_model(model, "./trained_model/" + remark + str(epoch+1) + ".pt")
    model.eval()

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs =  model(inputs)
            losses.append(criterion(outputs, labels))
            
            argmaxed_outputs = torch.argmax(outputs, dim=1)
            mean_iou_scores.append(compute_iou(argmaxed_outputs, labels))
            accuracy.append(compute_pixel_acc(argmaxed_outputs, labels))
            
    model.train()
    return torch.mean(torch.tensor(losses)).item(), torch.mean(torch.tensor(mean_iou_scores)).item(), torch.mean(torch.tensor(accuracy)).item()



if __name__ == "__main__":
    
    # Transformations
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    target_transform = voc.MaskToTensor()

    # Dataset and Dataloader initialization
    train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform, device = device)
    val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform, device = device)
    test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform, device = device)

    train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)

    # Training variables
    epochs = 20
    n_class = 21

    fcn_model = FCN_baseline(n_class=n_class)
    fcn_model.apply(init_weights)
    fcn_model = fcn_model.to(device)

    optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.0005)
    criterion =  torch.nn.CrossEntropyLoss()

    # Training
    training_stats = train(fcn_model, train_loader, val_loader, criterion, optimizer, epochs)

    print(training_stats)
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as T
from model_2 import CNN
import random

from dataset_new import BuildingDataset

import numpy as np

import matplotlib.pyplot as plt

import os, sys

IMAGE_HEIGHT = 3024
IMAGE_WIDTH = 4032

device = "cuda" if torch.cuda.is_available() else 'cpu'
    
def split_train_val(org_train_set, valid_ratio=0.1):

    num_train = len(org_train_set)

    split = int(np.floor(valid_ratio * num_train))        

    indices = list(range(num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train - split == len(new_train_set)
    assert split == len(val_set)

    return new_train_set, val_set

def test(net, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            
            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)
    
    return 100.0 * correct / len(loader.dataset)

def permute(batch, final_size):
    """
    inputs: data - tensor (Bx3xWxH)
    outputs:
    effects: 
        1. apply the transform group to batch - there are 3 groups - 
            1 - perspective transform and crop to final size
            2 - rescale to final size (no transform)
            3 - rescale to final size and RandomErase
    """

    idx = random.randint(0,6)
    distort = 0.4

    scaler = random.random()
    rotate = torchvision.transforms.Compose([
                        T.RandomRotation(180),
                        T.RandomCrop(size = final_size)
                    ])
    perspective = torchvision.transforms.Compose([
                        T.RandomPerspective(distortion_scale=distort*scaler, p=1.0), 
                        T.RandomCrop(size = final_size)
                    ])
    random_cut = torchvision.transforms.Compose([
                        T.RandomErasing(p=1.0, scale=(0.2, 0.275), ratio = (0.2, 5)),
                        T.Resize(final_size),
                    ])
    simple_resize = T.Resize(final_size)

    transformers = [rotate, perspective, random_cut, random_cut, simple_resize, simple_resize, simple_resize]
    trans = transformers[idx]

    new_batch = trans(batch)    
    return new_batch

def train(net, loader, optimizer, epoch, device, size, log_interval=1):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        
        #transformation
        new_data = permute(data, size)
        
        # process data here. 
        new_data, target = new_data.to(device), target.to(device)
        
        # clear up gradients for backprop
        optimizer.zero_grad()
        output = F.log_softmax(net(new_data), dim=1)

        # use NLL loss
        loss = F.nll_loss(output, target)

        # compute gradients and make updates
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(new_data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()), flush=True)

    print('\tAccuracy: {:.2f}%'.format(100.0 * correct / len(loader.dataset)), flush=True)  

if __name__ == '__main__':

    # image parameters
    rf1 = 8
    rf2 = 1.25

    new_h = int(IMAGE_HEIGHT / rf1)
    new_w = int(IMAGE_WIDTH / rf1)
    final_input_h = int(IMAGE_HEIGHT / rf1 / rf2)
    final_input_w = int(IMAGE_WIDTH / rf1 / rf2)

    print(f"new_h: {new_h}, new_w: {new_w}")
    print(f"final_input_h: {final_input_h}, final_input_w: {final_input_w}")
    
    normalize = torchvision.transforms.Normalize(mean=[0.46063736, 0.47591286, 0.46565274], std=[0.225, 0.225, 0.225])
   
    #initial resize
    resize = torchvision.transforms.Resize(size = (new_h, new_w))
    resize_final = torchvision.transforms.Resize(size = (final_input_h, final_input_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)

    train_transforms = torchvision.transforms.Compose([resize, convert, normalize])
    test_transforms = torchvision.transforms.Compose([resize_final, convert, normalize])  
    
    if device == 'cpu':
        print("TRAINING ON CPU")
        train_dataset = BuildingDataset('train_labels.csv', './some_images/train', transform_pre=train_transforms)
        test_dataset = BuildingDataset('test_labels.csv', './some_images/test', transform_pre=test_transforms)
    else:
        print("TRAINING ON GPU")
        train_dataset = BuildingDataset('train_labels.csv', '/gpfs/u/home/MLA1/MLA1vnvr/scratch-shared/all_images', transform_pre=train_transforms)
        test_dataset = BuildingDataset('test_labels.csv', '/gpfs/u/home/MLA1/MLA1vnvr/scratch-shared/all_images', transform_pre=test_transforms)
    
    # # Plotting
    image = test_dataset[1][0]
    print(image.shape)
    image = image.permute(1,2,0)

    plt.figure()
    plt.imshow(image)
    #plt.imshow(torch.reshape(image, (new_h, new_w)), cmap='gray_r')
    plt.show()
    
    # set training hyperparameters
    train_batch_size = 100
    test_batch_size = 100
    n_epochs = 600
    learning_rate = 2e-2
    seed = 100
    input_dim = (3, final_input_h, final_input_w)
    print(f"input {input_dim}")
    out_dim = 11
    momentum = 0.9

    # put data into loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    network = CNN(in_dim=input_dim, out_dim=out_dim)
    network = network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 300, 400, 500], gamma=0.5)
    
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    PATH = model_path + 'cnn_buildings_3.pth'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        network.load_state_dict(torch.load(PATH))

    # sanity check -- output should be close to 1/11
    print('Initial accuracy', flush=True)
    test(network, test_loader, device)

    test_1 = False
    if test_1 == True:
        data, target = next(iter(train_loader))
        for epoch in range(1, n_epochs + 1):
            # process data here. 
            data, target = data.to(device), target.to(device)
            
            #testing transform function
            new_data = permute(data, (final_input_h, final_input_w))

            # clear up gradients for backprop
            optimizer.zero_grad()
            output = F.log_softmax(network(new_data), dim=1)

            # use NLL loss
            loss = F.nll_loss(output, target)

            # compute gradients and make updates
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            print(loss)
    else:
        # training loop
        for epoch in range(1, n_epochs + 1):
            train(network, train_loader, optimizer, epoch, device, size = (final_input_h, final_input_w))
            acc = test(network, test_loader, device)
            scheduler.step()

    torch.save(network.state_dict(), PATH)

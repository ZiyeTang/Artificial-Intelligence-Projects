# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models import resnet18

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        self.data = []
        self.labels = []
        for file in data_files:
            upk = unpickle(file)
            self.data += upk[b'data']
            self.labels += upk[b'labels']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        image = self.data[idx]
        label = self.labels[idx]

        transformed_img = []
        for i in range(32):
            row = []
            for j in range(32):
                row.append([image[i*32+j], image[1024+i*32+j], image[2048+i*32+j]])
            transformed_img.append(row)
        transformed_img = np.array(transformed_img)
        
        if self.transform is not None:
            image = self.transform(transformed_img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image,label
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """

    dataset = CIFAR10(data_files, transform=transform)
    return dataset


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    dataloader = DataLoader(dataset, **loader_params)
    return dataloader


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        backbone_model = resnet18(pretrained = True)
        backbone_model.load_state_dict(torch.load('resnet18.pt'))
        # backbone_model.eval()
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])

        for name, param in self.backbone.named_parameters(): 
                param.requires_grad_(False)

        self.classifier = nn.Linear(512, 8)
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################

        features = self.backbone(x)

        features = features.view(features.size(0), -1)

        y = self.classifier(features)
        
        return y
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(model_params, lr=hparams["lr"])
    else:
        optimizer = torch.optim.SGD(model_params, lr=hparams["lr"], momentum=hparams["momentum"])
    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################

    model.train()
    size = len(train_dataloader.dataset)
    for batch, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        
        outputs = model(inputs)
        optimizer.zero_grad()
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model, loss_fn):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    # test_loss = something
    # print("Test loss:", test_loss)
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    train_files = ["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2","cifar10_batches/data_batch_3","cifar10_batches/data_batch_4","cifar10_batches/data_batch_5"]
    test_files = ["cifar10_batches/test_batch"]
    transform = get_preprocess_transform("train")
    trainset = build_dataset(train_files, transform)
    testset = build_dataset(test_files, transform)
    
    loader_params = {"batch_size": 4, "shuffle": True}
    trainloader = build_dataloader(trainset, loader_params=loader_params)
    testloader = build_dataloader(testset, loader_params=loader_params)

    model = FinetuneNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer("Adam", model.parameters(), {"lr":0.001})

    num_epochs = 1
    for i in range(num_epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer)
        test(testloader, model, loss_fn)
    
    print("Done")
    return model
    
    

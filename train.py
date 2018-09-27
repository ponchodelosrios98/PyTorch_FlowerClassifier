# Torch Vision
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Torch
import torch
import torch.utils.data as data
import torch.nn as nn

# Tools
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import PIL
import json

##
# Arguments invoked from command line
##
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',  action='store_true', help='Use GPU in case computer has')
parser.add_argument('--data_dir', type=str, help='Path to dataset')
parser.add_argument('--epochs', type=int, help='Number of epochs (Cycles)')
parser.add_argument('--arch', type=str, help='Architecture (Ready PyTorch for more info)')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
args, _ = parser.parse_known_args()

# Data directory
if args.data_dir:
  data_dir = args.data_dir
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  ##
  # Data Structure
  ##
  data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
  }

  image_datasets = {
    'train': datasets.ImageFolder(root=data_dir + '/train', transform=data_transforms['train']),
    'test': datasets.ImageFolder(root=data_dir + '/test', transform=data_transforms['test']),
    'valid': datasets.ImageFolder(root=data_dir + '/valid', transform=data_transforms['valid'])
  }

  dataloaders = {
    'train': data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2),
    'test': data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True, num_workers=2),
    'valid': data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True, num_workers=2)
  }

##
# Sequential Model
##
def sequential_model(arch='vgg19', num_labels=0, hidden_units=4096):
    if arch=='vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    if arch=='vgg16':
        model = torchvision.models.vgg19(pretrained=True)
    elif arch=='alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    else:
        print('Using VGG19 Architecture as default')
        model = torchvision.models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(hidden_units, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return model
##
# Model Training
##
def train_model(dataset, env='train', arch='vgg19', hidden_units=4096, epochs=25, learning_rate=0.001, gpu=False, checkpoint=''):
  print('Starting model with ENV: {}'.format(env))
  dataloader = dataloaders[env]

  # Structure declaration through Command line
  if args.arch:
    arch = args.arch

  if args.learning_rate:
    learning_rate = args.learning_rate

  if args.hidden_units:
    hidden_units = args.hidden_units

  if args.epochs:
    epochs = args.epochs

  if args.checkpoint:
    checkpoint = args.checkpoint   

  num_labels = len(dataset.classes)

  model = sequential_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)

  # Use GPU if defined, otherwise use cpu
  if gpu and torch.cuda.is_available():
      device = torch.device("cuda:0")
      model.cuda()
  else:
      device = torch.device("cpu")   

  dataset_size = len(dataset)
        
  for epoch in range(epochs):
    print('Starting Epoch {} / {}'.format(epoch + 1, epochs))

    if env == 'train':
      model.train()
    else:
      model.eval()

    loss = 0.0
    correct_answers = 0

    for inputs, labels in dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
          
      optimizer.zero_grad()

      with torch.set_grad_enabled(env == 'train'):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        if env == 'train':
            loss.backward()
            optimizer.step()

      loss += loss.item() * inputs.size(0)
      correct_answers += torch.sum(preds == labels.data)

    current_loss = loss / dataset_size
    current_accuracy = correct_answers.double() / dataset_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(env, current_loss, current_accuracy))

    model.class_to_idx = image_datasets['train'].class_to_idx

  if checkpoint:
    print ('Saving directory ', checkpoint) 
    checkpoint_dict = {
      'arch': arch,
      'class_to_idx': model.class_to_idx, 
      'state_dict': model.state_dict(),
      'hidden_units': hidden_units
    }
    torch.save(checkpoint_dict, checkpoint)

  return model

if args.data_dir:
  train_model(image_datasets['train']) 
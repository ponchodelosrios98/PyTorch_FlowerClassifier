# Torch Vision
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict

# Torch
import torch
import torch.utils.data as data
import torch.nn as nn

# Tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import PIL
import json

##
# Arguments invoked from command line
##
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset')
parser.add_argument('--env',  type=str, help='train, test or valid')
parser.add_argument('--arch', type=str, help='Architecture (Ready PyTorch for more info)')
parser.add_argument('--epochs', type=int, help='Number of epochs (Cycles)')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--gpu',  action='store_true', help='Use GPU in case computer has')
parser.add_argument('--save_dir', type=str, help='Save trained model checkpoint to file')

args, _ = parser.parse_known_args()

env = 'train'

if args.env:
  env = args.env

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
    'train': data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=2),
    'test': data.DataLoader(image_datasets['test'], batch_size=4, shuffle=True, num_workers=2),
    'valid': data.DataLoader(image_datasets['valid'], batch_size=4, shuffle=True, num_workers=2)
  }

##
# Sequential Model
##
def sequential_model(arch='vgg16', num_labels=0):
  if arch == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
  if arch == 'vgg16':
    model = torchvision.models.vgg16(pretrained=True)
  elif arch == 'alexnet':
    model = torchvision.models.alexnet(pretrained=True)
  else:
    model = torchvision.models.vgg16(pretrained=True)

  for param in model.parameters():
    param.requires_grad = False
  
  features = list(model.classifier.children())[0]
  num_filters = model.classifier[0].in_features
  
  features = OrderedDict([
    ('fc1', nn.Linear(num_filters, 1000)),
    ('relu', nn.ReLU(True)),
    ('fc2', nn.Linear(1000, num_labels))
  ])

  model.classifier = nn.Sequential(features)

  return model


def train_model(dataset, dataloader, arch='vgg16', env='train', hidden_units=4096, epochs=5, learning_rate=0.001, gpu=False, checkpoint=''):
  num_labels = len(dataset.classes)
  dataset_length = len(dataloader)

  if args.arch:
    arch = args.arch
  if args.epochs:
    epochs = args.epochs
  if args.learning_rate:
    learning_rate = args.learning_rate
  if args.gpu:
    gpu = args.gpu
  if args.save_dir:
    checkpoint = args.save_dir

  print('--' * 20)
  print('Starting model with ENV: {}'.format(env))
  print('Arch: {}'.format(arch))
  print('Output Classes: {}'.format(num_labels))
  print('Epochs: {}'.format(epochs))
  print('LR: {}'.format(learning_rate))
  print('GPU: {}'.format(gpu))
  print('Save: {}'.format(checkpoint))
  print('Analyzing {} per batch'.format(dataset_length))

  model = sequential_model(arch=arch, num_labels=num_labels)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)

  if gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.cuda()
  else:
    device = torch.device("cpu")   

  dataset_size = len(dataset)
        
  for epoch in range(epochs):
    print('--' * 20)
    print('Epoch Cycle: {} / {}'.format(epoch + 1, epochs))

    if env == 'train':
      model.train()
    else:
      model.eval()

    running_loss = 0.0
    running_corrects = 0
    count = 0

    for inputs, labels in dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      count += 1
            
      optimizer.zero_grad()

      if count % 546 == 0:
        print(count)

      with torch.set_grad_enabled(env == 'train'):
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        if env == 'train':
          loss.backward()
          optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(env, epoch_loss, epoch_acc))
    model.class_to_idx = image_datasets[env].class_to_idx

  if checkpoint:
    print ('Saving directory ', checkpoint) 
    checkpoint_dict = {
      'arch': arch,
      'class_to_idx': model.class_to_idx, 
      'state_dict': model.state_dict(),
    }
    torch.save(checkpoint_dict, checkpoint)

  return model

if args.data_dir:
  train_model(dataset = image_datasets[env], dataloader = dataloaders[env]) 
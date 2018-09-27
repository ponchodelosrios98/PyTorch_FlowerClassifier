# Torch Vision
import torchvision.transforms as transforms

# Torch
import torch
import torch.utils.data as data
import torch.nn as nn

# Tools
import numpy as np
import PIL
import argparse
import json

# Local Imports
from train import sequential_model

##
# Arguments invoked from command line
##
parser = argparse.ArgumentParser()
parser.add_argument('--top_k', type=int, help='top K most likely classes')
parser.add_argument('--image_path', type=str, help='Image to test')
parser.add_argument('--checkpoint', type=str, help='From where we are getting the model')
parser.add_argument('--category_names', type=str, help='Where does model will get the categories values')
args, _ = parser.parse_known_args()

print(args)
def predict(image_path, checkpoint = 'flower_classifier.pt', category_names='cat_to_name.json', top_k=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if args.top_k:
      top_k = args.top_k
    if args.checkpoint:
      checkpoint = args.checkpoint
    if args.category_names:
      category_names = args.category_names
    
    with open(category_names, 'r') as f:
      cat_to_name = json.load(f)
  
    # Printing Information
    print('-' * 20)
    print('Image Path: {}'.format(image_path))
    print('Loading Checkpoint: {}'.format(checkpoint))
    print('Using GPU: {}'.format(gpu))
    print('Top K: {}'.format(top_k))
    print('-' * 20)

    checkpoint_dict = torch.load(checkpoint)

    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
    
    model = sequential_model(num_labels=num_labels, hidden_units=hidden_units)
    model.eval()
    
    if gpu and torch.cuda.is_available():
      model.cuda()
  
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = PIL.Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean_v = np.array([0.485, 0.456, 0.406])
    std_v = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean_v) / std_v    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    np_image = torch.autograd.Variable(torch.FloatTensor(np_image), requires_grad=True)
    np_image = np_image.unsqueeze(0)
    
    result = model(np_image).topk(top_k)

    probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
    classes = result[1].data.numpy()[0]

    print('--' * 20)
    print('Predictions (%):', list(zip(classes, probs)))
    for val in range(len(probs)):
      cat_name = cat_to_name[str(classes[val])]
      print(' - {}: {}%'.format(cat_name, probs[val] * 100))

    return probs, classes

probs, classes = predict(image_path=args.image_path)

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
parser.add_argument('--checkpoint', type=str, help='From where we are getting the model')
parser.add_argument('--top_k', type=int, help='top K most likely classes')
parser.add_argument('--image_path', type=str, help='Image to test')
parser.add_argument('--category_names', type=str, help='Where does model will get the categories values')
args, _ = parser.parse_known_args()

##
# Default Variables
##
checkpoint = 'flower_classifier.pt'
category_names = 'cat_to_name.json'

##
# Replacing Values
##
if args.checkpoint:
  checkpoint = args.checkpoint
if args.category_names:
  category_names = args.category_names

##
# Loading Category Names
##
with open(category_names, 'r') as f:
  cat_to_name = json.load(f)

##
# Model Loading
##
model_checkpoint = torch.load(checkpoint)

label_length = len(model_checkpoint['class_to_idx'])

model = sequential_model(num_labels=label_length)

model.load_state_dict(model_checkpoint['state_dict'])
model.class_to_idx = model_checkpoint['class_to_idx']

##
# Process Image
##
def process_image(image):
  img_loader = transforms.Compose([
  transforms.Resize(256), 
  transforms.CenterCrop(224), 
  transforms.ToTensor()])
  
  pil_image = PIL.Image.open(image)
  pil_image = img_loader(pil_image).float()
    
  np_image = np.array(pil_image)    
    
  mean_v = np.array([0.485, 0.456, 0.406])
  std_v = np.array([0.229, 0.224, 0.225])
  np_image = (np.transpose(np_image, (1, 2, 0)) - mean_v) / std_v    
  np_image = np.transpose(np_image, (2, 0, 1))
            
  return np_image

##
# Prediction
##
def predict(image_path, model, top_k=5):
  if args.top_k:
    top_k = args.top_k

  img = process_image(image_path)
  
  image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
  model_input = image_tensor.unsqueeze(0)
  
  probs = torch.exp(model.forward(model_input))
  
  top_probs, top_labs = probs.topk(top_k)
  top_probs = top_probs.detach().numpy().tolist()[0] 
  top_labs = top_labs.detach().numpy().tolist()[0]
  
  idx_to_class = {val: key for key, val in model.class_to_idx.items()}

  top_labels = [idx_to_class[lab] for lab in top_labs]
  top_flower_labels = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

  return top_probs, top_labels, top_flower_labels

if args.image_path:
  image_path = args.image_path
  probs, classes, flowers = predict(image_path=image_path, model=model)
  print(probs, classes, flowers)
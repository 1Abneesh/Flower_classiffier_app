# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:45:22 2023

@author: 01abn
"""



#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import PIL
import math
from torch import nn
import argparse
import json
from torch import optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
import streamlit as st

# define Mandatory and Optional Arguments for the script
# def arg_parser():
#     parser = argparse.ArgumentParser(description="predict.py")
    
#     parser.add_argument('--image',type=str,help='Point to image file for prediction.',required=True)
#     parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
#     parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
#     parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
#     parser.add_argument('--top_k',type=int,help='Choose top K matches as int.')

#     args = parser.parse_args()
#     return args


st.set_page_config(page_title="Image Input", page_icon=":camera:", layout="wide")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])
torch.nn.Module.dump_patches = True

#checking for GPU avilability
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")   
    elif gpu_arg == "cpu":
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def predict(image_path, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    model.eval();
    print('Device ' ,device)
    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(image_path, axis=0)).type(torch.FloatTensor).to(device)

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


# Scales, crops, and normalizes a PIL image for a PyTorch model,returns an Numpy array
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
        #size = 256, 256
    #loading image
    # im = PIL.Image.open (image) 
    #original size
    im = image
    width, height = im.size

    if width > height: 
        height = 256
        
    else: 
        width = 256
    im.thumbnail ((width,50000), Image.ANTIALIAS) 
    #new size of im
    width, height = im.size 
    #crop 224x224 in the center
    reduce = 224
    left = (width - reduce)/2 
    top = (height - reduce)/2
    right = left + 224 
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))
    
    #preparing numpy array
    #to make values from 0 to 1
    numpy_img = np.array(im)/255 
    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
    
    numpy_img= numpy_img.transpose ((2,0,1))
    return numpy_img



def print_probability(probs, flowers):
    st.write("### Probability Results:")
    for i, j in enumerate(zip(flowers, probs)):
        st.write("Rank {}:".format(i+1))
        st.write("Flower: {}, liklihood: {}%".format(j[1], math.ceil(j[0]*100)))


# Loading the trained model
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path) 
    model = models.vgg19(pretrained=True)
    model.name = "vgg19"
    for param in model.parameters(): 
        param.requires_grad = False
#     print(checkpoint.keys)
    # Load from checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    return model

        
#taking all the arguments
def main():
    # args = arg_parser()
    category_names = 'cat_to_name.json'
    with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    #load the trained models
    model = load_checkpoint('checkpoint.pth')
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_tensor = process_image(image)
        #checkng for available device
        device = check_gpu('gpu');
        #getting the predictions
        top_probs, top_labels, top_flowers = predict(image_tensor,model,device,3)
        print(top_probs)
        print_probability(top_flowers, top_probs)

if __name__ == "__main__":
    main()
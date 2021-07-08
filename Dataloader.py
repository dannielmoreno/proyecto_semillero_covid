#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:22:15 2021

@author: drmoreno
"""
import os
import pandas  as pd
from skimage import io, transform
import pydicom
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

#%% Dataset class

class Dataset_COVID_classify(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_df, study_df, transform=None):
        super(Dataset_COVID_classify, self).__init__()
        'Initialization'
        self.image_df = image_df
        self.study_df = study_df
        self.transform = transform

  def __len__(self):
        'Denotes the total number of studies to be classified'
        return len(self.study_df)

  def __getitem__(self, index):
        'Generates one sample of data'
        
        self.study_df.columns = [c.replace(' ', '_') for c in self.study_df.columns]
        label = [self.study_df.Negative_for_Pneumonia[index],
                  self.study_df.Typical_Appearance[index],
                  self.study_df.Indeterminate_Appearance[index],
                  self.study_df.Atypical_Appearance[index]
                  ]
        
        train_path = os.path.join("data", "train")
        study = self.study_df.id[index][:-6]
        dirs = os.listdir(os.path.join(train_path, study))
        folder = dirs[0]
        image = os.listdir(os.path.join(train_path, study, folder))[0]
        dicom_path = os.path.join(train_path, study, folder, image)
        dicom = pydicom.dcmread(dicom_path)
        
        img = dicom.pixel_array
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class Rescale(object):
   """Rescale the image in a sample to a given size.
   Args:
       output_size (tuple or int): Desired output size. If tuple, output is
           matched to output_size. If int, smaller of image edges is matched
           to output_size keeping aspect ratio the same.
   """
   def __init__(self, output_size):
       assert isinstance(output_size, (int, tuple))
       self.output_size = output_size

   def __call__(self, X):
       image = X

       h, w = image.shape[:2]
       if isinstance(self.output_size, int):
           if h > w:
               new_h, new_w = self.output_size * h / w, self.output_size
           else:
               new_h, new_w = self.output_size, self.output_size * w / h
       else:
           new_h, new_w = self.output_size

       new_h, new_w = int(new_h), int(new_w)

       img = transform.resize(image, (new_h, new_w), mode='constant')
       return img

#%% 

# Obtiene dataframes
       
image_df = pd.read_csv(os.path.join("data", "train_image_level.csv"))
study_df = pd.read_csv(os.path.join("data", "train_study_level.csv"))
    
dataset = Dataset_COVID_classify(image_df=image_df, study_df=study_df, transform=Rescale((255,255)))

image, label = dataset[3]

plt.figure()
plt.imshow(image, cmap="gray")
plt.show()

print(label)




#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import math

# Crea un dataframe de pandas a partir del archivo csv con las bounding boxes
train_df = pd.read_csv(os.path.join("data", "train_image_level.csv"))
# Toma un pequeno set de ese dataframe
sample_df = train_df.sample(5)

for i, rows in sample_df.iterrows():
    # Crea el path a una archivo dicom del dataframe
    
    train_path = os.path.join("data", "train")
    study = rows["StudyInstanceUID"]
    dirs = os.listdir(os.path.join(train_path, rows["StudyInstanceUID"]))
    folder = dirs[0]
    image = rows["id"][:-6]
    
    dicom_path = os.path.join(train_path, study, folder, image + ".dcm")
    
    # Lee el archivo dicom
    
    dicom = pydicom.dcmread(dicom_path)
    
    # Extrae la imagen
    
    img = dicom.pixel_array
    
    # Visualiza la imagen
    
    fig, ax = plt.subplots(1,1)
    ax.imshow(img, cmap = 'gray')
    plt.axis("Off")
    
    # Permite visualizacion de anotaciones de bounding boxes
    
    if pd.notna(rows["boxes"]):
        boxes = ast.literal_eval(rows["boxes"])
        for box in boxes:
            x, y, width, height = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
            rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')
            ax.add_patch(rect)
            
    plt.show()
    
    
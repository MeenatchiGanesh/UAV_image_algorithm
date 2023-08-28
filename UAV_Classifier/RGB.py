
#myfunctions.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from tqdm.notebook import tqdm
from myfunctions import fitting_rf_for_region, calculating_features_rgb, classif_imname_region

# Define paths and directories
PATH_ORIGINAL_IMAGES = "../../../Nextcloud2/"
RGB_FOLDER = "./data/drone_rgb/"
PATH_LABELS = "./data/labels_RGB_4classes.csv"
PATH_META_LANDCOVER = "./data/meta_rgb_landcover.csv"
PATH_RESULT_JPG = "./data/RGB_masks_jpg/"
PATH_RESULT_RGB = './data/RGB_masks_tiff/'

# Create output directories if they do not exist
if not os.path.exists(PATH_RESULT_RGB):
    os.makedirs(PATH_RESULT_RGB)
if not os.path.exists(PATH_RESULT_JPG):
    os.makedirs(PATH_RESULT_JPG)

# Load and preprocess data
# ...

# Loop over the images and classify them for different landcover types
for i, name_im in tqdm(enumerate(os.listdir(RGB_FOLDER)[:2])):
    print(i, name_im)
    classif_imname_region(name_im, labdat, grefiles)

print("Classification complete.")
# To measure the executable time
import time

# ...

for i, name_im in tqdm(enumerate(os.listdir(RGB_FOLDER)[:2])):
    print(i, name_im)
    start_time = time.time()
    classif_imname_region(name_im, labdat, grefiles)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

print("Classification complete.")

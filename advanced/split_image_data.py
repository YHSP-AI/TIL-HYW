
import json,re
# from datasets import Dataset, Features, Value, Array2D, Array3D
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
# Function to calculate area of a bounding box
def calculate_area(bbox):
    return bbox[2] * bbox[3]

# Read the JSONL file and extract unique categories
input_file = 'vlm.jsonl'
image_dir = 'images/'  # Directory where images are stored
images = []
unique_categories = set()

# First pass to collect unique categories
with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        images.append(data)
        for annotation in data['annotations']:
            caption = annotation['caption']

            unique_categories.add(caption)


            
# category_mapping = {category: idx for idx, category in enumerate(sorted(unique_categories))}

# print(category_mapping)



# id2cat = { v:k for k,v in category_mapping.items() }
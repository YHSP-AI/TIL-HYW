import json,re
# from datasets import Dataset, Features, Value, Array2D, Array3D
from PIL import Image
import os
import numpy as np

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

        for annotation in data['annotations']:
            caption = annotation['caption']
            unique_categories.add(caption)


category_mapping = {category: idx for idx, category in enumerate(sorted(unique_categories))}

# print(category_mapping)


id2cat = { v:k for k,v in category_mapping.items() }

with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)

        img = Image.open(os.path.join('images',data['image']))
        width, height = img.size
        img = dict( filename = os.path.abspath(os.path.join('images',data['image'])), height = height , width= width , detection = dict( instances = []))
        for annotation in data['annotations']:
            cat_name = annotation['caption']

            x1,y1,box_width,box_height = annotation['bbox']
            bbox= [x1,y1, x1 + box_width, y1+box_height]
            print(bbox)
            img['detection']['instances'].append( dict(bbox = bbox,label= category_mapping[cat_name] , category = cat_name))
        images.append(img)

# print(images[0])
import jsonlines


with jsonlines.open('mm_grounding_format.jsonl','w') as f:
    f.write_all(images)

with open('mm_grounding_dino_category_mapping.json' , 'w') as f:
    json.dump(id2cat, f)


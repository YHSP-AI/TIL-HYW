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

with open('mm_grounding_dino_category_mapping.json') as f:
    category_mapping = json.load(f)
data = dict(info = dict(   description= "TIL Data",
    url= "http://cocodataset.org",
    version= "1.0",
    year= 2017,
    contributor= "TIL",
    date_created= "2017/09/01") , licenses = [] , images = [] , annotations=[] , categories = [])


for cat_name,cat_id in category_mapping.items():
    data['categories'].append( dict(supercategory = 'TIL' , id = cat_id , name = cat_name ))



i =0




with open(input_file, 'r') as f:
    for line in f:
        original_data = json.loads(line)

        img = Image.open(os.path.join('images',original_data['image']))
        width, height = img.size
        img_id = int(original_data['image'].replace('.jpg','').split('_')[1])

        data['images'].append(dict( file_name = os.path.abspath(os.path.join('image',original_data['image'])),
                                  height = height , width= width , id  = img_id ,
                                  coco_url = '',
                                  date_captured = "2017/09/01"))


        for annotation in original_data['annotations']:
            cat_name = annotation['caption']
            data['annotations'].append( dict(bbox = annotation['bbox'],
                                             category_id= category_mapping[cat_name] ,
                                             iscrowd = 0 ,
                                             area = calculate_area(annotation['bbox']) ,
                                             image_id = img_id , id =i))
            i+=1

# print(images[0])
# import jsonlines


# with jsonlines.open('mm_grounding_format.jsonl','w') as f:
#     f.write_all(images)


print(data['images'][0])
print(data['annotations'][0])

with open('TIL COCO.json','w') as f:
    json.dump(data  , f )
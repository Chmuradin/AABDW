from pathlib import Path
import os
import json
import pandas as pd

root = r'C:/Users/Chris\Documents/1KU_Leuven/Advanced_analytics_in_a_big_data_world/AABDW/Assignment 2'
data_path = Path(root)

os.chdir(data_path)
os.listdir()

# read labels

with open('dataset.json', 'r') as f:
 
    # Reading from json file
    labels = json.load(f)

labels[0]

# parse the labels
rows_resto = []
rows_imgs = []
image_ids = set()

for resto in labels:
  full_imgs = resto['more_details']['full_images']
  if len(full_imgs) > 0:
    for img in full_imgs:
      if resto['price_category']:
        price = resto['price_category']['label']
      else:
        price = 'UNKNOWN'

      if resto['name']:
        resto_name = resto['name']
      else:
        resto_name = 'UNKNOWN'
      if resto['price_category']:
        img_id = img['image_id']
        img_info = {'image_id': img_id,
                    'resto_name': resto_name,
                    'price': price}
        rows_imgs.append(img_info)

        image_ids.add(img_id)

images = pd.DataFrame(rows_imgs)

print(images.head())

# images not labeled
os.chdir(r'./images')

raw_image_file_names = os.listdir()
stripped_image_file_names = set()

for filename in os.scandir():
    if filename.is_file():
        stripped_image_file_names.add(filename.name.rstrip('.jpg'))

non_labeled_image_ids = stripped_image_file_names.difference(image_ids)

os.chdir(str(data_path))
images.to_csv('./image_ids_and_labels.csv')

os.getcwd()
os.listdir()
print(images['price'].unique())
print(images['price'].value_counts())




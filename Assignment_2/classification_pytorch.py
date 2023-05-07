import pandas as pd
import torchvision
from torch.utils.data import Dataset,DataLoader,random_split
from pathlib import Path
from torchvision.datasets.folder import default_loader
import os
import torch
from sklearn.model_selection import train_test_split
from train import train
from utils import save_model

root = r'C:/Users/Chris\Documents/1KU_Leuven/Advanced_analytics_in_a_big_data_world/AABDW/Assignment 2'
data_path = Path(root)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

images_ids_and_labels=pd.read_csv('./image_ids_and_labels.csv')

provisional,test_set=train_test_split(images_ids_and_labels, test_size=0.15, random_state=101)

training_set,validation_set=train_test_split(provisional, test_size=0.176470588, random_state=101)

# Load image paths and labels
train_image_paths = []
val_image_paths = []
test_image_paths = []
train_labels = []
val_labels = []
test_labels = []

label_dict = {'On a budget': 0, 'A moderate spend': 1, 'Special occasion': 2, 'Spare no expense': 3}
for index, row in training_set.iterrows():
    image_id=row['image_id']
    image_path=data_path/'images'/f'{image_id}.jpg'
    label=row['price']
    label= label_dict[label]
    train_image_paths.append(image_path)
    train_labels.append(label)

for index, row in validation_set.iterrows():
    image_id=row['image_id']
    image_path=data_path/'images'/f'{image_id}.jpg'
    label=row['price']
    label= label_dict[label]
    val_image_paths .append(image_path)
    val_labels.append(label)

for index, row in test_set.iterrows():
    image_id=row['image_id']
    image_path=data_path/'images'/f'{image_id}.jpg'
    label=row['price']
    label= label_dict[label]
    test_image_paths.append(image_path)
    test_labels.append(label)



class ImageDataSet(Dataset):
    def __init__(self,image_paths, labels,transform=None,loader=default_loader):
        self.image_paths=image_paths
        self.labels=labels
        self.transform=transform
        self.loader=loader
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):
        image_path=self.image_paths[index]
        label=self.labels[index]
        image=self.loader(image_path)
        if self.transform:
            image=self.transform(image)
        return image, label


weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
auto_transforms=weights.transforms()

train_dataset=ImageDataSet(train_image_paths,train_labels,auto_transforms)
val_dataset=ImageDataSet(val_image_paths,val_labels,auto_transforms)
test_dataset=ImageDataSet(test_image_paths,test_labels,auto_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model=torchvision.models.efficientnet_v2_s(weights=weights).to(device)

#Freezing the feature extraction part of the network
for param in model.features.parameters():
    param.requires_grad=False

#Update the classifier part
from torch import nn
torch.manual_seed(101)
torch.cuda.manual_seed(101)

model.classifier=nn.Sequential(nn.Dropout(p=0.2,inplace=True), nn.Linear(in_features=1280,out_features=4)).to(device)

# print(f"FEATURES {model.features}")
# print(f"AVERAGE POOL {model.avgpool}")
# print(f"CLASSIFIER PART {model.classifier}")
# from torchinfo import summary
# summary(model=model, input_size=[32,3,1000,1000], col_names=['input_size','output_size','num_params','trainable'],col_width=20,row_settings=['var_names'])
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

#start the time
from timeit import default_timer as timer
start_time=timer()
#setting up trainin and save the results
results=train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=val_dataloader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              epochs=5,
              device=device)
#End the timer and print out the time used to train the model
end_time=timer()
print(f'[INFO] Total training time: {end_time-start_time} seconds')
save_model(model=model,target_dir='./model',model_name="categorizing_restaurants_priciness.pth")
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:26:29 2022

@author: amitn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:20:25 2022

@author: amitn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:16:50 2022

@author: amitn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:14:08 2022

@author: amitn
"""

import os
import seaborn as sns
import random
from typing import Tuple
import pandas as pd
import tensorflow as tf
#from openpyxl.workbook import Workbook
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


import torch.nn as nn
import torch.nn.functional as F


class RawData():
    def __init__(self):
        self.label = self._read_label()
        self.data = self._read_all_sensors()

    def _read_label(self):
        label = np.genfromtxt(os.path.join('profile.txt'))
        return label

    def _read_sensor(self, name):
        sensor_data = np.genfromtxt(
            os.path.join('sensors', name + '.txt')
        )
       
        # upsample the data
        if sensor_data.shape[1] == 600:
            sensor_data = sensor_data.repeat(10, axis=1)
        elif sensor_data.shape[1] == 60:
            sensor_data = sensor_data.repeat(100, axis=1)

        sensor_data= self.calculate_mean(sensor_data)
        return sensor_data

    def _read_all_sensors(self):
        sensor_names = (
            'PS1', 'PS2', 'PS3'
            , 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1','SE', 'CE', 'CP'
        )

        sensor_data = [self._read_sensor(name) for name in sensor_names]
        
        sensor_data = np.stack(sensor_data, axis=1)
        print(sensor_data.shape)
        sensor_data = sensor_data.reshape(1*2205,17)
        
        return sensor_data

    def calculate_mean(self,x):
        
        m_c = x.mean(axis=1)
        m_c = pd.DataFrame(m_c)
        
        return m_c

    def _get_split_indices(self, ):
        indices = [*range(len(self.label))]
        random.seed(1)
        random.shuffle(indices)
        random.seed(None)

        training_split = int(len(self.label) * 0.8)
        return indices[:training_split], indices[training_split:]

    def get_datasets(self):
        train_indices, val_indices = self._get_split_indices()

        train_data = self.data[train_indices]
        train_label = self.label[train_indices]

        val_data = self.data[val_indices]
        val_label = self.label[val_indices]

        # scale inputs to zero mean and unit variance
        scaler = StandardScaler()
        scaler.fit(train_data.reshape(-1, val_data.shape[1]))
        train_data_scaled = scaler.transform(
            train_data.reshape(-1, train_data.shape[1])
        ).reshape(train_data.shape)
        val_data_scaled = scaler.transform(
            val_data.reshape(-1, val_data.shape[1])
        ).reshape(val_data.shape)
        
        return train_data_scaled, train_label, val_data_scaled, val_label


class HydraulicSystemDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.from_numpy(inputs)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        return input, label


def get_dataloader(batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader,
                                                  torch.utils.data.DataLoader]:
    """to get the dataloader for the hydraulic system

    Args:
        batch_size (int, optional): [batch-size]. Defaults to 32.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        training and testing loader to use during training
    """
    Dataset = RawData()
    # Display image and label.
    
    train_features, train_labels, test_features, test_labels = Dataset.get_datasets()
    
    return train_features, train_labels, test_features, test_labels

X_train,X_test, y_train, y_test = get_dataloader()


xxxx = torch.tensor(0.2)
#### Creating Modelwith Pytorch

class DFNN_Model(nn.Module):
    def __init__(self,input_features=17,hidden1=15,hidden2=13,hidden3=11,hidden4=9,out_features=5):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.f_connected3=nn.Linear(hidden2,hidden3)
        self.f_connected4=nn.Linear(hidden3,hidden4)
        self.out=nn.Linear(hidden4,out_features)
    def forward(self,x):
        x=F.prelu(self.f_connected1(x),xxxx)
        x=F.prelu(self.f_connected2(x),xxxx)
        x=F.prelu(self.f_connected3(x),xxxx)
        x=F.prelu(self.f_connected4(x),xxxx)
        x=self.out(x)
        return x


####instantiate my ANN_model
#torch.manual_seed(20)
model= DFNN_Model()
print(model)
print(model.parameters)

###Backward Propogation-- Define the loss_function,define the optimizer
loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0009)

epochs=1000
final_losses=[]
test_losses = []
k = 0
for i in range(epochs):
    k=k+1
    y_pred=model.forward(torch.tensor(X_train).float())
    loss=loss_function(y_pred,(torch.FloatTensor(X_test)))
    y_pred2 = model.forward(torch.tensor(y_train).float())
    loss2 = loss_function(y_pred2,torch.tensor(y_test).float())
    final_losses.append(loss)
    test_losses.append(loss2)
    if k%10==1:
        print("Epoch number: {} and the loss : {}".format(k,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Commented out IPython magic to ensure Python compatibility.
### plot the loss function 

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(range(epochs),final_losses,label='Training loss')
plt.legend()
plt.plot(range(epochs),test_losses, label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

final_losses = np.array(torch.FloatTensor(final_losses))
test_losses = np.array(torch.FloatTensor(test_losses))
excel1=pd.DataFrame(final_losses)
excel1.to_excel(excel_writer= "Train Comparison D.xlsx")

excel2=pd.DataFrame(test_losses)
excel2.to_excel(excel_writer= "Val Comparison D.xlsx")

#### Prediction In X_test data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(torch.tensor(X_train).float()):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        
        




from sklearn.metrics import confusion_matrix
cm=confusion_matrix(X_train,predictions)
cm


plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

from sklearn.metrics import accuracy_score
def accuracy(x_data,y_data):
    with torch.no_grad():
        output = model(x_data)


        
    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    print(predictions)
    train_acc = accuracy_score(y_data, predictions)
    print('The accuracy is: {}'.format(round((train_acc.item())*100)))
    return train_acc, predictions

train_accuracy, predict = accuracy(X_train,y_test)
validation_acc = accuracy(y_train, y_test)



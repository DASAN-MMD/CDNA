import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os,sys
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
import copy
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
from sklearn.metrics import classification_report
import time
import pickle
import argparse
import random
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
import lmmd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epoch_num_Encoder = 0
epoch_num_Aligner = 0
learning_rate = 0.00146

input_size1 = 16 #468  # Input size with 114 sub-carriers for each antenna pair with 4 such pairs
input_size2 = 100
hidden_size1 = 100  # Hidden layer Size
hidden_size2 = 16
output_dim = 50   # Output size for predicting 7 activities
num_layers = 4

BATCH_SIZE = batch_size_Tr = 420 #840 #420 #140   #280 #210             # Change here
batch_size_T =  420 #840 #420 #140 #210
batch_size1 = 10
load_model=False
GAMMA = 10 ^ 3
num_epoch = 150

classes = ('Sit still on a chair', 'Falling down', 'Lie down', 'Stand still', 'Walking from Tx to Rx', 'Walking from Rx to Tx', 'Pick a pen from ground')
#classes = ('bending', 'falling', 'lie_down', 'running', 'sit_down', 'stand_up', 'walking')
#classes = ('bending', 'falling', 'lie_down', 'running', 'sit_down', 'stand_up', 'walking')

train_dataset_path="./Dataset3/E123/S22"
test_dataset_path="./Dataset3/E123/S12"

#mean=[0.9442, 0.8749, 0.4495] #E23
#mean = [0.9273, 0.8921, 0.4653] #E13
#mean = [0.9316, 0.8801, 0.4614] #E12
#mean = [0.9399, 0.8869, 0.4534] #S21+S22
#mean = [0.9486, 0.8628, 0.4456] #S11+S12
#mean = [0.9146,0.8972,0.4772] #S1+S2

#mean = [0.9336, 0.8838, 0.4593] #S1_S11
#mean = [0.9295, 0.8767, 0.4635] #S2_S12
#mean = [0.9295, 0.8961, 0.4629] #S1_S21
#mean = [0.9251, 0.8881, 0.4677] #S2_S22
#mean = [0.9454, 0.8741, 0.4484] #S11_S21
mean = [0.9430, 0.8760, 0.4506] #S12_S22

#std=[0.0836, 0.1631, 0.4550] #E23
#std=[0.0984, 0.1496, 0.4445]  #E13
#std= [0.0956, 0.1628, 0.4472] #E12
#std=[0.0865, 0.1499, 0.4522] #S21+S22
#std=[0.0808, 0.1764, 0.4578] #S11+S12
#std=[0.1102,0.1492,0.4367] #S1+S2

#std = [0.0938, 0.1607, 0.4485] #S1_S11
#std = [0.0974, 0.1644, 0.4459] #S2_S12
#std = [0.0965, 0.1480, 0.4459] #S1_S21
#std = [0.1002, 0.1511, 0.4430] #S2_S22
#std = [0.0823, 0.1628, 0.4556] #S11_S21
std = [0.0852, 0.1630, 0.4542] #S12_S22

#train_dataset_path = "./Dataset_S1_S2/training/training"
#test_dataset_path = "./Dataset_S1_S2/validation/validation"

#For Dataset_S1_S2_S3
#mean=[0.7316, 0.8197, 0.6564] #S123
#mean = [0.7316, 0.8197, 0.6564] #S12
#mean = [0.7316, 0.8197, 0.6564] #S13
#mean =  [0.7316, 0.8197, 0.6564] #S23

#std=[0.2945, 0.1452, 0.2873] #S123
#std= [0.2945, 0.1452, 0.2873] #S12
#std=  [0.2945, 0.1452, 0.2873] #S13
#std= [0.2945, 0.1452, 0.2873] #S23

#For Dataset_H
#train_dataset_path="./Dataset_H/S1_S47"
#test_dataset_path="./Dataset_H/S2_S22"

#train_dataset_path="./Dataset2'/S1"
#test_dataset_path="./Dataset2'/S2"

#mean=[0.9442, 0.8749, 0.4495] #E23
#mean = [0.9273, 0.8921, 0.4653] #E13
#mean = [0.9316, 0.8801, 0.4614] #E12
#mean = [0.9399, 0.8869, 0.4534] #S21+S22
#mean = [0.9486, 0.8628, 0.4456] #S11+S12
#mean = [0.9146,0.8972,0.4772] #S1+S2

#std=[0.0836, 0.1631, 0.4550] #E23
#std=[0.0984, 0.1496, 0.4445]  #E13
#std= [0.0956, 0.1628, 0.4472] #E12
#std=[0.0865, 0.1499, 0.4522] #S21+S22
#std=[0.0808, 0.1764, 0.4578] #S11+S12
#std=[0.1102,0.1492,0.4367] #S1+S2

train_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
'''
test_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))  
])
'''

train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=train_transforms)

#print(len(train_dataset))  #420  #406
#print(len(test_dataset))  #420   #415

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_Tr,shuffle=True, num_workers=4)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_T,shuffle=False, num_workers=4)

'''val_pct1=0.8
rand_seed=42

n_val1=int(val_pct1*len(test_dataset))
np.random.seed(rand_seed)
idxs=np.random.permutation(len(test_dataset))

n_val2 = int(n_val1*0.8)

test_sampler1=SubsetRandomSampler(idxs[:n_val1])
test_loader1 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=test_sampler1) #Use for model training with 0.8 per target samples

test_sampler2=SubsetRandomSampler(idxs[:n_val2])
test_loader2 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=test_sampler2) # Use for model training with 0.6 per target samples

test_sampler3=SubsetRandomSampler(idxs[n_val1:])
test_loader3 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=test_sampler3) # Use for model testing with 0.2 per target samples
'''
examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape, labels.shape)
samples = samples.reshape([420,12288])
#print(samples.type)
#df = [labels,samples]
samples = pd.DataFrame(samples)
labels = pd.Series(labels)
#print(samples.head())
#df = [labels,samples]
df_train = pd.concat([labels,samples],axis=1)
#df = pd.DataFrame(df)
#print(df.head())
#print(df.iloc[:, 1:].values)


examples = iter(test_loader)
samples, labels = examples.next()
#print(samples.shape, labels.shape)
samples = samples.reshape([420,12288])
#print(samples.type)
#df = [labels,samples]
samples = pd.DataFrame(samples)
labels = pd.Series(labels)
#print(samples.head())
#df = [labels,samples]
df_test = pd.concat([labels,samples],axis=1)

class MNIST(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.images = df.iloc[:, 1:].values.astype(np.float32)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values
        else:
            self.images = df.values.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item].reshape(64, 64, 3)

        if self.is_train:
            anchor_label = self.labels[item]

            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(64, 64, 3)

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(64, 64, 3)

            #if self.transform:
            #anchor_img = self.transform(self.to_pil(anchor_img))
            #positive_img = self.transform(self.to_pil(positive_img))
            #negative_img = self.transform(self.to_pil(negative_img))
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            #if self.transform:
            #anchor_img = self.transform(self.to_pil(anchor_img))
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img

train_dataset = MNIST(df_train,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True, num_workers=4)

test_dataset = MNIST(df_test,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))

val_pct1=  0.168 #0.084 #0.168 #0.5
rand_seed=42

n_val1=int(val_pct1*len(test_dataset))
np.random.seed(rand_seed)
idxs=np.random.permutation(len(test_dataset))

n_val2 = int(n_val1*0.72)

test_sampler1=SubsetRandomSampler(idxs[:n_val1])
test_loader1 = DataLoader(test_dataset, batch_size=batch_size1, num_workers=4, sampler=test_sampler1) #Use for model training with 0.8 per target samples

test_sampler2=SubsetRandomSampler(idxs[:n_val2])
test_loader2 = DataLoader(test_dataset, batch_size=batch_size1, num_workers=4, sampler=test_sampler2) # Use for model training with 0.6 per target samples

test_sampler3=SubsetRandomSampler(idxs[n_val1:])
test_loader3 = DataLoader(test_dataset, batch_size=batch_size1, num_workers=4, sampler=test_sampler3) # Use for model testing with 0.2 per target samples

#print(len(train_loader))
#print(len(test_loader2))

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class AE(nn.Module):
    def __init__(self,input_size1,input_size2,hidden_size1,hidden_size2, num_layers):
    #def __init__(self):
        super(AE, self).__init__()
        latent_size = 2000
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.unpool = nn.UpsamplingNearest2d(64)
        #self.dropout = nn.Dropout(0.3)
        self.rnn1 = nn.GRU(input_size1, hidden_size1, num_layers, batch_first=True)

        # Decoder

        self.rnn2 = nn.GRU(input_size2, hidden_size2, num_layers, batch_first=True)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        self.fc1 = nn.Linear(64*100,latent_size)
        self.fc2 = nn.Linear(latent_size,64*100)
        self.latent = None

    def encode(self, x):
        h01 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        #print(x.shape)
        #x = self.dropout(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 64, 16)
        # print(x.shape)
        x, _ = self.rnn1(x, h01)
        #print(x.shape)
        x = x.reshape(-1, 6400)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        self.latent = x
        return x

    def decode(self, x):
        h02 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(device)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = x.view(-1, 64, 100)
        x, _ = self.rnn2(x, h02)
        # print(x.shape)
        x = x.view(-1, 4, 16, 16)
        x = F.leaky_relu(self.t_conv1(x))
        #x = self.unpool(x)
        #print(x.shape)
        x = F.leaky_relu(self.t_conv2(x))
        #x = self.unpool(x)
        #print(x.shape)
        return x

    def forward(self, x):
        x = self.encode(x)
        # print(x.shape)
        x = self.decode(x)
        return x

class Aligner(nn.Module):
    def __init__(self):
        super(Aligner, self).__init__()
        # self.image_size = 14
        self.image_size = 64
        #self.batch_size = 1
        # read layer
        self.fc1 = nn.Linear(3*self.image_size * self.image_size, 3*self.image_size * self.image_size)

        # exp unit
        self.relu = nn.ReLU(.5)

        # out layer
        self.fc2 = nn.Linear(3*self.image_size * self.image_size, 3*self.image_size * self.image_size)

    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.view(self.batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(self.batch_size, 3, self.image_size, self.image_size)
        return x

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(2000, 500)
        self.fc1 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU(.5)
        #self.lmmd_loss = lmmd.LMMD_loss(class_num=7)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x1 = self.relu(x)
        x = self.fc4(x1)
        #loss_lmmd = self.lmmd_loss.get_loss(x1, t1, s_label, t_pred)
        #return F.log_softmax(x, dim=1), loss_lmmd

        return F.log_softmax(x, dim=1), x1 #for MMD

def save_checkpoint(state,filename="my_checkpoint.pth"):
    print("=> Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    AE_model.load_state_dict(checkpoint["state_dict"])
    AE_optimiser.load_state_dict(checkpoint['optimizer'])

AE_model = AE(input_size1,input_size2,hidden_size1,hidden_size2,num_layers).to(device)

classifier = Classifier().to(device)

aligner = Aligner().to(device)
discriminator = AE(input_size1,input_size2,hidden_size1,hidden_size2,num_layers).to(device)

AE_optimiser = torch.optim.Adam(AE_model.parameters(),lr= 0.001)
#AE_optimiser = torch.optim.SGD(discriminator.parameters(), lr=0.001, momentum= 0.9)


#classifier_optimiser = torch.optim.Adam(classifier.parameters(), lr=0.001)
classifier_optimiser = torch.optim.SGD(classifier.parameters(),lr= 0.001, momentum= 0.9)


aligner_optimiser = torch.optim.Adam(aligner.parameters(), lr=0.0002)
#aligner_optimiser = torch.optim.SGD(aligner.parameters(), lr=0.0002, momentum= 0.9)

discriminator_optimiser = torch.optim.Adam(discriminator.parameters(), lr=0.002)
#discriminator_optimiser = torch.optim.SGD(discriminator.parameters(), lr=0.0002, momentum= 0.9)

AE_criterion = nn.MSELoss()
classifier_criterion = nn.CrossEntropyLoss()
aligner_criterion = nn.MSELoss()
triplet_criterion = torch.jit.script(TripletLoss())

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth"))
    load_checkpoint(torch.load("my_checkpoint1.pth"))
    load_checkpoint(torch.load("my_checkpoint2.pth"))
    load_checkpoint(torch.load("my_checkpoint3.pth"))

def discrepancy(out1, out2):
    return torch.mean(torch.abs(out1 - out2))

fixed_ae_train_losses = []
total_train_losses = []
fixed_ae_test_losses = []
total_test_losses = []
cl_tr_losses = []
cl_tr_accuracy = []
cl_test_losses = []
cl_test_accuracy = []
overall_accuracy = []
ae_train_accuracy = []
ae_train_losses = []
cl_train_losses = []
NUM_EXP = 1

training_start_time = time.time()
for ne in range(NUM_EXP):
    # Initialize adjust_loss
    adjust_loss = torch.tensor(1, dtype=torch.float)
    patience, trials, best_acc = 5, 0, 1000
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        total_tr_losses = []
        cl_tr_loss = []
        ae_tr_loss = []
        size = 0
        correct = 0
        ae_running_loss = 0.0
        total_running_loss = 0.0
        cl_running_loss = 0.0

        # Training
        AE_model.train()
        classifier.train()
        aligner.train()
        #discriminator.train()

        for i, data in enumerate(train_loader, 0):
        #for i in range(1):
            # get the inputs; data is a list of [inputs, labels]
            _adjust_loss = adjust_loss
            inputs, positive_img, negative_img, labels = data
            orig_dataset_iter = iter(test_loader2)
            inputs_T, positive_img_T, negative_img_T, _ = orig_dataset_iter.next()

            # I CHANGED HERE
            inputs = inputs.to(device)
            labels = labels.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            # I CHANGED HERE
            inputs_T = inputs_T.to(device)
            # labels_S = labels.to(device)
            positive_img_T = positive_img_T.to(device)
            negative_img_T = negative_img_T.to(device)

            # zero the parameter gradients
            AE_optimiser.zero_grad()
            classifier_optimiser.zero_grad()

            # forward + backward + optimize

            # calc auto encoder loss
            outputs = AE_model(inputs)
            ae_loss1 = AE_criterion(inputs, outputs)
            ae_loss = ae_loss1

            # calc classifier loss
            cl_ae_outputs = AE_model(inputs)
            cl_ae_loss = AE_criterion(inputs, outputs)
            ae_tr_loss.append(cl_ae_loss)

            cl_outputs, _ = classifier(AE_model.latent.clone())
            cl_loss = classifier_criterion(cl_outputs, labels)
            cl_tr_loss.append(cl_loss)

            # calc classifier loss                      #please see here for triplet implementation
            #cl_outputs_A = AE_model(inputs_T)
            #anchor = classifier(AE_model.latent.clone())

            cl_ae_outputs_P = AE_model(positive_img)
            positive_out, _ = classifier(AE_model.latent.clone())

            # calc classifier loss
            cl_ae_outputs_N = AE_model(negative_img)
            negative_out, _ = classifier(AE_model.latent.clone())

            # Triplet loss for Source Images
            TL_source = triplet_criterion(cl_outputs, positive_out, negative_out)
            #TL_source = triplet_criterion(anchor, positive_out, negative_out)

            #loss_discrepancy = discrepancy(cl_outputs, positive_out)


            # calc total loss
            #total_loss = _adjust_loss * cl_ae_loss + cl_loss + (0.1*TL_source) #+ (0.1*loss_discrepancy)
            total_loss =  cl_loss + (0.1 * TL_source)  # + (0.1*loss_discrepancy)
            total_tr_losses.append(total_loss)
            total_loss.backward()

            AE_optimiser.step()
            classifier_optimiser.step()

            ae_running_loss += ae_loss.item()#* inputs.size(0)
            cl_running_loss += cl_loss.item()  # * inputs.size(0)
            total_running_loss += total_loss.item()#* inputs.size(0)

            _, predicted = cl_outputs.max(1)
            correct += (predicted == labels).sum().item()
            size += labels.size(0)


        ae_running_loss = ae_running_loss / (len(train_loader))
        total_running_loss = total_running_loss / (len(train_loader))
        cl_running_loss = cl_running_loss / (len(train_loader))
        ae_train_losses.append(ae_running_loss)
        cl_train_losses.append(cl_running_loss)
        tr_accuracy = float(correct / size)
        ae_train_accuracy.append(tr_accuracy*100)
        print(f'Epoch:{epoch + 1}, Auto_Encoder Loss:{ae_running_loss:.4f},Total Auto_En+Classifier Loss:{total_running_loss:.4f}, Training Accuracy:{tr_accuracy:.4f}')

        if (total_running_loss < best_acc):
            trials = 0
            best_acc = total_running_loss
            # torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pth')
            # logging.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            epoch_num_Encoder = epoch+1

        else:
            epoch_num_Encoder = epoch + 1
            if (tr_accuracy > 0.9):
                trials += 1
                if trials >= patience:
                    # logging.info(f'Early stopping on epoch {epoch}')
                    break

time1 = time.time()-training_start_time
time1 = time1/60
print('Encoder_Training_Time:{:.2f}'.format(time1))
print('Finished Encoder Training')
print('Encoder Training Stops at epoch:', epoch_num_Encoder )

#checkpoint={'state_dict':AE_model.state_dict(),'optimizer':AE_optimiser.state_dict()}
#save_checkpoint(checkpoint)

model = copy.deepcopy(AE_model.state_dict())
discriminator.load_state_dict(model)
ones = torch.tensor(1, dtype=torch.float)
minus_ones = ones * -1

#def mmd_loss(x_src, x_tar):
#    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])

theta1 = 1
theta2 = 0.5
patience, trials, best_acc = 5, 0, 100  #5, 0, 0

aligner_train_losses = []
aligner_train_accuracy = []
discriminator_train_losses = []
aligner_test_losses = []
aligner_test_accuracy = []

training_start_time = time.time()
for epoch in range(num_epoch):  # loop over the dataset multiple times
    tr_aligner_losses = []
    tr_discriminator_losses = []
    tr_classifier_losses = []
    test_classifier_losses = []
    size = 0
    correct = 0
    AE_model.train()
    classifier.train()
    aligner.train()
    discriminator.train()


    for i, data in enumerate(test_loader2, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, positive_img, negative_img, labels = data
        orig_dataset_iter = iter(train_loader)
        inputs_S, _, _, labels_S = orig_dataset_iter.next()

        # I CHANGED HERE
        inputs = inputs.to(device)
        labels = labels.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        # I CHANGED HERE
        inputs_S = inputs_S.to(device)
        labels_S = labels.to(device)
        # positive_img_S = positive_img.to(device)
        # negative_img_S = negative_img.to(device)

        # Train aligner
        aligner_optimiser.zero_grad()
        #discriminator_optimiser.zero_grad()
        aligned = aligner(inputs)
        #aligned_img_tensor = aligned.detach()
        #outputs_align = discriminator(aligned)

        aligned_P = aligner(positive_img)
        #aligned_img_tensor_P = aligned_P.detach()
        #outputs_align_P = discriminator(aligned_P)

        aligned_N = aligner(negative_img)
        #aligned_img_tensor_N = aligned_N.detach()
        #outputs_align_N = discriminator(aligned_N)

        #cl_ae_outputs_A = AE_model(inputs_S)
        #anchor_out_S, s1 = classifier(AE_model.latent.clone())

        #cl_ae_outputs_A_T = AE_model(inputs)
        #anchor_out_T, _ = classifier(AE_model.latent.clone())

        # classified
        ae_out = AE_model(aligned)
        cl_outputs, t1 = classifier(AE_model.latent.clone())
        cl_loss = classifier_criterion(cl_outputs, labels)
        tr_classifier_losses.append(cl_loss.item())
        _, predicted = cl_outputs.max(1)
        correct += (predicted == labels).sum().item()
        size += labels.size(0)
        #print(labels_S)

        ae_out_P = AE_model(aligned_P)
        positive_out, _ = classifier(AE_model.latent.clone())

        ae_out_N = AE_model(aligned_N)
        negative_out, _ = classifier(AE_model.latent.clone())

        # Triplet loss for Target Images
        #TL_target = triplet_criterion(anchor_out_S, positive_out, negative_out)
        TL_target = triplet_criterion(cl_outputs , positive_out, negative_out)

        #loss_mmd = mmd_loss(s1, t1)
        #lmmd_loss = lmmd.LMMD_loss(class_num=7)
        #loss_lmmd = lmmd_loss.get_loss(s1, t1, labels_S, cl_outputs)

        #Discrepancy Loss
        #loss_discrepancy = discrepancy(anchor_out_T, cl_outputs)
        loss_discrepancy = discrepancy(cl_outputs, positive_out)

        # loss
        #loss_align_1 = aligner_criterion(outputs_align, aligned)
        #loss_align = cl_loss + loss_align_1 + (0.1*loss_discrepancy) + (0.1*TL_target) #+ (0.01*loss_lmmd) + (0.1*loss_mmd)
        loss_align = cl_loss + (0.1 * loss_discrepancy) + (0.1 * TL_target)  # + (0.01*loss_lmmd) + (0.1*loss_mmd)
        # loss_align = cl_loss + torch.mean((torch.abs(outputs_align - aligned)))

        tr_aligner_losses.append(loss_align.item())
        loss_align.backward(retain_graph=True)
        aligner_optimiser.step()

    accuracy = float(correct/ size)
    #print(accuracy)
    avg_aligner_loss = float(sum(tr_aligner_losses) / len(tr_aligner_losses))
    #avg_disc_loss = float(sum(tr_discriminator_losses) / len(tr_discriminator_losses))
    avg_cl_loss = float(sum(tr_classifier_losses) / len(tr_classifier_losses))

    aligner_train_losses.append(avg_aligner_loss)
    #discriminator_train_losses.append(avg_disc_loss)
    aligner_train_accuracy.append(accuracy * 100)
    #print(f'Train Epoch: [{epoch}] Align-loss: {loss_align} Accuracy: {accuracy} ({correct}/{size})')
    print(f'Epoch:{epoch + 1}, avg_aligner_loss:{avg_aligner_loss:.4f}, Training Accuracy:{accuracy:.4f}')

    if (avg_aligner_loss < best_acc):
        trials = 0
        best_acc = avg_aligner_loss  #avg_aligner_loss
        # torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pth')
        # logging.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        epoch_num_Aligner = epoch + 1
    else:
        epoch_num_Aligner = epoch + 1
        if (accuracy > 0.94):
            trials += 1
            if trials >= patience:
                # logging.info(f'Early stopping on epoch {epoch}')
                break

time2 = time.time()-training_start_time
time2 = time2/60
time3 = time1 + time2
print('Aligner_Training_Time:{:.2f}'.format(time2))
print('Finished Aligner Training')
print('Aligner Training Stops at epoch:', epoch_num_Aligner)
print('Training_Time:{:.2f}'.format(time3))
#################### END of Validation

#data_target = iter(test_loader2).next()
with torch.no_grad():
    correct = 0
    size = 0
    raw_correct = 0
    raw_size = 0
    #n_class_correct = [0 for i in range(7)]
    #n_class_samples = [0 for i in range(7)]
    all_preds = torch.tensor([]).to(device)
    all_preds_raw = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    accuracyT = []
for i, data_target in enumerate(test_loader3, 0):
    inputs_t, _, _, labels_t = data_target

    inputs = inputs_t
    labels = labels_t

    # I CHANGED HERE
    inputs = inputs.to(device)
    labels = labels.to(device)

    all_labels = torch.cat((all_labels, labels), dim=0)  # For Plotting Purpose in CMT & Hist

    AE_model.eval()
    classifier.eval()
    aligner.eval()
    discriminator.eval()

    # classified aligned images
    aligned = aligner(inputs)
    outputs = AE_model(aligned)
    cl_outputs, _ = classifier(AE_model.latent.clone())
    _, predicted = cl_outputs.max(1)
    all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT
    correct = (predicted == labels).sum().item()
    size = labels.size(0)

    # classified raw(non-aligned) images
    raw_outputs = AE_model(inputs)
    raw_cl_outputs, _ = classifier(AE_model.latent.clone())
    _, raw_predicted = raw_cl_outputs.max(1)
    all_preds_raw = torch.cat((all_preds_raw, raw_predicted), dim=0)  # For Plotting Purpose in CMT
    raw_correct = (raw_predicted == labels).sum().item()
    raw_size = labels_t.size(0)


accuracy = float(correct / size)
#print(accuracy)

raw_accuracy = float(raw_correct / raw_size)
#print(raw_accuracy)

overall_accuracy.append(('Aligned Images', accuracy, 'Non-aligned images', raw_accuracy))

'''for i in range(7):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc} %')'''

print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=classes, zero_division=0))
print(classification_report(all_labels.cpu().numpy(), all_preds_raw.cpu().numpy(), target_names=classes,zero_division=0))

cm1 = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())  # Making Confusion Matrix
cm2 = confusion_matrix(all_labels.cpu().numpy(), all_preds_raw.cpu().numpy())  # Making Confusion Matrix
print('f1_score1(micro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='micro')))
print('f1_score1(macro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')))
print('f1_score2(micro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds_raw.cpu().numpy(), average='micro')))
print('f1_score2(macro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds_raw.cpu().numpy(), average='macro')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--pickle1")
    parser.add_argument("-p2", "--pickle2")
    parser.add_argument("-p3", "--pickle3")
    #parser.add_argument("-p4", "--pickle4")
    parser.add_argument("-p5", "--pickle5")
    parser.add_argument("-p6", "--pickle6")
    parser.add_argument("-p7", "--pickle7")
    parser.add_argument("-p8", "--pickle8")
    '''

    parser.add_argument("-s1", "--saved1")
    parser.add_argument("-s2", "--saved2")
    parser.add_argument("-s3", "--saved3")
    parser.add_argument("-s4", "--saved4")
    '''

    args = parser.parse_args()

    pickle_file1 = args.pickle1
    pickle_file2 = args.pickle2
    pickle_file3 = args.pickle3
    #pickle_file4 = args.pickle4
    pickle_file5 = args.pickle5
    pickle_file6 = args.pickle6
    pickle_file7 = args.pickle7
    pickle_file8 = args.pickle8
    '''

    saved_file1 = args.saved1
    saved_file2 = args.saved2
    saved_file3 = args.saved3
    saved_file4 = args.saved4

    checkpoint={'state_dict':AE_model.state_dict(),'optimizer':AE_optimiser.state_dict()}
    save_checkpoint(checkpoint,filename=saved_file1)
    checkpoint={'state_dict':classifier.state_dict(),'optimizer':classifier_optimiser.state_dict()}
    save_checkpoint(checkpoint,filename=saved_file2)
    checkpoint={'state_dict':aligner.state_dict(),'optimizer':aligner_optimiser.state_dict()}
    save_checkpoint(checkpoint,filename=saved_file3)
    checkpoint={'state_dict':discriminator.state_dict(),'optimizer':discriminator_optimiser.state_dict()}
    save_checkpoint(checkpoint,filename=saved_file4)
    '''


    x1 = []
    x2 = []
    for x in range(epoch_num_Encoder):
        x += 1
        x1.append(x)

    for x in range(epoch_num_Aligner):
        x += 1
        x2.append(x)

    plt.figure(dpi=300)
    plt.plot(x1,ae_train_losses)
    #plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss of Encoder'],loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    #plt.savefig('Figures_M_1/T1/Encoder_loss.png')
    plt.savefig(pickle_file1)

    plt.figure(dpi=300)
    plt.plot(x1,cl_train_losses)
    #plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss of Classifier'],loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    #plt.savefig('Figures/Fig1/Classifier_loss.png')
    plt.savefig(pickle_file2)

    plt.figure(dpi=300)
    plt.plot(x2,aligner_train_losses)
    #plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss of Aligner'],loc='upper right')
    plt.xlim([1, epoch_num_Aligner])
    #plt.savefig('Figures/Fig1/Aligner_loss.png')
    plt.savefig(pickle_file3)

    plt.figure(dpi=300)
    plt.plot(x1,ae_train_accuracy)
    #plt.plot(Loss_Val)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy of Encoder'],loc='lower right')
    plt.xlim([1, epoch_num_Encoder])
    #plt.savefig('Figures/Fig1/Encoder_Accuracy.png')
    plt.savefig(pickle_file5)

    plt.figure(dpi=300)
    plt.plot(x2,aligner_train_accuracy)
    #plt.plot(Loss_Val)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy of Aligner'],loc='lower right')
    plt.xlim([1, epoch_num_Aligner])
    #plt.savefig('Figures/Fig1/Aligner_Accuracy.png')
    plt.savefig(pickle_file6)


    plt.figure( dpi=300)
    plot_confusion_matrix(cm1, classes)
    #plt.savefig('Figures/Fig1/CM_Aligned.png')
    plt.savefig(pickle_file7)

    plt.figure(dpi=300)
    plot_confusion_matrix(cm2, classes)
    #plt.savefig('Figures/Fig1/CM_Non_Aligned.png')
    plt.savefig(pickle_file8)



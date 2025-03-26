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
import Model as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_layers = 4
num_epoch = 200

BATCH_SIZE = batch_size = 420 


classes = ('Sit still on a chair', 'Falling down', 'Lie down', 'Stand still', 'Walking from Tx to Rx', 'Walking from Rx to Tx', 'Pick a pen from ground')

train_dataset_path="./Dataset3/E123/S22"
test_dataset_path="./Dataset3/E123/S12"

mean = [0.9430, 0.8760, 0.4506] 

std = [0.0852, 0.1630, 0.4542] 


train_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_Tr,shuffle=True, num_workers=4)

test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=train_transforms)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_T,shuffle=False, num_workers=4)


training_samples = iter(train_loader)
samples, labels = training_samples.next()
samples = samples.reshape([420,12288])
samples = pd.DataFrame(samples)
labels = pd.Series(labels)
df_train = pd.concat([labels,samples],axis=1)


test_samples = iter(test_loader)
samples, labels = test_samples.next()
samples = samples.reshape([420,12288])
samples = pd.DataFrame(samples)
labels = pd.Series(labels)
df_test = pd.concat([labels,samples],axis=1)


train_dataset = models.SamplePairGenerator(df_train,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size1, shuffle=True, num_workers=4)

test_dataset = models.SamplePairGenerator(df_test,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))

training_proportion =  0.7
rand_seed=42

test_training_proportion = int(training_proportion*len(test_dataset))
np.random.seed(rand_seed)
idxs = np.random.permutation(len(test_dataset))


target_train_sampler=SubsetRandomSampler(idxs[:test_training_proportion])
target_train_loader = DataLoader(test_dataset, batch_size=batch_size1, num_workers=4, sampler=target_train_sampler)

target_test_sampler=SubsetRandomSampler(idxs[test_training_proportion:])
target_test_loader = DataLoader(test_dataset, batch_size=batch_size1, num_workers=4, sampler=target_test_sampler)


AE_model = models.AE(input_size_enc,input_size_dec,hidden_size_enc,hidden_size_dec,num_layers).to(device)

classifier = models.Classifier().to(device)

aligner = models.Aligner().to(device)


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

        for i, data in enumerate(train_loader, 0):
            inputs_src, positive_img_src, negative_img_src, labels = data

            # I CHANGED HERE
            inputs_src = inputs_src.to(device)
            labels = labels.to(device)
            positive_img_src = positive_img_src.to(device)
            negative_img_src = negative_img_src.to(device)

            # zero the parameter gradients
            AE_optimiser.zero_grad()
            classifier_optimiser.zero_grad()

            # forward + backward + optimize

            # calc auto encoder loss
            # inputs = image_src.copy()
            outputs = AE_model(inputs_src)
            ae_loss = AE_criterion(inputs_src, outputs)
            ae_tr_loss.append(ae_loss)

            # calc classifier loss
            cl_outputs, _ = classifier(AE_model.latent.clone())
            cl_loss = classifier_criterion(cl_outputs, labels)
            cl_tr_loss.append(cl_loss)


            positive_outputs = AE_model(positive_img_src)
            positive_outputs, _ = classifier(AE_model.latent.clone())

            # calc classifier loss
            negative_outputs = AE_model(negative_img_src)
            negative_outputs, _ = classifier(AE_model.latent.clone())

            # Triplet loss for Source Images
            loss_triplet_source = models.triplet_criterion(cl_outputs, positive_outputs, negative_outputs)
            total_loss =  ae_loss + cl_loss + (0.1 * loss_triplet_source)
            total_tr_losses.append(total_loss)
            total_loss.backward()

            AE_optimiser.step()
            classifier_optimiser.step()

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


time1 = time.time()-training_start_time
time1 = time1/60
print('Encoder_Training_Time:{:.2f}'.format(time1))
print('Finished Encoder Training')

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


    for i, data in enumerate(target_train_loader, 0):
        inputs_tgt, positive_img_tgt, negative_img_tgt, labels = data

        # I CHANGED HERE
        inputs_tgt = inputs_src.to(device)
        labels = labels.to(device)
        positive_img_tgt = positive_img_tgt.to(device)
        negative_img_tgt = negative_img_tgt.to(device)

        # Train aligner
        aligner_optimiser.zero_grad()

        aligned = aligner(inputs_tgt)
        aligned_positive = aligner(positive_img_tgt)
        aligned_negative = aligner(negative_img_tgt)


        # classified
        ae_out = AE_model(aligned)
        cl_outputs, t1 = classifier(AE_model.latent.clone())
        cl_loss = classifier_criterion(cl_outputs, labels)
        tr_classifier_losses.append(cl_loss.item())
        _, predicted = cl_outputs.max(1)
        correct += (predicted == labels).sum().item()
        size += labels.size(0)

        ae_out_positive = AE_model(aligned_positive)
        positive_outputs, _ = classifier(AE_model.latent.clone())

        ae_out_negative = AE_model(aligned_negative)
        negative_outputs, _ = classifier(AE_model.latent.clone())

        # Triplet loss for Target Images
        loss_triplet_target = models.triplet_criterion(cl_outputs , positive_out, negative_out)

        # Discrepancy Loss
        loss_discrepancy = models.discrepancy(cl_outputs, positive_out)

        # loss
        loss_align = cl_loss + (0.1 * loss_discrepancy) + (0.1 * loss_triplet_target)

        tr_aligner_losses.append(loss_align.item())
        loss_align.backward(retain_graph=True)
        aligner_optimiser.step()

    accuracy = float(correct/ size)
    avg_aligner_loss = float(sum(tr_aligner_losses) / len(tr_aligner_losses))
    avg_cl_loss = float(sum(tr_classifier_losses) / len(tr_classifier_losses))

    aligner_train_losses.append(avg_aligner_loss)
    aligner_train_accuracy.append(accuracy * 100)
    print(f'Epoch:{epoch + 1}, avg_aligner_loss:{avg_aligner_loss:.4f}, Training Accuracy:{accuracy:.4f}')

time2 = time.time()-training_start_time
time2 = time2/60
time3 = time1 + time2
print('Aligner_Training_Time:{:.2f}'.format(time2))
print('Finished Aligner Training')
print('Aligner Training Stops at epoch:', epoch_num_Aligner)
print('Training_Time:{:.2f}'.format(time3))
#################### END of Validation

with torch.no_grad():
    correct = 0
    size = 0
    raw_correct = 0
    raw_size = 0
    all_preds = torch.tensor([]).to(device)
    all_preds_raw = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    accuracyT = []
for i, data_target in enumerate(target_test_loader, 0):
    inputs_tgt, _, _, labels_tgt = data_target

    inputs_tgt = inputs_t
    labels_tgt = labels_t

    # I CHANGED HERE
    inputs_tgt = inputs_tgt.to(device)
    labels_tgt = labels_tgt.to(device)

    all_labels = torch.cat((all_labels, labels_tgt), dim=0)  # For Plotting Purpose in CMT & Hist

    AE_model.eval()
    classifier.eval()
    aligner.eval()

    # classified aligned images
    aligned = aligner(inputs_tgt)
    outputs = AE_model(aligned_tgt)
    cl_outputs, _ = classifier(AE_model.latent.clone())
    _, predicted = cl_outputs.max(1)
    all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT
    correct = (predicted == labels).sum().item()
    size = labels.size(0)

    # classified raw(non-aligned) images
    raw_outputs = AE_model(inputs_tgt)
    raw_cl_outputs, _ = classifier(AE_model.latent.clone())
    _, raw_predicted = raw_cl_outputs.max(1)
    all_preds_raw = torch.cat((all_preds_raw, raw_predicted), dim=0)  # For Plotting Purpose in CMT
    raw_correct = (raw_predicted == labels).sum().item()
    raw_size = labels_t.size(0)


accuracy = float(correct / size)
#print(accuracy)

raw_accuracy = float(raw_correct / raw_size)
#print(raw_accuracy)

print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=classes, zero_division=0))
print(classification_report(all_labels.cpu().numpy(), all_preds_raw.cpu().numpy(), target_names=classes,zero_division=0))


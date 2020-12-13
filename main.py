import torch
import torchaudio
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
from model import WaveNet, MuLaw
from dataset import LJSpeechDataset
from featurizer_spec import MelSpectrogram, MelSpectrogramConfig
from training_func import train_model, evaluate


wandb.init(project="dla_5", entity="vladatrus")
mulaw = MuLaw()
df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
train, test = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = LJSpeechDataset(train)
test_dataset = LJSpeechDataset(test)

train_dataloader = DataLoader(train_dataset,
                              batch_size=6,
                              num_workers=8,
                              shuffle=False,
                              pin_memory=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=6,
                             num_workers=8,
                             pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WaveNet()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
model.to(device)
wandb.watch(model, log="all")

N_EPOCHS = 14

for epoch in tqdm(range(N_EPOCHS)):
    train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)

    wandb.log({"learning_rate" : 0.0001,
               "model" : 'wavenet',
               "optimizer" : 'Adam',
               "train_loss": train_loss,
               "train_accuracy": train_acc,
               "test_loss": test_loss,
               "test_accuracy": test_acc,
               })
           



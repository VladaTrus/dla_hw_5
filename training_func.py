import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(model: nn.Module,
          iterator: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module):


    model.train()
    losses = []
    accs = []
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    #print('Train')
    i = 1
    for input in train_dataloader:
        epoch_accuracy = 0
        epoch_loss = 0
        #print(i)
        i += 1
        input = input.to(device)
        zeros = torch.zeros((input.shape[0], 1)).to(device)
        mels = featurizer(input)

        wavs_padded = torch.cat([zeros, input[:, :-1]], dim=1)
        output = model(wavs_padded, mels, mulaw)

        target = output.argmax(dim=1)
        encoded = mulaw.encode(input)
        quantized = mulaw.quantize(encoded)
        loss = criterion(output, quantized)

        epoch_accuracy = (target == quantized).sum().item() / target.shape[-1] / target.shape[0]
        accs.append(epoch_accuracy)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss = loss.item()
        losses.append(epoch_loss)

    return np.mean(losses), np.mean(accs)


def evaluate(model: nn.Module,
             iterator: DataLoader,
             criterion: nn.Module):
  

    model.eval()
    losses = []
    accs = []
    #print('Test')
    i = 1
    with torch.no_grad():
        featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
        for input in test_dataloader:
            epoch_accuracy = 0
            epoch_loss = 0
            #print(i)
            i += 1
            input = input.to(device)
            zeros = torch.zeros((input.shape[0], 1)).to(device)
            mels = featurizer(input)

            wavs_padded = torch.cat([zeros, input[:, :-1]], dim=1)
            output = model(wavs_padded, mels, mulaw)

            target = output.argmax(dim=1)
            encoded = mulaw.encode(input)
            quantized = mulaw.quantize(encoded)
            loss = criterion(output, quantized)

            epoch_accuracy = (target == quantized).sum().item() / target.shape[-1] / target.shape[0]
            accs.append(epoch_accuracy)
            epoch_loss = loss.item()
            losses.append(epoch_loss)

    return np.mean(losses), np.mean(accs)



import torch
import torchaudio
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, dilation, kernel_1d = 2):
        super().__init__()
        self.padding = (kernel_1d - 1) * dilation
        self.causal_conv = torch.nn.Conv1d(in_channels = 120, out_channels = 240,
                                      kernel_size = kernel_1d, stride=1,
                                      padding=self.padding,
                                      dilation=dilation)
        self.conv_1 = nn.Conv1d(80, 240, kernel_size=1)

        self.skip_1 = nn.Conv1d(240, 240, kernel_size=1)
        self.residual_1 = nn.Conv1d(240, 120, kernel_size=1)

    def forward(self, input, condition):
        output = self.causal_conv(input)
        output = output[:, :, :-self.padding]
        condition_out = self.conv_1(condition)
        output += condition_out
        tanh = torch.tanh(output)
        sigmoid = torch.sigmoid(output)
        output = tanh * sigmoid
        skip = self.skip_1(output)
        output = self.residual_1(output)
        output += input
        return output, skip

class WaveNet(nn.Module):
    def __init__(self, num_mels=80, dilations_d=8, repeat=2, kernel=2):
        super().__init__()

        self.conv_input = nn.Conv1d(1, 120, kernel_size=1)
        self.upsample = nn.ConvTranspose1d(num_mels, num_mels, kernel_size=800, stride=250, padding=150)

        self.dilations = [2 ** pow for pow in range(dilations_d)] * repeat
        blocks = []
        for dilation in self.dilations:
            blocks.append(ResidualBlock(dilation))
        self.blocks = nn.ModuleList(blocks)

        self.r_field = (kernel - 1) * sum(self.dilations) + 1

        self.ending = nn.Sequential(nn.ReLU(),
                                         nn.Conv1d(240, 240, kernel_size=1),
                                         nn.ReLU(),
                                         nn.Conv1d(240, 256, kernel_size=1))

    def forward(self, input, melspec, mulaw, need_upsample = True):

        output = self.conv_input(mulaw.encode(input).unsqueeze(1))
        if need_upsample:
            melspec = self.upsample(melspec)
        skips = []
        for block in self.blocks:
            output, skip = block(output, melspec)
            skips.append(skip)
      
        output = self.ending(sum(skips))
        return output

    def inference(self, melspec, mulaw):
        wavs = torch.zeros((melspec.shape[0], self.r_field), dtype=torch.float).to(melspec.device)
        melspec = self.upsample(melspec)

        for i in range(melspec.shape[2]):
            w = wavs[:, -self.r_field:]
            m = melspec[:, :, i:i + w.shape[-1]]
            if w.shape[-1] != m.shape[-1]:
                break
            output = self.forward(w, m, mulaw, need_upsample = False)
            quantized = torch.argmax(output[:, :, -1].detach(), dim=1).unsqueeze(-1)
            wavs = torch.cat([wavs, mulaw.decode(mulaw.dequantize(quantized))], dim=1)
        
        wavs = wavs[:, -melspec.shape[2]:]
        return wavs
        
        
class MuLaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = 255
    def encode(self, input):
        output = torch.sign(input) * (torch.log(1. + self.mu * torch.abs(input)) / np.log(1. + self.mu))
        return output
    def decode(self, input):
        output = torch.sign(input) / self.mu * (torch.pow(1. + self.mu, torch.abs(input)) - 1.)
        return output
    def quantize(self, input):
        output = ((1. + input) * (self.mu / 2) + 0.1).to(torch.long)
        return output
    def dequantize(self, input):
        output = (input - 0.1) * (2 / self.mu) - 1.
        return output

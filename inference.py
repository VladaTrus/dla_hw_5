import torch
import torchaudio
import numpy as np
from model import WaveNet, MuLaw
from featurizer_spec import MelSpectrogram, MelSpectrogramConfig


mulaw = MuLaw()
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WaveNet()
model.load_state_dict(torch.load('wavenet_15.pth'))
model.to(device)
model.eval()

wav, _ = torchaudio.load("LJ033-0047.wav")
wav = wav.to(device)

model.eval()
with torch.no_grad():
    mels = featurizer(wav)
    prediction = model.inference(mels).squeeze()

result = mulaw.decode(prediction).cpu()
torchaudio.save('result.wav', result, 22050)
 

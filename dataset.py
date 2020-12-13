import os
import numpy as np
from torch.utils.data import Dataset
import torchaudio

class LJSpeechDataset(Dataset):
    def __init__(self, df):
        self.dir = "LJSpeech-1.1/wavs"
        self.paths = df.index.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.dir, f'{self.paths[index]}.wav')
        wav, sr = torchaudio.load(path)
        wav = wav.squeeze()
        if wav.shape[0] <= 20000:
            pad_wav = torch.full((20000, ), fill_value=0.0)
            pad_wav[:wav.shape[1]] = wav[:wav.shape[1]]
            output_wav = pad_wav
        else:
            rand_pos = random.randint(0, wav.shape[0] - 20000)
            output_wav = wav[rand_pos:rand_pos + 20000]
        return output_wav

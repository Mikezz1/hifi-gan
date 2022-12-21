import torch
import torchaudio
import numpy as np
import yaml

from hifi_gan.utils.preprocessing import MelSpectrogram


with open("configs/base_config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
melspec_transform = MelSpectrogram(config).to(device)

wavs = [torchaudio.load(f"data/audio_{i}.wav", normalize=True) for i in range(1, 4)]
wavs = [0.95 * i[0] for i in wavs]
melspecs = [melspec_transform(wav) for wav in wavs]
for i, melspec in enumerate(melspecs):
    np.save(f"data/test_spec_{i}", melspec.numpy())

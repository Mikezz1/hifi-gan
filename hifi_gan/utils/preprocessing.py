from dataclasses import dataclass
import torch
from torch import nn
import torchaudio
import librosa


# @dataclass
# class MelSpectrogramConfig:
#     sr: int = 22050
#     win_length: int = 1024
#     hop_length: int = 256
#     n_fft: int = 1024
#     f_min: int = 0
#     f_max: int = 8000
#     n_mels: int = 80
#     power: float = 1.0

#     # value of melspectrograms if we fed a silence into `MelSpectrogram`
#     pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(self, config):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["preprocessing"]["sr"],
            win_length=config["preprocessing"]["win_length"],
            hop_length=config["preprocessing"]["hop_length"],
            n_fft=config["preprocessing"]["n_fft"],
            f_min=config["preprocessing"]["f_min"],
            f_max=config["preprocessing"]["f_max"],
            n_mels=config["preprocessing"]["n_mels"],
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config["preprocessing"]["power"]

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config["preprocessing"]["sr"],
            n_fft=config["preprocessing"]["n_fft"],
            n_mels=config["preprocessing"]["n_mels"],
            fmin=config["preprocessing"]["f_min"],
            fmax=config["preprocessing"]["f_max"],
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        mel = self.mel_spectrogram(audio[..., :-1]).clamp_(min=1e-5).log_()

        # print(
        #     f'len / hop_len: {audio.size(1) /self.config["preprocessing"]["hop_length"]}'
        # )
        # print(f"mel: {mel.size()}")

        return mel

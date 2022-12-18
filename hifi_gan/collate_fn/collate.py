import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    # Should return (batch_size, n_mels, t)
    spectrogram = pad_sequence(
        [item["mels"].squeeze().permute(1, 0) for item in dataset_items],
        padding_value=0,
    )
    spectrogram = spectrogram.permute(1, 2, 0)

    spectrogram_length = torch.LongTensor(
        [item["mels"].size()[2] for item in dataset_items]
    )

    # print(dataset_items[0]["wav"].size())
    audio = pad_sequence(
        [item["wav"].squeeze() for item in dataset_items], padding_value=0
    )
    audio = audio.permute(1, 0).unsqueeze(1)

    audio_path = [item["audio_path"] for item in dataset_items]

    result_batch = {
        "wavs": audio,
        "mels": spectrogram,
        "spectrogram_length": spectrogram_length,
        "audio_path": audio_path,
    }
    return result_batch

import torch
import torchaudio
import numpy as np
from hifi_gan.model.generator import Generator
import yaml


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/local_config.yaml") as f:
        config = yaml.safe_load(f)

    generator = Generator(
        config["model"]["upsample_kernel_sizes"],
        config["model"]["upsample_first"],
        config["model"]["kernels"],
        config["model"]["dilations"],
    )
    generator.load_state_dict(
        torch.load(config["training"]["checkpoint"], map_location=device)["generator"]
    )
    generator.eval()

    test_mels = [np.load(f"data/test_spec_{i}.npy") for i in range(3)]

    with torch.no_grad():
        wavs = [generator(torch.Tensor(mel)) for mel in test_mels]

        for i, wav in enumerate(wavs):
            torchaudio.save(f"data/predicted_audio_{i}.wav", wav.squeeze(1), 22050)


if __name__ == "__main__":
    main()

# init objects - datasets, dataloaders etc
import argparse
import yaml
import torch
from hifi_gan.collate_fn.collate import collate_fn
from hifi_gan.dataset.lj_speech import LJspeechDataset
from hifi_gan.logger.wandb_writer import WanDBWriter
from hifi_gan.trainer.trainer import Trainer
from hifi_gan.model.discriminators import MSD, MPD
from hifi_gan.model.generator import Generator
from hifi_gan.model.loss import GeneratorLoss, DiscriminatorLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from hifi_gan.utils.preprocessing import MelSpectrogram


# train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/base_config.yaml",
        type=str,
        help="config file name (without .py)",
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to checkpoint (or empty string if  train from scratch)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    melspec_transform = MelSpectrogram(config).to(device)

    dataset = LJspeechDataset(
        config_parser=config, part="train", data_dir=config["base"]["data_dir"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["bs"],
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        drop_last=config["dataset"]["drop_last"],
    )

    generator = Generator(
        config["model"]["upsample_kernel_sizes"],
        config["model"]["upsample_first"],
        config["model"]["kernels"],
        config["model"]["dilations"],
    )
    MSD = MSD()
    MPD = MPD(config["model"]["periods"])  # [2, 3, 5, 7, 11]

    generator = generator.to(device)
    MSD = MSD.to(device)
    MPD = MPD.to(device)

    # print(sum(p.numel() for p in generator.parameters()))
    # print(sum(p.numel() for p in MSD.parameters()))
    # print(sum(p.numel() for p in MPD.parameters()))

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config["training"]["g_learning_rate"],
        betas=(0.8, 0.99),
        weight_decay=config["training"]["weight_decay"],
        eps=1e-9,
    )

    optimizer_d = torch.optim.AdamW(
        list(MSD.parameters()) + list(MPD.parameters()),
        lr=config["training"]["d_learning_rate"],
        betas=(0.8, 0.99),
        weight_decay=config["training"]["weight_decay"],
        eps=1e-9,
    )

    # scheduler_g = OneCycleLR(optimizer_g, **{
    #     "steps_per_epoch": len(dataloader),
    #     "epochs": config['training']['epochs'],
    #     "anneal_strategy": "cos",
    #     "max_lr": config['training']['g_learning_rate'],
    #     "pct_start": 0.1
    # })

    # scheduler_d = OneCycleLR(optimizer_d, **{
    #     "steps_per_epoch": len(dataloader),
    #     "epochs": config['training']['epochs'],
    #     "anneal_strategy": "cos",
    #     "max_lr": config['training']['d_learning_rate'],
    #     "pct_start": 0.1
    # })

    logger = WanDBWriter(config)

    generator_loss = GeneratorLoss(
        config["loss"]["lambda_mel"], config["loss"]["lambda_fmap"]
    )

    discriminator_loss = DiscriminatorLoss()

    trainer = Trainer(
        config,
        dataloader,
        MPD,
        MSD,
        generator,
        optimizer_d,
        optimizer_g,
        discriminator_loss,
        generator_loss,
        melspec_transform,
        logger,
    )

    trainer.train()

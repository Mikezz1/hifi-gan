import torch
import torch.nn as nn
from typing import List
import torch.nn
from torch.nn.functional import l1_loss


class GeneratorLoss:
    def __init__(self, lambda_mel, lambda_fmap):
        self.lambda_mel = lambda_mel
        self.lambda_fmap = lambda_fmap

    def __call__(
        self,
        msd_out_fake: List[List[torch.Tensor]],
        mpd_out_fake: List[torch.Tensor],
        mpd_fmap_fake: List[torch.Tensor],
        mpd_fmap_real: List[torch.Tensor],
        msd_fmap_fake: List[torch.Tensor],
        msd_fmap_real: List[torch.Tensor],
        mels_true: torch.Tensor,
        mels_fake: torch.Tensor,
    ):

        loss = torch.Tensor([0])
        mpd_loss = torch.Tensor([0])
        msd_loss = torch.Tensor([0])
        feature_loss_mpd = torch.Tensor([0])
        feature_loss_msd = torch.Tensor([0])

        # mpd
        for output in mpd_out_fake:
            mpd_loss += torch.mean((output - 1) ** 2)

        for fmap_fake, fmap_real in zip(mpd_fmap_fake, mpd_fmap_real):
            l = torch.mean(
                torch.Tensor(
                    [
                        l1_loss(fmap_f, fmap_r)
                        for fmap_f, fmap_r in zip(fmap_fake, fmap_real)
                    ]
                )
            )
            feature_loss_mpd += l
        feature_loss_mpd = feature_loss_mpd / len(mpd_fmap_fake)

        # msd
        for output in msd_out_fake:
            msd_loss += torch.mean((output - 1) ** 2)

        for fmap_fake, fmap_real in zip(msd_fmap_fake, msd_fmap_real):
            l = torch.mean(
                torch.Tensor(
                    [
                        l1_loss(fmap_f, fmap_r)
                        for fmap_f, fmap_r in zip(fmap_fake, fmap_real)
                    ]
                )
            )
            feature_loss_msd += l

        feature_loss_msd = feature_loss_msd / len(msd_fmap_fake)

        # print(feature_loss_msd)
        # print(feature_loss_mpd)

        # mel loss
        mel_loss = l1_loss(mels_fake, mels_true)

        # total loss
        loss = (
            self.lambda_mel * mel_loss
            + self.lambda_fmap * feature_loss_mpd
            + self.lambda_fmap * feature_loss_msd
            + msd_loss
            + mpd_loss
        )
        # print(loss, mel_loss, feature_loss_mpd, feature_loss_msd, msd_loss, mpd_loss)

        return (
            loss,
            mel_loss.detach().cpu(),
            feature_loss_mpd.detach().cpu(),
            feature_loss_msd.detach().cpu(),
            msd_loss.detach().cpu(),
            mpd_loss.detach().cpu(),
        )


class DiscriminatorLoss:
    def __init__(self):
        pass

    def __call__(
        self,
        msd_out_fake: List[List[torch.Tensor]],
        mpd_out_fake: List[torch.Tensor],
        msd_out_real: List[List[torch.Tensor]],
        mpd_out_real: List[torch.Tensor],
    ):

        msd_loss = torch.Tensor([0])
        mpd_loss = torch.Tensor([0])

        for output_fake, output_real in zip(mpd_out_fake, mpd_out_real):
            # print(output_fake.size())
            # print(output_real.size())
            mpd_loss = mpd_loss + torch.mean((output_real - 1) ** 2 + output_fake**2)

        for output_fake, output_real in zip(msd_out_fake, msd_out_real):
            output_real = output_real.flatten()
            output_fake = output_fake.flatten()

            l = torch.mean((output_real - 1) ** 2 + output_fake**2)
            # print(l.size())
            # print(msd_loss.size())
            msd_loss = msd_loss + l

        return (msd_loss + mpd_loss) / 2

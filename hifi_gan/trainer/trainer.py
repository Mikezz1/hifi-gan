import torch
import torch.nn as nn
import PIL
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
import numpy as np
from hifi_gan.logger.utils import plot_spectrogram_to_buf
from hifi_gan.utils.utils import save_checkpoint


class Trainer:
    def __init__(
        self,
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
        scheduler_d=None,
        scheduler_g=None,
        device="cpu",
    ):
        self.config = config
        self.dataloader = dataloader
        self.MPD = MPD
        self.MSD = MSD
        self.generator = generator
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss
        self.melspec_transform = melspec_transform
        self.logger = logger
        self.scheduler_d = scheduler_d
        self.scheduler_g = scheduler_g
        self.device = device

    def train(
        self,
    ):

        self.MSD.train()
        self.MPD.train()
        self.generator.train()

        step = 0

        for epoch in range(self.config["training"]["epochs"]):
            for batch in tqdm(self.dataloader):
                step += 1
                self.logger.set_step(step)

                mels = batch["mels"].to(self.device)
                wavs = batch["wavs"].to(self.device)

                self.optimizer_d.zero_grad()
                fake_wav = self.generator(mels).detach()

                assert fake_wav.size(2) == wavs.size(2)

                # MSD
                msd_out_real, _ = self.MSD(wavs)  # List(Tensor)
                msd_out_fake, _ = self.MSD(fake_wav)  # List(Tensor)

                # MPD
                mpd_out_real, _ = self.MPD(wavs)  # List(Tensor), List(Tensor)
                mpd_out_fake, _ = self.MPD(fake_wav)  # List(Tensor), List(Tensor)

                d_total_loss = self.discriminator_loss(
                    msd_out_fake,
                    mpd_out_fake,
                    msd_out_real,
                    mpd_out_real,
                )

                d_total_loss.backward()

                self.optimizer_d.step()

                nn.utils.clip_grad_norm_(
                    self.MPD.parameters(), self.config["training"]["grad_clip"]
                )

                nn.utils.clip_grad_norm_(
                    self.MSD.parameters(), self.config["training"]["grad_clip"]
                )

                self.optimizer_g.zero_grad()

                fake_wav = self.generator(mels)

                msd_out_fake, msd_fmap_fake = self.MSD(fake_wav)
                msd_out_real, msd_fmap_real = self.MSD(wavs)

                mpd_out_fake, mpd_fmap_fake = self.MPD(fake_wav)
                mpd_out_real, mpd_fmap_real = self.MPD(wavs)

                mels_fake = self.melspec_transform(fake_wav)

                (
                    g_total_loss,
                    mel_loss,
                    feature_loss_mpd,
                    feature_loss_msd,
                    msd_loss,
                    mpd_loss,
                ) = self.generator_loss(
                    msd_out_fake,
                    mpd_out_fake,
                    mpd_fmap_fake,
                    mpd_fmap_real,
                    msd_fmap_fake,
                    msd_fmap_real,
                    mels,
                    mels_fake,
                )

                g_total_loss.backward()
                self.optimizer_g.step()
                nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self.config["training"]["grad_clip"]
                )

                if step % self.config["training"]["log_steps"] == 0:
                    grad_norm_g = self.get_grad_norm(self.generator)
                    grad_norm_mpd = self.get_grad_norm(self.MPD)
                    grad_norm_msd = self.get_grad_norm(self.MSD)

                    self.log_everything(
                        step,
                        epoch,
                        g_total_loss,
                        d_total_loss,
                        mel_loss,
                        feature_loss_mpd,
                        feature_loss_msd,
                        msd_loss,
                        mpd_loss,
                        mels[0].squeeze(0),
                        mels_fake[0].squeeze(0),
                        wavs[0],
                        fake_wav[0],
                        grad_norm_g,
                        grad_norm_mpd,
                        grad_norm_msd,
                        self.scheduler_g.get_last_lr(),
                    )

                    self.inference()

                if step % self.config["training"]["save_steps"] == 0:
                    save_checkpoint(
                        self.generator,
                        self.MPD,
                        self.MSD,
                        self.optimizer_g,
                        self.optimizer_d,
                        self.config["base"]["checkpoint_path"],
                        self.config["base"]["model_name"],
                        step,
                    )

            self.scheduler_g.step()
            self.scheduler_d.step()

            # return generator, msd, mpd

    def log_everything(
        self,
        step,
        epoch,
        generator_loss,
        discriminator_loss,
        mel_loss,
        feature_loss_mpd,
        feature_loss_msd,
        msd_loss,
        mpd_loss,
        melspec_gt,
        melspec_pred,
        audio_true,
        audio_pred,
        grad_norm_g,
        grad_norm_mpd,
        grad_norm_msd,
        learning_rate,
    ):
        self.logger.add_scalar("step", step)
        self.logger.add_scalar("epoch", epoch)
        self.logger.add_scalar("generator_loss", generator_loss.detach().item())
        self.logger.add_scalar("discriminator_loss", discriminator_loss.detach().item())
        self.logger.add_scalar("mel_loss", mel_loss.detach().item())
        self.logger.add_scalar("feature_loss_mpd", feature_loss_mpd.detach().item())
        self.logger.add_scalar("feature_loss_msd", feature_loss_msd.detach().item())
        self.logger.add_scalar("msd_loss", msd_loss.detach().item())
        self.logger.add_scalar("mpd_loss", mpd_loss.detach().item())
        self.logger.add_scalar("learning_rate_g", learning_rate)
        self.logger.add_scalar("grad_norm_g", grad_norm_g)
        self.logger.add_scalar("grad_norm_msd", grad_norm_msd)
        self.logger.add_scalar("grad_norm_mpd", grad_norm_mpd)
        self._log_spectrogram(melspec_gt, caption="melspec_gt")
        self._log_spectrogram(melspec_pred, caption="melspec_pred")
        self._log_audio(audio_true, caption="audio_gt")
        self._log_audio(audio_pred, caption="audio_pred")

    def inference(self):

        test_mels = [np.load(f"data/test_spec_{i}.npy") for i in range(3)]

        self.generator.eval()
        with torch.no_grad():
            wavs = [
                self.generator(torch.Tensor(mel).to(self.device)) for mel in test_mels
            ]

            for i, wav in enumerate(wavs):
                self._log_audio(wav.squeeze(1), caption=f"test_audio_{i}")

        self.generator.train()

    def _log_spectrogram(self, spectrogram, caption="spectrogram_t"):
        """
        Move to utils
        """
        spectrogram = spectrogram.detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.logger.add_image(caption, ToTensor()(image))

    def _log_audio(self, audio, caption="audio_t"):
        """
        Move to utils
        """
        self.logger.add_audio(caption, audio.detach().cpu(), sample_rate=22_050)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        """
        Move to utils
        """
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

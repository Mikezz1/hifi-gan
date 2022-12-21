from pathlib import Path
import torch
import os


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def save_checkpoint(
    generator, MPD, MSD, optimizer_g, optimizer_d, path, model_name, step
):
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "MPD": MPD.state_dict(),
            "MSD": MSD.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        },
        os.path.join(path, f"checkpoint_{model_name}_{step}.pth.tar"),
    )
    print("save model at step %d ..." % step)

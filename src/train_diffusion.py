"""Train an unconditional DDPM on OrganAMNIST.

The model is a MONAI DiffusionModelUNet predicting the noise added at each
timestep. Mixed precision is on by default because the run is otherwise about
twice as long for no measurable gain in sample quality at this resolution.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

from data import loader

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


def amp_setting(dev):
    """Pick an autocast dtype the GPU actually supports.

    Ampere and later do bfloat16. Volta, which is what the V100s are, does not,
    and asking for it there either errors or silently runs in a slow path. Those
    cards do float16 well, which needs a gradient scaler because the range is
    narrow enough for gradients to underflow.
    """
    if dev != "cuda":
        return None, False
    major, _ = torch.cuda.get_device_capability()
    # torch.cuda.is_bf16_supported() answers True on Volta because bfloat16 is
    # emulated there, and the emulated path is slower than float16. Compute
    # capability 8.0 is where the hardware support starts, so ask for that.
    if major >= 8:
        return torch.bfloat16, False
    return torch.float16, True


def build_unet():
    """UNet used for every experiment here. Attention at the coarsest level only."""
    return DiffusionModelUNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(64, 128, 192), attention_levels=(False, False, True),
        num_res_blocks=2, num_head_channels=64,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--out", default="ddpm.pt")
    args = ap.parse_args()

    WEIGHTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = loader("train", args.batch_size)
    model = build_unet().to(dev)
    sched = DDPMScheduler(num_train_timesteps=args.timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_dtype, needs_scaler = amp_setting(dev)
    scaler = torch.amp.GradScaler(dev, enabled=needs_scaler)
    print(f"device {dev}, autocast {amp_dtype}, grad scaler {needs_scaler}", flush=True)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, _ in train:
            x = x.to(dev, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=dev)
            noise = torch.randn_like(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(sched.add_noise(x, noise, t), t)
                loss = F.mse_loss(pred.float(), noise)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        epoch_loss = running / seen
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {epoch_loss:.4f}"
              f"  elapsed {(time.time() - start) / 60:.1f} min", flush=True)

    torch.save({"state_dict": model.state_dict(), "timesteps": args.timesteps},
               WEIGHTS / args.out)
    (RESULTS / "diffusion_training.json").write_text(json.dumps(
        {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
         "timesteps": args.timesteps, "amp_dtype": str(amp_dtype),
         "minutes": (time.time() - start) / 60,
         "history": history}, indent=2))
    print(f"saved {WEIGHTS / args.out}")


if __name__ == "__main__":
    main()

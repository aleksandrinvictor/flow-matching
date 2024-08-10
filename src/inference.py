import argparse
from math import isqrt

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchdiffeq import odeint
from torchvision.utils import save_image

from src.models.unet import Unet


@torch.inference_mode()
def infer(
    model: nn.Module,
    num_samples: int = 16,
    num_steps: int = 100,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    method: str = "dopri5",
):
    """Generates images from model using numerical integration.

    Args:
        model: Model that predicts conditional vector field.
        num_samples: How many samples to generate. Defaults to 16.
        num_steps: Num steps for numerical integration. Defaults to 100.
        atol: Upper bound on absolute error. Defaults to 1e-4.
        rtol: Upper bound on relative error. Defaults to 1e-4.
        method: Numerical integration method. Defaults to "dopri5".

    Returns:
        Predicted images on every timestamp, [num_steps, num_samples, 1, h, w]
    """
    model.eval()
    device = next(model.parameters()).device

    base_distribution = Normal(0, 1)
    x_0 = base_distribution.sample(sample_shape=(num_samples, 1, 28, 28)).to(device)

    timesteps = torch.linspace(0.0, 1.0, num_steps).to(device)

    output = odeint(
        func=lambda t, x: model(x, t.repeat(num_samples)),
        y0=x_0,
        t=timesteps,
        atol=atol,
        rtol=rtol,
        method=method,
    )

    return output


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_filepath", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_filepath", type=str, default="result.png")
if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Unet(channels=1, dim_mults=(1, 2, 4), dim=28, resnet_block_groups=1)

    checkpoint = torch.load(args.checkpoint_filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    result = infer(
        model=model,
        num_samples=args.num_samples,
        num_steps=100,
        atol=1e-5,
        rtol=1e-5,
        method="dopri5",
    )  # [num_steps, num_samples, 1, h, w]

    n_rows = isqrt(args.num_samples)
    if n_rows**2 != args.num_samples:
        n_rows += 1
    save_image(result[-1, ...], args.output_filepath, nrow=n_rows)

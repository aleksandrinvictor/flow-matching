import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets.mnist import get_mnist_dataloader
from src.flows import ConditionalFlow, OTConditionalFlow
from src.inference import infer
from src.models.unet import Unet


def loss_fn(
    model: nn.Module,
    target_flow: ConditionalFlow,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Counts MSE loss between predicted and target conditional vector fields.

    Check eq. (9) in paper: https://arxiv.org/abs/2210.02747

    Args:
        model: Model that predicts conditional vector field.
        target_flow: Object that models target conditional vector field.
        x_0: Samples from base distribution, [batch_size, 1, h, w].
        x_1: Samples from target distribution, [batch_size, 1, h, w].
        t: Time samples, [batch_size].

    Returns:
        MSE loss between predicted and target conditional vector fields.
    """
    x_t = target_flow.sample_p_t(x_0=x_0, x_1=x_1, t=t)
    predicted_cond_vector_field = model(x_t, t)

    target_cond_vector_field = target_flow.get_conditional_vector_field(x_0=x_0, x_1=x_1, t=t)

    return F.mse_loss(predicted_cond_vector_field, target_cond_vector_field)


def train(
    model: nn.Module,
    target_flow: ConditionalFlow,
    dataloader: DataLoader,
    optimizer,
    device,
    num_epochs: int,
    save_path: Path,
):
    """Trains conditional vector field model.

    Args:
        model: Model that predicts conditional vector field.
        target_flow: Object that models target conditional vector field.
        dataloader: Dataloader.
        optimizer: Optimizer.
        device: Target device.
        num_epochs: Num epochs to train.
        save_path: Where to save checkpoints and intermediate results.
    """
    time_distribution = Uniform(0, 1)
    base_distribution = Normal(0, 1)

    for epoch in range(num_epochs):
        model.train()

        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                optimizer.zero_grad()

                x_1 = batch["image"]
                batch_size = x_1.shape[0]
                x_1 = x_1.to(device)
                x_0 = base_distribution.sample(sample_shape=x_1.shape).to(device)

                t = time_distribution.sample(sample_shape=(batch_size,)).to(device)

                loss = loss_fn(model=model, target_flow=target_flow, x_0=x_0, x_1=x_1, t=t)

                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

                loss.backward()
                optimizer.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, save_path / f"{epoch}-checkpoint.pth")

        epoch_output = infer(
            model=model, num_samples=16, num_steps=100, atol=1e-5, rtol=1e-5, method="dopri5"
        )  # [num_steps, num_samples, 1, h, w]
        save_image(epoch_output[-1, ...], save_path / f"samples_epoch-{epoch}.png", nrow=4)


parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=28)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_path", type=str, default="results")
if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device(args.device)
    dataloader = get_mnist_dataloader(batch_size=args.batch_size)

    # Setup model and optimizer
    model = Unet(
        channels=1,
        dim_mults=(1, 2, 4),
        dim=args.image_size,
        resnet_block_groups=1,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Setup conditional flow
    target_flow = OTConditionalFlow(sigma_min=0)

    os.makedirs(args.output_path, exist_ok=True)

    train(
        model=model,
        target_flow=target_flow,
        dataloader=dataloader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        save_path=Path(args.output_path),
    )

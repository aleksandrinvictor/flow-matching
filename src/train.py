import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import os
from src.flows import ConditionalFlow, OTConditionalFlow
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from src.datasets.mnist import FlowMatchingMNIST
from torchvision.transforms import CenterCrop, Compose, Lambda, Resize, ToTensor
from src.models.unet import Unet
from src.models.simple_model import SimpleModel
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


def loss_fn(
    model: nn.Module,
    target_flow: ConditionalFlow,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
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
    time_distribution = Uniform(0, 1)
    base_distribution = Normal(0, 1)

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            x_1 = batch["image"]
            batch_size = x_1.shape[0]
            x_1 = x_1.to(device)
            x_0 = base_distribution.sample(sample_shape=x_1.shape)

            t = time_distribution.sample(sample_shape=(batch_size,))

            loss = loss_fn(model=model, target_flow=target_flow, x_0=x_0, x_1=x_1, t=t)

            if step % 100 == 0:
                print(f"epoch: {epoch}, loss: {loss.item()}")

                # checkpoint = {
                #     "epoch": epoch,
                #     "model_state_dict": model.state_dict(),
                #     "optimizer_state_dict": optimizer.state_dict(),
                #     "loss": loss,
                # }

                # torch.save(checkpoint, save_path / f"{epoch}-checkpoint.pth")

            loss.backward()
            optimizer.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        torch.save(checkpoint, save_path / f"{epoch}-checkpoint.pth")


parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=28)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=6)
# parser.add_argument("--diffusion_timesteps", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--device", type=str, default="cpu")
if __name__ == "__main__":
    args = parser.parse_args()

    # Setup dataloader
    transform = Compose(
        [
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    dataset = FlowMatchingMNIST("./data", train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Setup model and optimizer
    model = Unet(channels=1, dim_mults=(1, 2, 4), dim=args.image_size)
    # model = SimpleModel()
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Setup conditional flow
    target_flow = OTConditionalFlow(sigma_min=0.1)

    save_path = "results"
    os.makedirs(save_path, exist_ok=True)

    train(
        model=model,
        target_flow=target_flow,
        dataloader=dataloader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        save_path=Path(save_path),
    )

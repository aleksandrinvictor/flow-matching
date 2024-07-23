import torch
from einops import rearrange


class ConditionalFlow:
    def sample_p_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_conditional_vector_field(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OTConditionalFlow(ConditionalFlow):
    def __init__(self, sigma_min: float) -> None:
        self.sigma_min = sigma_min

    def sample_p_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = rearrange(t, "b -> b 1 1 1")
        mu_t = t * x_1
        sigma_t = 1 - (1 - self.sigma_min) * t

        x_t = mu_t + sigma_t * x_0

        return x_t

    def get_conditional_vector_field(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_1 - (1 - self.sigma_min) * x_0

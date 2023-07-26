import torch
import numpy as np
from utils import denoise_add_noise, denoise_ddim


# sample using standard DDPM algorithm
@torch.no_grad()
def sample_ddpm(model, n_sample, context, timesteps, height, device, a_t, b_t, ab_t, save_rate=20, noise=True):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        # sample some random noise to inject back in. For i = 1, don't add back in noise
        if noise:
            z = torch.randn_like(samples) if i > 1 else 0
        else:
            z = 0

        if context == None:
            eps = model(samples, t)    # predict noise e_(x_t,t)
        else:
            eps = model(samples, t, c = context)
        samples = denoise_add_noise(samples, i, eps, a_t, b_t, ab_t, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


# sample quickly using DDIM
@torch.no_grad()
def sample_ddim(model, n_sample, timesteps, height, device, ab_t, n=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)
    intermediate = []
    step_size = timesteps // n

    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        eps = model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, ab_t, i - step_size, eps)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
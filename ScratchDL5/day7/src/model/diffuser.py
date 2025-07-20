import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# A Module that performs Forward diffusion process
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    # Forward Diffusion Process
    def add_noise(self, x0, ts): # why ts? : sampled timesteps for all datas (N,)
        T = self.num_timesteps
        assert (ts>=1).all() and (ts<=T).all()
        t_idxs = ts - 1

        N = x0.shape[0]
        alpha_bar = self.alpha_bars[t_idxs]
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise
    
    # A function for denoising at each step, provide xt, t, and also a condition which is 'label' for neural network UNet
    # Gamma is a guidance strength parameter
    def denoise(self, model, x, ts, labels, gamma=3.0):
        T = self.num_timesteps
        assert (ts>=1).all() and (ts<=T).all()

        # Hyperparameters
        t_idxs = ts - 1
        alpha = self.alphas[t_idxs]
        alpha_bar = self.alpha_bars[t_idxs]
        alpha_bar_prev = self.alpha_bars[t_idxs - 1]

        N = x.shape[0]
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        # Model Noise Prediction 
        model.eval()
        with torch.no_grad():
            # At Denoising(Generation Time), Predict both conditional and unconditional noise.
            # And add gamma * (cond - uncond) to the baseline uncond for guidance
            noise_uncond = model(x, ts)
            noise_cond = model(x, ts, labels)
            pred_noise = noise_uncond + gamma * (noise_cond - noise_uncond) 
        model.train()

        # Reparameterization Trick
        noise = torch.randn_like(x, device=self.device)
        noise[ts==1] = 0 # If timestep is 1 (1->0 denoising), make reparameterization gaussian noise to 0.

        mu = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise)
        sigma = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        x_prev = mu + sigma * noise

        return x_prev

    # Generation (Sampling) -> Also give condition(label) when sampling !
    def sample(self, model, gen_sample_shape=(20, 1, 28, 28), labels=None, gamma=3.0):
        batch_size = gen_sample_shape[0]
        x = torch.randn(gen_sample_shape, device = self.device) # extract complete gaussian noise x_t for sampling

        # Add condition when sampling
        if labels is None:
            labels = torch.randint(0, 10, (batch_size,), device=self.device)

        # Reverse Denoising from T for sampling
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels, gamma) # Add condition when denoising
        
        # Return generated x_0s to PIL Image
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels

    # Reverse to IMG from tensor
    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        to_pil = transforms.ToPILImage()
        return to_pil(x)
    

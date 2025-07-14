import math
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam

from configs.train_arguments import get_arguments
from src.dataset.get_dataloader import get_dataloader
from src.model.get_model import get_model
from src.utils import show_images

def train(model, diffuser, dataloader, criterion, optimizer, args):
    losses = []
    for epoch in range(args.num_train_epochs):
        print(f"==========================Epoch {epoch+1}============================")
        loss_sum = 0.0
        cnt = 0

        for imgs, _ in tqdm(dataloader):
            optimizer.zero_grad()
            
            # Sampling
            batch_size = imgs.shape[0]
            x = imgs.to(args.device) # Random Sampling Train Images and move to device
            t = torch.randint(1, args.num_timesteps+1, (batch_size,), device=args.device) # Random Sampling timestep for each data in Uniform distribution {1,T}

            # Forward Diffusion Process
            x_t, noise = diffuser.add_noise(x, t) # Forward Diffusion Process -> Directly get x_t step noised data form x0
            # Denoising (Model Noise Prediction)
            noise_pred = model(x_t, t)

            loss = criterion(noise, noise_pred)
            loss.backward()
            optimizer.step()

            loss_sum += loss
            cnt += 1
        
        avg_epoch_loss = loss_sum / cnt
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} |||||||| Loss : {avg_epoch_loss}")
    
    return losses

def main(args):
    dataloader = get_dataloader(args)
    model, diffuser = get_model(args)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    losses = train(model, diffuser, dataloader, criterion, optimizer, args)
    print("Training Finished!")
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('training_loss.png')
    plt.show()

    # Generate After Training
    images = diffuser.sample(model, gen_sample_shape=args.gen_sample_shape)
    show_images(images)
    print("Generation Finished!")

if __name__ == "__main__":
    args = get_arguments()
    args.gen_sample_shape = ast.literal_eval(args.gen_sample_shape)  # Ensure gen_sample_shape is a tuple
    main(args)

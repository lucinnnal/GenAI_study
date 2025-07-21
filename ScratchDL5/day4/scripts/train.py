import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from configs.train_arguments import get_arguments
from src.model.get_model import get_model
from src.dataset.get_dataset import get_train_dataset
from src.dataset.get_dataloader import get_dataloader

def train(model, dataloader, optimizer, args):
    losses = []
    for epoch in range(args.num_train_epochs):
        print(f"==========================Epoch {epoch+1}============================")
        loss_sum = 0.0
        cnt = 0

        for imgs, labels in tqdm(dataloader):
            optimizer.zero_grad()
            
            # Batch Loading
            x = imgs.to(args.device) # Random Sampling Train Images and move to device
            labels = labels.to(args.device)

            # Forward + Loss Calculation
            loss = model.get_loss(x)

            # Backward
            loss.backward()
            # Parameter Update
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
        
        avg_epoch_loss = loss_sum / cnt
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} |||||||| Loss : {avg_epoch_loss}")
    
    return losses

def main(args):
    dataloader = get_dataloader(args)

    model = get_model(args)
    model.to(args.device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = train(model, dataloader, optimizer, args)
    print("Training Finished!")
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('training_loss.png')
    plt.show()

    # Generate After Training
    model.eval()
    with torch.no_grad():
        sample_size = 64

        # Gaussian latent variable sampling
        z = torch.randn(sample_size, args.latent_dim, device=args.device)
        x = model.decoder(z)
        generated_imgs = x.view(sample_size, 1, 28, 28)

    grid_imgs = torchvision.utils.make_grid(
        generated_imgs,
        nrow = 8,
        padding = 2,
        normalize = True
    )

    plt.imshow(grid_imgs.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.savefig('vae_generated_imgs.png')
    plt.show()

    print("Generation Finished!")

if __name__ == "__main__":
    args = get_arguments()
    main(args)

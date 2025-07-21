import argparse

def get_arguments():
    
    parser = argparse.ArgumentParser(description="Variational Autoencoder MNIST Data Training Script")
    #================= parser with model  ===========================#
    parser.add_argument('--input_dim', type=int, default=28*28*1, help='Input image size to flatten 1d vector') # MNIST는 28*28*1 사이즈를 가짐
    parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden layer dimension')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent z dimension')    

    #================= parser with train  ===========================#    
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Per device train batch size')
    parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')

    args = parser.parse_args()
    return args
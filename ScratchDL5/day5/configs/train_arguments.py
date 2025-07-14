import argparse

def get_arguments():
    
    parser = argparse.ArgumentParser(description="Diffusion Model Training Script")
    #================= parser with data  ===========================#
    parser.add_argument('--img_size', type=int, default=28, help='Input image size')
    parser.add_argument('--gen_sample_shape', type=str, default="(20, 1, 28, 28)", help='Shape of generated samples')

    #================= parser with model  ===========================#
    parser.add_argument('--input_channel', type=int, default=1, help='Channel of input image')
    parser.add_argument('--time_encoding_dim', type=int, default=100, help='Timestep encoding vector dimension')

    #================= parser with train  ===========================#    
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Per device train batch size')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')

    args = parser.parse_args()
    return args

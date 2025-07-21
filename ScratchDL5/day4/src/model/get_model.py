from src.model.vae import VAE

def get_model(args):
    # VAE model
    model = VAE(args.input_dim, args.hidden_dim, args.latent_dim)
    
    return model
from src.model.unet import UNetCond
from src.model.diffuser import Diffuser

def get_model(args):
    # UNet model
    model = UNetCond(in_ch=args.input_channel, time_embed_dim=args.time_encoding_dim, num_labels=args.num_labels)
    
    # Initialize Diffuser
    diffuser = Diffuser(num_timesteps=args.num_timesteps, device=args.device)
    
    return model, diffuser
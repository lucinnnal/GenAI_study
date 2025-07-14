from src.model.unet import UNet
from src.model.diffuser import Diffuser

def get_model(args):
    # UNet model
    model = UNet(in_ch=args.input_channel, time_embed_dim=args.time_encoding_dim)
    
    # Initialize Diffuser
    diffuser = Diffuser(num_timesteps=args.num_timesteps, device=args.device)
    
    return model, diffuser
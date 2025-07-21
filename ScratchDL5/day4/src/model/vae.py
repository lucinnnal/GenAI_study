import torch
import torch.nn as nn
from src.utils import reparameterize
from src.model.encoder import Encoder
from src.model.decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)
        self.mse_loss = nn.MSELoss(reduction = 'sum') 
    
    # Calculate ELBO
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = x.size(0)
        l1 = self.mse_loss(x, x_hat) # Induction from ELBO's J1 Term : Recontruction Error
        l2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2) # Induction from ELBO's J2 Term : Regularization Error, torch.sum은 다차원 탠서의 모든 요소를 합함 (dim을 지정해준다면 그쪽 dim으로만 합함. keepdim 인자는 원래의 형상과 똑같이 유지한 상태로 sum을 함)

        return (l1 + l2) / batch_size
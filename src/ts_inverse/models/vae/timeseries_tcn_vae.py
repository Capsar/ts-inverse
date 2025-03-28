import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from typing import List
from src.dilate_time_series_losses import dilate_loss
from src.models.tcn import TCN
from src.models.vae import BaseVAE

# Assuming MaskDropout, Chomp1d, TemporalBlock, and TCN classes are already defined as provided

class TimeSeriesTCNVAE(BaseVAE):

    def __init__(self,
                 seq_length: int,
                 n_features: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 tcn_layers: int = 3,
                 tcn_kernel_size: int = 7,
                 tcn_dilation_factor: int = 2,
                 dropout: float = 0.2,
                 **kwargs) -> None:
        super(TimeSeriesTCNVAE, self).__init__()

        self.latent_dim = latent_dim
        self.n_features = n_features

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder with TCN components
        self.final_conv_output_size = seq_length
        self.encoder_tcn = TCN(
            n_features=n_features,
            seq_length=seq_length,
            hidden_size=hidden_dims[-1],
            num_levels=tcn_layers,
            kernel_size=tcn_kernel_size,
            dilation_factor=tcn_dilation_factor,
            dropout=dropout
        )
        self.final_conv_output_size = seq_length  # TCN does not reduce the sequence length

        self.encoder = nn.Sequential(self.encoder_tcn)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.final_conv_output_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.final_conv_output_size, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.final_conv_output_size)
        hidden_dims.reverse()

        self.decoder_tcn = TCN(
            n_features=hidden_dims[0],
            seq_length=self.final_conv_output_size,
            hidden_size=hidden_dims[0],
            num_levels=tcn_layers,
            kernel_size=tcn_kernel_size,
            dilation_factor=tcn_dilation_factor,
            dropout=dropout
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[0],
                               hidden_dims[0],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims[0], out_channels=self.n_features,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.decoder_input.out_features // self.final_conv_output_size, self.final_conv_output_size)
        result = self.decoder_tcn(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss = dilate_loss(recons, input, 0.5, 0.8)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
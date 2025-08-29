# stdlib
from typing import Any, List

# third party
import pandas as pd
import torch

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    IntegerDistribution,
)
from vect_gan.synthesizers.vectgan import VectGan


class VECT_GAN_plugin(Plugin):

    def __init__(
        self,
        latent_dim: int = 64,
        discriminator_dim: tuple[int, ...] = (32, 32, 32),
        generator_lr: float = 1e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-5,
        batch_size: int = 32,
        discriminator_steps: int = 2,
        log_frequency: bool = True,
        verbose: bool = True,
        epochs: int = 100,
        pac: int = 4,
        cuda: bool | str = True,
        kl_weight: float = 1.0,
        encoder_dim: tuple[int, ...] = (256, 512),
        encoder_lr: float = 1e-5,
        encoder_decay: float = 2e-6,
        vae_weight: float = 1.0,
        cond_loss_weight: float = 1.0,
        lambda_: float = 10.0,
        enforce_min_max_values: bool = True,
        **kwargs: Any) -> None:

        """Initialise the VectGan model.

        Args:
            latent_dim: Size of the random sample passed to the Generator.
            discriminator_dim: Sequence of linear layer sizes for the Discriminator.
            generator_lr: Learning rate for the Generator.
            generator_decay: Weight decay for the Generator optimiser.
            discriminator_lr: Learning rate for the Discriminator.
            discriminator_decay: Weight decay for the Discriminator optimiser.
            batch_size: Number of data samples to process in each step.
            discriminator_steps: Number of Discriminator updates per Generator update.
            log_frequency: Whether to use log frequency in conditional sampling.
            verbose: Whether to print progress.
            epochs: Number of training epochs.
            pac: Number of samples to group together in the Discriminator.
            cuda: Whether (or which GPU) to use for computation.
            kl_weight: Weight for the KL divergence in the VAE.
            encoder_dim: Sequence of dimensions for Residual layers in the VAE encoder.
            encoder_lr: Learning rate for the VAE encoder.
            encoder_decay: Weight decay for the VAE encoder optimiser.
            vae_weight: Weight for the VAE loss.
            cond_loss_weight: Weight for the conditional loss.
            lambda_: Weight for the gradient penalty.
            enforce_min_max_values: Whether to enforce min/max values in generation.
        """

        super().__init__(**kwargs)

        self.model = VectGan(
            latent_dim=latent_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            enforce_min_max_values=enforce_min_max_values
        )
       
        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._kl_weight = float(kl_weight)
        self._encoder_dim = encoder_dim
        self._encoder_lr = encoder_lr
        self.encoder_decay = encoder_decay
        self.vae_weight = vae_weight
        self.cond_loss_weight = cond_loss_weight
        self.lambda_ = lambda_

    @staticmethod
    def name() -> str:
        return "vect_gan"

    @staticmethod
    def type() -> str:
        return "tabular"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return [
            CategoricalDistribution(name='discriminator_dim', choices=[(32, 32, 32), (64, 64, 64), (128, 128, 128), (64, 32, 16)]),
            CategoricalDistribution(name='generator_lr', choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name='generator_decay', choices=[1e-3, 1e-4, 1e-5, 1e-6]),
            CategoricalDistribution(name='discriminator_lr', choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name='discriminator_decay', choices=[1e-3, 1e-4, 1e-5, 1e-6]),
            CategoricalDistribution(name='batch_size', choices=[32, 64, 128, 256]),
            IntegerDistribution(name='discriminator_steps', low=1, high=5),
            IntegerDistribution(name='epochs', low=100, high=1000, step=100),
            CategoricalDistribution(name='pac', choices=[1, 2, 4, 8]),
            CategoricalDistribution(name='encoder_dim', choices=[(64, 128), (128, 256), (256, 512)]),
            CategoricalDistribution(name='encoder_lr', choices=[1e-3, 1e-4, 1e-5]),
            CategoricalDistribution(name='encoder_decay', choices=[1e-3, 1e-4, 1e-5, 1e-6]),
            CategoricalDistribution(name='vae_weight', choices=[0.1, 1.0, 10.0]),
            CategoricalDistribution(name='cond_loss_weight', choices=[0.1, 1.0, 10.0]),
            CategoricalDistribution(name='lambda_', choices=[0.1, 1.0, 10.0])
        ]

    def _fit(
        self,
        X: DataLoader,
        discrete_columns: list[int | str] = (),
        fine_tuning: bool = False,
        *args: Any, 
        **kwargs: Any) -> "VECT_GAN_plugin":
        
        train_data = X.dataframe()
        
        self.model.fit(
            train_data = train_data,
            discrete_columns = discrete_columns,
            fine_tuning = fine_tuning
        )

        return self

    def _generate(self, syn_schema: Schema, count: int, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)
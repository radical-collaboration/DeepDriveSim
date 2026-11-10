"""GMVAE PyTorch Model Configuration"""

from typing import List, Optional, Tuple

from pydantic import validator

from workflows.ddmd_workflow.config import MachineLearningTaskConfig


class GMVAEModelConfig(MachineLearningTaskConfig):
    # Select the n most recent HDF5 files for training
    last_n_h5_files: int = 10
    # Select k random HDF5 files to train on from previous DeepDriveMD iterations
    k_random_old_h5_files: int = 0
    # Name of the dataset in the HDF5 file.
    dataset_name: str = "contact_map"
    # Shape of contact maps stored in HDF5 file
    initial_shape: Tuple[int, int] = (28, 28)
    # Shape of contact maps after cropping (flattened before being passed to GMVAE)
    final_shape: Tuple[int, int] = (28, 28)
    # Number of epochs to train during first iteration
    initial_epochs: int = 10
    # Number of epochs to train on later iterations
    epochs: int = 10
    # Training batch size
    batch_size: int = 32
    # Percentage of data to use as training data (the rest is validation)
    split_pct: float = 0.8
    # Whether or not to shuffle training/validation data
    shuffle: bool = True

    # GMVAE Model hyperparameters
    # Number of Gaussian mixture components (clusters)
    k: int = 10
    # Latent dimension
    n_z: int = 64
    # q(y|x) hidden dimensions
    qy_dims: List[int] = [128, 64]
    # q(z|x,y) hidden dimensions
    qz_dims: List[int] = [128, 64]
    # p(x|z) hidden dimensions
    px_dims: List[int] = [64, 128]
    # p(z|y) hidden dimensions
    pz_dims: List[int] = [64]
    # Entropy regularization weight
    r_nent: float = 1.0
    # Learning rate
    lr: float = 0.0001
    # Reconstruction likelihood p(x|z): "bernoulli" for binary contact maps,
    # "gaussian" for continuous inputs
    likelihood: str = "bernoulli"

    @validator("likelihood")
    def likelihood_check(cls, v: str) -> str:
        valid_likelihoods = {"bernoulli", "gaussian"}
        if v not in valid_likelihoods:
            raise ValueError(f"likelihood must be one of {valid_likelihoods}, not {v}.")
        return v

    # Project name for wandb logging
    wandb_project_name: Optional[str] = None

    @property
    def input_dim(self) -> int:
        # Flattened contact map size passed to the GMVAE
        return self.final_shape[0] * self.final_shape[1]


if __name__ == "__main__":
    GMVAEModelConfig().dump_yaml("gmvae_pytorch_template.yaml")

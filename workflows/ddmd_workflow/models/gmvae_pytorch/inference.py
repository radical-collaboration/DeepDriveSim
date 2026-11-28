import itertools
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt

import numpy as np
import torch  # type: ignore[import]
from torch.utils.data import DataLoader, Dataset, Subset  # type: ignore[import]

from workflows.ddmd_workflow.models.gmvae_pytorch.config import GMVAEModelConfig
from workflows.ddmd_workflow.models.gmvae_pytorch.GMVAE.dataset import TorchDataset
from workflows.ddmd_workflow.models.gmvae_pytorch.train import build_model
from workflows.ddmd_workflow.models.keras_cvae.utils import sparse_to_dense
from workflows.ddmd_workflow.utils import PathLike, setup_mpi


def shard_dataset(dataset: Dataset, comm_size: int, comm_rank: int) -> Dataset:

    fullsize = len(dataset)
    chunksize = fullsize // comm_size
    start = comm_rank * chunksize
    end = start + chunksize
    subset_indices = list(range(start, end))
    # deal with remainder
    for idx, i in enumerate(range(comm_size * chunksize, fullsize)):
        if idx == comm_rank:
            subset_indices.append(i)
    # split the set
    dataset = Subset(dataset, subset_indices)

    return dataset


def _run_inference(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
    encoder_gpu: int,
    comm: Optional[Any],
    collect: Callable[[torch.Tensor, Dict[str, Any]], Tuple["npt.ArrayLike", ...]],
) -> List["npt.ArrayLike"]:
    """Run the GMVAE over every contact map in :obj:`h5_file`.

    :obj:`collect` maps a batch and the model outputs to a tuple of
    numpy arrays. Returns one concatenated array per tuple element,
    gathered across ranks if :obj:`comm` is given.
    """
    comm_size, comm_rank = setup_mpi(comm)

    model_cfg = GMVAEModelConfig.from_yaml(model_cfg_path)

    model = build_model(model_cfg, model_path=".")  # model_path not used for inference
    state_dict = torch.load(str(model_weights_path), map_location="cpu")
    model.load_state_dict(state_dict)

    data = sparse_to_dense(
        h5_file, model_cfg.dataset_name, model_cfg.initial_shape, model_cfg.final_shape
    )
    # Flatten (N, D, D) contact maps to (N, D * D) for the fully connected GMVAE
    data = np.reshape(data, (len(data), -1)).astype(np.float32)  # type: ignore[arg-type]
    dataset: Dataset = TorchDataset(data)

    # shard the dataset
    if comm_size > 1:
        dataset = shard_dataset(dataset, comm_size, comm_rank)

    # Put model on specified CPU/GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{encoder_gpu}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    # create data loader (requires shuffle=False to keep frame order)
    data_loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        drop_last=False,
        shuffle=False,
    )

    results: List[List["npt.ArrayLike"]] = []
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch, training=False)
            results.append(list(collect(batch, outputs)))

        if (comm_rank == 0) and (i % 100 == 0):
            print(f"Batch {i}/{len(data_loader)}")

    # Transpose list of per-batch tuples into per-output lists
    per_output = [list(col) for col in zip(*results)]

    if comm_size > 1:
        # gather results
        per_output = [
            list(itertools.chain.from_iterable(comm.allgather(col)))  # type: ignore[union-attr]
            for col in per_output
        ]

    return [np.concatenate(col) for col in per_output]  # type: ignore[no-untyped-call]


def generate_embeddings(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
    encoder_gpu: int = 0,
    comm: Optional[Any] = None,
    return_cluster_probs: bool = False,
) -> "npt.ArrayLike":
    """
    Returns the qy-weighted average latent embedding [n_samples, n_z],
    and optionally the cluster probabilities qy [n_samples, k].
    """

    def collect(batch: torch.Tensor, outputs: Dict[str, Any]) -> Tuple[Any, ...]:
        qy = outputs["qy"]  # [batch, k] - cluster assignment probabilities
        # Stack the q(z|x,y) means (not samples) from all clusters so embeddings
        # are deterministic, like the CVAE's z_mean: [batch, k, n_z]
        z_stacked = torch.stack(outputs["zm"], dim=1)
        # weighted_z = sum_k(qy[:, k] * z_stacked[:, k, :])
        weighted_z = torch.sum(qy.unsqueeze(-1) * z_stacked, dim=1)  # [batch, n_z]
        return weighted_z.cpu().numpy(), qy.cpu().numpy()

    latent_embeddings, cluster_probs = _run_inference(
        model_cfg_path,
        h5_file,
        model_weights_path,
        inference_batch_size,
        encoder_gpu,
        comm,
        collect,
    )

    if return_cluster_probs:
        return latent_embeddings, cluster_probs

    return latent_embeddings


def generate_embeddings_per_cluster(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
    encoder_gpu: int = 0,
    comm: Optional[Any] = None,
) -> Tuple["npt.ArrayLike", "npt.ArrayLike"]:
    """
    Generate embeddings for each cluster separately.
    Returns: (embeddings, cluster_assignments)
    - embeddings: [n_samples, n_z] latent embeddings from most probable cluster
    - cluster_assignments: [n_samples] cluster index for each sample
    """

    def collect(batch: torch.Tensor, outputs: Dict[str, Any]) -> Tuple[Any, ...]:
        # Get most probable cluster for each sample
        cluster_idx = torch.argmax(outputs["qy"], dim=1)  # [batch]
        # q(z|x,y) means, for deterministic embeddings: [batch, k, n_z]
        z_stacked = torch.stack(outputs["zm"], dim=1)
        # Select z from the most probable cluster for each sample
        batch_indices = torch.arange(batch.shape[0], device=batch.device)
        selected_z = z_stacked[batch_indices, cluster_idx]  # [batch, n_z]
        return selected_z.cpu().numpy(), cluster_idx.cpu().numpy()

    embeddings, cluster_assignments = _run_inference(
        model_cfg_path,
        h5_file,
        model_weights_path,
        inference_batch_size,
        encoder_gpu,
        comm,
        collect,
    )
    return embeddings, cluster_assignments


def generate_reconstructions(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
    encoder_gpu: int = 0,
    comm: Optional[Any] = None,
) -> "npt.ArrayLike":
    """
    Generate reconstructions from the model (for validation/debugging).
    Returns: [n_samples, n_x] qy-weighted reconstructed contact maps (flattened).
    For the bernoulli likelihood these are contact probabilities in [0, 1].
    """
    likelihood = GMVAEModelConfig.from_yaml(model_cfg_path).likelihood

    def collect(batch: torch.Tensor, outputs: Dict[str, Any]) -> Tuple[Any, ...]:
        qy = outputs["qy"]  # [batch, k]
        xm_stacked = torch.stack(outputs["xm"], dim=1)  # [batch, k, n_x]
        if likelihood == "bernoulli":
            # Decoder outputs are logits; convert to contact probabilities
            xm_stacked = torch.sigmoid(xm_stacked)
        weighted_xm = torch.sum(qy.unsqueeze(-1) * xm_stacked, dim=1)  # [batch, n_x]
        return (weighted_xm.cpu().numpy(),)

    (reconstructions,) = _run_inference(
        model_cfg_path,
        h5_file,
        model_weights_path,
        inference_batch_size,
        encoder_gpu,
        comm,
        collect,
    )
    return reconstructions

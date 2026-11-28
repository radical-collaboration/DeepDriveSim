import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt

# Disable HDF5 file locking for parallel filesystems (GPFS/Lustre)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch  # type: ignore[import]
from torch.utils.data import DataLoader  # type: ignore[import]

from workflows.ddmd_workflow.data.api import DeepDriveMD_API
from workflows.ddmd_workflow.data.utils import get_virtual_h5_file
from workflows.ddmd_workflow.models.gmvae_pytorch.config import GMVAEModelConfig
from workflows.ddmd_workflow.models.gmvae_pytorch.GMVAE.dataset import TorchDataset
from workflows.ddmd_workflow.models.gmvae_pytorch.GMVAE.model import GMVAE
from workflows.ddmd_workflow.models.keras_cvae.utils import sparse_to_dense
from workflows.ddmd_workflow.selection.latest.select_model import get_model_path
from workflows.ddmd_workflow.utils import Timer, parse_args


def get_init_weights(cfg: GMVAEModelConfig) -> Optional[str]:
    if cfg.init_weights_path is None:
        if cfg.stage_idx == 0:
            # Case for first iteration with no pretrained weights
            return None

        token = get_model_path(
            stage_idx=cfg.stage_idx - 1, experiment_dir=cfg.experiment_directory
        )
        if token is None:
            # Case for no pretrained weights
            return None
        else:
            # Case where model selection has run before
            _, init_weights = token
    else:
        # Case for pretrained weights
        init_weights = cfg.init_weights_path

    return init_weights.as_posix()


def get_h5_training_file(cfg: GMVAEModelConfig) -> Tuple[Path, List[str]]:
    # Collect training data
    api = DeepDriveMD_API(cfg.experiment_directory)
    md_data = api.get_last_n_md_runs()
    all_h5_files = md_data["data_files"]

    virtual_h5_path, h5_files = get_virtual_h5_file(
        output_path=cfg.output_path,
        all_h5_files=all_h5_files,
        last_n=cfg.last_n_h5_files,
        k_random_old=cfg.k_random_old_h5_files,
        virtual_name=f"virtual_{cfg.model_tag}",
        node_local_path=cfg.node_local_path,
    )

    return virtual_h5_path, h5_files


def preprocess(
    h5_file: Path,
    initial_shape: Tuple[int, int],
    final_shape: Tuple[int, int],
    dataset_name: str = "contact_map",
    split_pct: float = 0.8,
    shuffle: bool = True,
) -> Tuple["npt.ArrayLike", "npt.ArrayLike"]:

    data = sparse_to_dense(h5_file, dataset_name, initial_shape, final_shape)
    # Flatten (N, D, D) contact maps to (N, D * D) for the fully connected GMVAE
    data = np.reshape(data, (len(data), -1)).astype(np.float32)  # type: ignore[arg-type]

    if shuffle:
        np.random.shuffle(data)

    # Split data into train and validation
    train_val_split = int(split_pct * len(data))
    train_data, valid_data = data[:train_val_split], data[train_val_split:]

    return train_data, valid_data


def build_model(cfg: GMVAEModelConfig, model_path: str) -> GMVAE:
    model = GMVAE(
        model_path=model_path,
        k=cfg.k,
        n_x=cfg.input_dim,
        n_z=cfg.n_z,
        qy_dims=cfg.qy_dims,
        qz_dims=cfg.qz_dims,
        px_dims=cfg.px_dims,
        pz_dims=cfg.pz_dims,
        r_nent=cfg.r_nent,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        likelihood=cfg.likelihood,
    )
    # Run a dummy forward pass to initialize the LazyLinear layers
    with torch.no_grad():
        model(torch.zeros(1, cfg.input_dim), training=False)
    return model


def run_epoch(
    model: GMVAE, loader: DataLoader, device: torch.device, training: bool
) -> Dict[str, float]:
    loss_accum = {"loss": 0.0, "nent": 0.0, "recon": 0.0}
    num_batches = 0

    for data in loader:
        data = data.to(device)
        if training:
            loss_dict = model.train_steps(data)
        else:
            with torch.no_grad():
                outputs = model(data, training=False)
                _, loss_dict = model.compute_loss(data, outputs, training=False)

        for key in loss_accum:
            loss_accum[key] += loss_dict[key]
        num_batches += 1

    return {key: val / max(num_batches, 1) for key, val in loss_accum.items()}


def main(cfg: GMVAEModelConfig) -> None:

    cfg.output_path.mkdir(exist_ok=True)

    with Timer("machine_learning_get_init_weights"):
        init_weights = get_init_weights(cfg)

    with Timer("machine_learning_get_h5_training_file"):
        h5_file, h5_files = get_h5_training_file(cfg)

    # Log selected H5 files
    with open(cfg.output_path / "virtual-h5-metadata.json", "w") as f:
        json.dump(h5_files, f)

    with Timer("machine_learning_preprocess"):
        train_data, valid_data = preprocess(
            h5_file,
            cfg.initial_shape,
            cfg.final_shape,
            cfg.dataset_name,
            cfg.split_pct,
            cfg.shuffle,
        )

    print(
        f"Having {len(train_data)} training and {len(valid_data)} validation samples."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with Timer("machine_learning_gmvae"):
        model = build_model(cfg, cfg.output_path.as_posix())

    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location="cpu"))

    model.to(device)
    print(model)

    train_loader = DataLoader(
        TorchDataset(train_data),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        drop_last=True,
    )
    valid_loader = DataLoader(
        TorchDataset(valid_data),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    wandb = None
    if cfg.wandb_project_name is not None:
        import wandb  # type: ignore[import, no-redef]

        wandb.init(
            project=cfg.wandb_project_name,
            name=cfg.model_tag,
            id=cfg.model_tag,
            dir=cfg.output_path.as_posix(),
            config=cfg.dict(),
            resume=False,
        )

    # Optionaly train for a different number of
    # epochs on the first DDMD iterations
    if cfg.stage_idx == 0:
        epochs = cfg.initial_epochs
    else:
        epochs = cfg.epochs

    history: Dict[str, List[float]] = {
        f"{split}_{key}": []
        for split in ("train", "valid")
        for key in ("loss", "nent", "recon")
    }

    with Timer("machine_learning_train"):
        for epoch in range(epochs):
            train_loss = run_epoch(model, train_loader, device, training=True)
            valid_loss = run_epoch(model, valid_loader, device, training=False)

            for key in train_loss:
                history[f"train_{key}"].append(train_loss[key])
                history[f"valid_{key}"].append(valid_loss[key])

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss['loss']:.4f} "
                f"(NENT: {train_loss['nent']:.4f}, Recon: {train_loss['recon']:.4f}), "
                f"Valid Loss: {valid_loss['loss']:.4f} "
                f"(NENT: {valid_loss['nent']:.4f}, Recon: {valid_loss['recon']:.4f})"
            )

            if wandb is not None:
                wandb.log(
                    {**{k: v[-1] for k, v in history.items()}, "epoch": epoch}
                )

    # Log checkpoint
    with Timer("machine_learning_logging"):
        checkpoint_path = cfg.output_path / "checkpoint"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Format must match selection/latest: epoch-<N>-<timestamp>.pt
        time_stamp = time.strftime(f"epoch-{epochs}-%Y%m%d-%H%M%S.pt")
        torch.save(model.state_dict(), checkpoint_path / time_stamp)

        # Log loss history
        with open(cfg.output_path / "loss.json", "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    with Timer("machine_learning_stage"):
        args = parse_args()
        cfg = GMVAEModelConfig.from_yaml(args.config)
        main(cfg)

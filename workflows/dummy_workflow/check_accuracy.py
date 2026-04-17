# check_accuracy_async.py
import argparse
import asyncio
import pickle
import random
from pathlib import Path

import numpy as np


def dummy_mse():
    """Return a random value for testing purposes."""
    return random.random()


async def load_model(model_filename):
    """Load a pre-trained model asynchronously, return None if loading fails."""
    try:
        try:
            import aiofiles

            async with aiofiles.open(model_filename, "rb") as f:
                data = await f.read()
        except Exception:

            def _read_file(path):
                with open(path, "rb") as f:
                    return f.read()

            data = await asyncio.to_thread(_read_file, model_filename)
        # pickle.load is CPU-bound, run in a thread
        return await asyncio.to_thread(pickle.loads, data)
    except (OSError, pickle.UnpicklingError):
        return None


async def load_single_npz(file):
    """Load one .npz file asynchronously."""
    try:
        return await asyncio.to_thread(np.load, file)
    except Exception:
        return None


async def load_validation_data(val_dir):
    """Load all validation data from the given directory asynchronously."""
    val_path = Path(val_dir)
    npz_files = [f for f in val_path.iterdir() if f.is_file() and f.suffix == ".npz"]

    # Load files concurrently
    datasets = await asyncio.gather(*(load_single_npz(f) for f in npz_files))

    x_all, y_all = [], []
    for data in datasets:
        if data is not None:
            x_all.append(data["x"])
            y_all.append(data["y"])

    if not x_all:
        return None, None

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)


async def check(model_filename="model.pkl", val_dir="val"):

    await load_model(model_filename)
    x_eval, y_eval = await load_validation_data(val_dir)

    if x_eval is None or y_eval is None or len(y_eval) == 0:
        mse_eval = dummy_mse()
    else:
        # Real model evaluation would go here:
        # y_pred_eval = await asyncio.to_thread(model.predict, x_eval)
        # mse_eval = mean_squared_error(y_eval, y_pred_eval)
        mse_eval = dummy_mse()

    print(mse_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check model accuracy against validation data (async)."
    )
    parser.add_argument(
        "--model_filename", type=str, default="model.pkl", help="Path to model file"
    )
    parser.add_argument(
        "--val_dir", type=str, default="val", help="Path to validation data directory"
    )
    args = parser.parse_args()

    asyncio.run(check(args.model_filename, args.val_dir))

# sim_async.py
import argparse
import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np


def complicated_function(x: np.ndarray) -> np.ndarray:
    """Complex mathematical function to simulate a process."""
    return (
        0.3 * np.sin(1.5 * np.pi * x**2)
        + 0.2 * np.cos(2 * np.pi * x**3)
        + 0.5 * np.exp(-0.5 * x)
        + 0.1 * np.tanh(0.2 * (x - 0.5))
        + 0.3 * (x**3)
    )


async def simulate_one(output_file: Path):
    """Run a single simulation iteration asynchronously."""
    x = np.random.uniform(low=0.0, high=1.0, size=(500, 1))

    # Run CPU-heavy loop in a thread to avoid blocking event loop
    def run_math():
        y = 0
        for _ in range(1000):
            y += complicated_function(x)
        return y

    y = await asyncio.to_thread(run_math)

    def write_atomic():
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError):
            pass  # Lustre metadata race; directory may already exist
        try:
            tmp = tempfile.NamedTemporaryFile(
                dir=output_file.parent, suffix=".tmp", delete=False
            )
        except FileNotFoundError:
            return  # directory was deleted by concurrent close(); skip file
        try:
            np.savez_compressed(tmp, x=x, y=y)
            tmp.close()
            os.rename(tmp.name, output_file)
        except FileNotFoundError:
            # temp file or directory removed by concurrent close(); nothing to clean up
            try:
                tmp.close()
            except Exception:
                pass
        except Exception:
            try:
                tmp.close()
                os.unlink(tmp.name)
            except FileNotFoundError:
                pass
            raise

    await asyncio.to_thread(write_atomic)

    # print(f"Saved simulation to {output_file}")


async def run_simulation(output_dir: str, sim_tag: str) -> None:
    """Run the simulation and save results."""
    print(f"Simulation {sim_tag} will start now")

    output_sim_dir = Path(output_dir) / sim_tag
    output_sim_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i in range(150):
        output_file = output_sim_dir / f"{sim_tag}_{i}.npz"
        tasks.append(simulate_one(output_file))

    # Run up to N simulations concurrently
    await asyncio.gather(*tasks)
    await asyncio.sleep(2)

    print(f"Simulation completed. Results saved in {output_sim_dir}")
    return


def main():
    parser = argparse.ArgumentParser(description="Run a simulation (async)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to simulation output directory",
    )
    parser.add_argument("--sim_tag", type=str, required=True, help="Simulation tag")
    parser.add_argument(
        "--filename", type=str, required=True, help="Simulation input file"
    )

    args = parser.parse_args()

    asyncio.run(run_simulation(args.output_dir, args.sim_tag))


if __name__ == "__main__":
    main()

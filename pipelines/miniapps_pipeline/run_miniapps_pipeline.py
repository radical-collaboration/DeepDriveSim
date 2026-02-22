#!/usr/bin/env python3
import asyncio
import argparse
from radical.asyncflow import WorkflowEngine
from pipelines.miniapps_pipeline.miniapps_pipeline import MiniAppsWorkflow


SIM_CORES = 3  # For Testing only we set 3 CPUs for simulations
TRAIN_CORE = 1  # For Testing only we set 1 CPUs for training


async def run_miniapps(config_file, use_dragon):

    if use_dragon:
        try:
            from rhapsody.backends import DragonExecutionBackendV3
        except:
            use_dragon = False

    if use_dragon:
        engine = await DragonExecutionBackendV3()
    else:
        from rhapsody.backends import ConcurrentExecutionBackend
        engine = await ConcurrentExecutionBackend()

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)

    # Initialize the workflow
    workflow = MiniAppsWorkflow(
        asyncflow=asyncflow, training_cores=TRAIN_CORE, max_sim_batch=SIM_CORES
    )

    try:
        # Run the workflow
        await workflow.start()
    except Exception as e:
        print(f"An error occurred during teaching: {e}")
    finally:
        # Ensure cleanup regardless of errors
        await workflow.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py", description="Dummy DDSim pipeline"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to pipeline configuration file",
    )

    parser.add_argument("--use_dragon", action="store_true", help="Use Dragon backend")

    args = parser.parse_args()

    asyncio.run(run_miniapps(args.config_file, args.use_dragon))

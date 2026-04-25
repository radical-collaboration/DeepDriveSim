#!/usr/bin/env python3
import argparse
import asyncio

from radical.asyncflow import WorkflowEngine

from ddsim.util import find_gpus, load_config, make_policies
from workflows.exchange_workflow.exchange_workflow import ExchangeWorkflow


async def run_exchange(config_file: str) -> None:
    cfg = load_config(config_file)
    backend = cfg.get("engine", "dragon")
    num_replicas = cfg.get("num_replicas", 1)

    # --- Build backend and asyncflow ---
    policy = None

    if backend == "dragon":
        try:
            from rhapsody.backends import DragonExecutionBackendV3

            engine_dragon = await DragonExecutionBackendV3()
            asyncflow = await WorkflowEngine.create(engine_dragon)
            policies = make_policies(find_gpus(), nprocs=num_replicas)
            policy = policies[0] if policies else None
        except ImportError:
            backend = "concurrent"

    else:
        from rhapsody.backends import ConcurrentExecutionBackend

        engine_concurrent = await ConcurrentExecutionBackend()
        asyncflow = await WorkflowEngine.create(engine_concurrent)

    # --- Telemetry
    telemetry = None
    if cfg.get("telemetry", True):
        telemetry_dir = cfg.get("telemetry_dir", "telemetry-output")
        if hasattr(asyncflow, "start_telemetry"):
            telemetry = await asyncflow.start_telemetry(
                resource_poll_interval=0.5,
                checkpoint_path=telemetry_dir,
            )
            print("Started Asyncflow telemetry ...")

    workflow = ExchangeWorkflow(
        config=cfg,
        asyncflow=asyncflow,
        policies=[policy] if policy is not None else [],
    )

    try:
        await workflow.start()
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
    finally:
        try:
            summary = telemetry.summary()
            print(f"Tasks — {summary['tasks']}")
            if summary.get("duration"):
                d = summary["duration"]
                print(
                    f"Mean task time: {d['mean_seconds']:.2f} s  "
                    f"Max: {d['max_seconds']:.2f} s"
                )
        except Exception:
            pass

        if telemetry:
            await telemetry.stop()

        # ExchangeWorkflow.close() calls asyncflow.shutdown() internally.
        await workflow.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py", description="Exchange workflow entry point"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to workflow config file (default: config.yaml next to this script)",
    )

    args = parser.parse_args()

    asyncio.run(run_exchange(args.config_file))

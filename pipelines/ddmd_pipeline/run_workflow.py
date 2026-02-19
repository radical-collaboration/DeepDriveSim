#!/usr/bin/env python3
import asyncio
import argparse
from radical.asyncflow import WorkflowEngine
from ddmd_pipeline import DDMdWorkflow
from pipelines.ddmd_pipeline.utils import parse_args

SIM_CORES = 3   # For Testing only we set 3 CPUs for simulations
TRAIN_CORE = 1  # For Testing only we set 1 CPUs for training

async def run_ddmd(config, use_dragon=False):

    if use_dragon:
        try:
            from rhapsody.backends import DragonExecutionBackendV3
        except:
            use_dragon = False

    if use_dragon:
        engine = await DragonExecutionBackendV3()
    else:
        from rhapsody.backends import ConcurrentExecutionBackend
        #from radical.asyncflow import ConcurrentExecutionBackend
        #from concurrent.futures import ThreadPoolExecutor
        #engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        engine = await ConcurrentExecutionBackend()

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)
    
    # Initialize the workflow
    workflow = DDMdWorkflow(asyncflow=asyncflow, config=config)
    
    #try:
        # Run the workflow
    await workflow.start()
    # except Exception as e:
    #     print(f"An error occurred during teaching: {e}")
    # finally:
    #     # Ensure cleanup regardless of errors
    await workflow.close()

if __name__ == '__main__':

    args = parse_args()

    asyncio.run(run_ddmd(args.config))

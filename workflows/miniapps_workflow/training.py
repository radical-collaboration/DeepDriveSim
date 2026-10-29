#!/usr/bin/env python

import argparse
import os
import socket
import time

import wfMiniAPI.kernel as wf

try:  # optional: report data handoffs to the workflow (no-op when not launched by it)
    from ddsim import task_telemetry
except Exception:  # pragma: no cover - telemetry must never break the task
    task_telemetry = None


def parse_args():
    parser = argparse.ArgumentParser(description="Exalearn_miniapp_training")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--device", default="gpu", help="Wheter this is running on cpu or gpu"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="the current phase of workflow, phase0 will not read model",
    )
    parser.add_argument(
        "--data_root_dir", default="./", help="the root dir of gsas output data"
    )
    parser.add_argument(
        "--model_dir", default="./", help="the directory where save and load model"
    )
    parser.add_argument(
        "--num_sample", type=int, default=100, help="num of samples in matrix mult"
    )
    parser.add_argument(
        "--num_mult", type=int, default=10, help="number of matrix mult to perform"
    )
    parser.add_argument(
        "--dense_dim_in",
        type=int,
        default=12544,
        help="dim for most heavy dense layer, input",
    )
    parser.add_argument(
        "--dense_dim_out",
        type=int,
        default=128,
        help="dim for most heavy dense layer, output",
    )
    parser.add_argument(
        "--mat_size",
        type=int,
        default=5000,
        help="the matrix size is mat_size * mat_size, the same as it is in simulation",
    )
    parser.add_argument(
        "--preprocess_time", type=float, default=20.0, help="time for doing preprocess"
    )
    parser.add_argument(
        "--read_size", type=int, default=0, help="size of bytes read from disk"
    )
    parser.add_argument(
        "--write_size",
        type=int,
        default=3500000,
        help="size of bytes written to disk, -1 means write data to disk once",
    )
    parser.add_argument(
        "--instance_index",
        type=int,
        required=True,
        help="use to distinguish different training task. Should be from 0~n-1",
    )

    args = parser.parse_args()

    return args


def main():

    print(
        f"Temp for Darshan, ml, PID = {os.getpid()}, hostname = {socket.gethostname()}"
    )
    start_time = time.time()

    args = parse_args()
    print(args)

    root_path = args.data_root_dir + f"/phase{args.phase}" + "/"
    print("root_path for data = ", root_path)

    device = args.device

    try:
        import cupy
        cupy.cuda.runtime.getDevice()
        device = "gpu"
    except Exception:
        device = "cpu"

    data_file = os.path.join(root_path, f"data_0_{args.instance_index}.h5")
    t_read = time.time()
    version = task_telemetry.file_version(data_file) if task_telemetry else ""
    wf.readNonMPI(args.read_size, root_path, args.instance_index)
    if task_telemetry:
        task_telemetry.data_read(data_file, args.read_size, t_read, version)
    wf.sleep(args.preprocess_time)
    wf.generateRandomNumber(device, args.num_sample * args.dense_dim_in)
    wf.generateRandomNumber(device, args.dense_dim_in * args.dense_dim_out)
    if device == "gpu":
        wf.dataCopyH2D(args.dense_dim_in * args.dense_dim_out)

    for _ in range(args.num_epochs):
        if device == "gpu":
            wf.dataCopyH2D(args.num_sample * args.dense_dim_in)
        for _ in range(args.num_mult):
            wf.matMulGeneral(
                device,
                [args.num_sample, args.dense_dim_in],
                [args.dense_dim_in, args.dense_dim_out],
                ([1], [0]),
            )
            wf.axpy(device, args.dense_dim_in * args.dense_dim_out)

    if device == "gpu":
        wf.dataCopyD2H(args.dense_dim_in * args.dense_dim_out)
    t_write = time.time()
    wf.writeNonMPI(args.write_size, root_path, args.instance_index)
    if task_telemetry:
        task_telemetry.data_written(data_file, args.write_size, t_write)

    end_time = time.time()
    print(f"Total training time is {end_time - start_time} seconds")


if __name__ == "__main__":
    main()

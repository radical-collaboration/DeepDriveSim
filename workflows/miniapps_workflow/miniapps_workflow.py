"""MiniApps workflow — Dragon/asyncflow backend, Delta HPC."""

import asyncio
import os
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import yaml

from ddsim import task_telemetry
from ddsim.ddsim_manager import DDSimManager

try:
    from rhapsody.telemetry import define_event
    from rhapsody.telemetry.events import make_event

    MiniAppsStageTimer = define_event(
        "ddsim.MiniAppsStageTimer",
        replica=str,
        stage=str,
        iteration=int,
        duration_ms=float,
    )
    # Raw data observations reported by task processes via ddsim.task_telemetry. t_start
    # = operation start; the event's event_time = operation end (write closed / read
    # done).
    DataWritten = define_event(
        "ddsim.DataWritten",
        replica=str,
        task_key=str,
        object_id=str,
        version=str,
        payload_bytes=int,
        t_start=float,
    )
    DataRead = define_event(
        "ddsim.DataRead",
        replica=str,
        task_key=str,
        object_id=str,
        version=str,
        payload_bytes=int,
        t_start=float,
    )
    # ED6 producer-to-consumer latency (spec A.5.6): one event per data handoff
    # (delivery).
    DataDelivery = define_event(
        "ddsim.DataDelivery",
        replica=str,
        iteration=int,
        exchange=str,  # sim_to_train | train_to_agent | agent_to_manager
        delivery_id=str,
        producer=str,
        consumer=str,
        object_id=str,
        version=str,
        payload_bytes=int,
        representation=str,
        t_ready=float,
        t_usable=float,
        latency_ms=float,  # t_usable - t_ready; -1 when status == "unknown"
        status=str,  # delivered | unexpected_producer | unknown
        producer_host=str,
        consumer_host=str,
    )
except ImportError:
    define_event = None
    make_event = None
    MiniAppsStageTimer = None
    DataWritten = DataRead = DataDelivery = None

# How the payload is "usable" by each consumer (spec 5.6: usability = complete payload
# at the required location *and representation*). Real weight formats (torch.load, GPU
# placement, validation) would add steps before usability; the mini-app only loads
# arrays / parses YAML.
_REPR_H5 = "numpy.float32 array via h5py (wfMiniAPI.readNonMPI)"
_REPR_YAML = "dict via yaml.safe_load"


class MiniAppsWorkflow(DDSimManager):
    """MiniApps workflow — Dragon/asyncflow backend."""

    def __init__(self, config: dict = None, **kwargs):
        cfg = config or {}
        super().__init__(name=kwargs.get("name", "miniapps"))
        self.debug = config.get("debug", False)

        self.flow = kwargs.get("asyncflow", None)
        self.telemetry = kwargs.get("telemetry", None)
        # Optional ddsim.otel_export.OtelExporter: also post metrics to an OTLP endpoint
        self.otel = kwargs.get("otel", None)
        self._on_ready = kwargs.get("on_ready", None)
        self._data_ready_signaled = False
        self.miniapps_data_ready = int(cfg.get("miniapps_data_ready", 3))

        _home_base = Path(
            kwargs.get("home_dir", cfg.get("home_dir", Path.home() / "MiniApps"))
        )
        self.home_dir = self._ensure_dir(_home_base / self.name)
        self.clean_dir(self.home_dir)

        self.sim_output_dir = self._ensure_dir(self.home_dir / "sim_output")

        _default_src = str(Path(__file__).parent)
        self.src_dir = cfg.get("src_dir") or _default_src

        _default_exe = os.path.expandvars(cfg.get("executable") or "") or sys.executable

        def _exe(key):
            val = cfg.get(key)
            return os.path.expandvars(val) if val else _default_exe

        self.sim_executable = _exe("sim_executable")
        self.train_executable = _exe("train_executable")
        self.predict_executable = _exe("predict_executable")
        self.selection_executable = _exe("selection_executable")

        self.prediction_file = self.home_dir / "predictions.yaml"

        self.max_sim_batch = int(cfg.get("max_sim_batch", 4))
        self.training_cores = int(cfg.get("training_cores", 1))
        self.sim_batch_size = self.max_sim_batch + self.training_cores
        self.free_resources_for_train = bool(cfg.get("free_resources_for_train", True))

        self.call_cancel_simulations = True
        self.call_finalize_results = True
        self.call_evaluate_simulations = True

        self.total_num_sim = int(cfg.get("total_num_sim", 25))
        self.iteration = 0
        self.phase = int(cfg.get("phase", 0))
        self.num_step = int(cfg.get("num_step", 1000))
        self.num_epochs = int(cfg.get("num_epochs", 1000))
        self.num_mult = int(cfg.get("num_epochs", 1))
        self.retrain_model = True
        self.sim_predictions = {}

        self.clean_unregistered_sims = bool(cfg.get("clean_unregistered_sims", False))
        self.engine = cfg.get("engine", "concurrent")

        # Bytes the consumers read from their input file. The stock scripts default to 0
        # (open/close only); default here = the producers' default write_size (3.5 MB),
        # i.e. the whole payload. Set to 0 in the config to reproduce stock behaviour.
        self.train_read_size = int(cfg.get("train_read_size", 3500000))
        self.agent_read_size = int(cfg.get("agent_read_size", 3500000))

        # ED6 bookkeeping
        self._sim_attempts = {}  # sim_idx -> number of launches so far
        self._sim_keys = {}  # sim_idx -> task key of the latest launch
        # (object_id, version) -> DataWritten record (who wrote it, when)
        self._writes = {}
        self._pending = []  # deliveries whose producer write has not been reported yet

        self.register_tasks()
        self.sim_inputs = {}

    # --------------------------------------------------------------------------
    @staticmethod
    def _ensure_dir(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def clean_dir(dir_name):
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

    def _emit_stage(self, stage: str, duration_ms: float) -> None:
        if self.telemetry is None or MiniAppsStageTimer is None:
            return
        self.telemetry.emit(
            make_event(
                MiniAppsStageTimer,
                session_id=self.telemetry.session_id,
                backend=self.engine,
                replica=self.name,
                stage=stage,
                iteration=self.iteration,
                duration_ms=duration_ms,
            )
        )

    # ---------------------------------------------------------------- ED6 data handoffs
    def _emit(self, cls, event_time=None, node_id=None, **fields) -> None:
        if self.telemetry is None or cls is None:
            return
        extra = {} if event_time is None else {"event_time": event_time}
        self.telemetry.emit(
            make_event(
                cls,
                session_id=self.telemetry.session_id,
                backend=self.engine,
                node_id=node_id,
                **extra,
                **fields,
            )
        )

    def _task_env_prefix(self, task_key: str) -> str:
        """Pass task identity + telemetry transport to the task process.

        asyncflow splits the command and runs it without a shell, and backends handle
        `env` differently (concurrent replaces the whole environment; Dragon uses
        process templates), so `/usr/bin/env K=V cmd` is the portable way to add
        variables to the inherited environment.
        """
        if self.telemetry is None:
            return ""
        env = task_telemetry.env_for_task(task_key, self.name)
        return "/usr/bin/env " + " ".join(f"{k}={v}" for k, v in env.items()) + " "

    def _ingest(self, task_result) -> list[dict]:
        """Forward a finished task's telemetry to RHAPSODY; return its reads."""
        reads = []
        try:
            for ev in task_telemetry.collect(task_result):
                name = ev.get("event")
                common = dict(
                    event_time=ev["t"],
                    node_id=ev.get("host"),
                    replica=ev.get("replica", self.name),
                    task_key=ev.get("task_key", ""),
                    object_id=ev.get("object_id", ""),
                    version=ev.get("version", ""),
                    payload_bytes=int(ev.get("payload_bytes", 0)),
                    t_start=float(ev.get("t_start", ev["t"])),
                )
                if name == "ddsim.DataWritten":
                    self._emit(DataWritten, **common)
                    self._writes[(ev["object_id"], ev["version"])] = ev
                    self._resolve_pending()
                elif name == "ddsim.DataRead":
                    self._emit(DataRead, **common)
                    reads.append(ev)
                else:
                    self.logger.warning(
                        f"Unknown task telemetry event {name!r} "
                        f"from {ev.get('task_key')}",
                        component=self.name,
                    )
        except Exception as e:  # telemetry must never break the workflow
            self.logger.warning(
                f"Task telemetry ingest failed: {e!r}", component=self.name
            )
        return reads

    def _deliver(
        self,
        exchange,
        consumer,
        read,
        expected_producer,
        representation,
        iteration,
        t_usable=None,
        consumer_host=None,
    ):
        """Record one ED6 handoff; emitted now or once the producer's write is known.

        ED6 (spec A.5.6) = t_usable - t_ready for one delivery. These are *pull-based
        file handoffs*: the delivery is created when the consumer requests the
        data (read start), so t_ready = max(producer write done, read start) and
        t_usable = read done (complete payload loaded in the consumer's
        representation). Time the data sits before the consumer exists (manager
        polling, task launch) is scheduling, which ED6 excludes -- see ED1 (data
        age) / EDec1. For agent_to_manager the manager is already waiting, so
        t_ready = agent write done.
        """
        d = dict(
            exchange=exchange,
            consumer=consumer,
            read=read,
            expected=expected_producer,
            representation=representation,
            iteration=iteration,
            t_usable=read["t"] if t_usable is None else t_usable,
            consumer_host=consumer_host or read.get("host", ""),
        )
        if not self._try_emit_delivery(d):
            self._pending.append(d)

    def _try_emit_delivery(self, d, final=False) -> bool:
        read = d["read"]
        w = self._writes.get((read["object_id"], read["version"]))
        if w is None and not final:
            return False  # producer task not finished/reported yet; retry when it is
        if w is None:
            # no record of who wrote this version: insufficient evidence (spec 5.6)
            status, producer, t_ready, latency, payload, producer_host = (
                "unknown",
                "",
                -1.0,
                -1.0,
                0,
                "",
            )
        else:
            producer = w.get("task_key", "")
            # Shared file name (data_0_<k>.h5 is written by sim k, then training, then
            # the agent): a version written by someone other than the expected producer
            # is flagged, not counted.
            status = (
                "delivered"
                if producer.startswith(d["expected"])
                else "unexpected_producer"
            )
            t_ready = max(float(w["t"]), float(read.get("t_start", read["t"])))
            latency = (d["t_usable"] - t_ready) * 1000.0
            payload = min(
                int(read.get("payload_bytes", 0)), int(w.get("payload_bytes", 0))
            )
            producer_host = w.get("host", "")
        fields = dict(
            replica=self.name,
            iteration=d["iteration"],
            exchange=d["exchange"],
            delivery_id=f"{self.name}.{d['exchange']}.it{d['iteration']}",
            producer=producer,
            consumer=d["consumer"],
            object_id=read["object_id"],
            version=read["version"],
            payload_bytes=payload,
            representation=d["representation"],
            t_ready=t_ready,
            t_usable=d["t_usable"],
            latency_ms=latency,
            status=status,
            producer_host=producer_host,
            consumer_host=d["consumer_host"],
        )
        self._emit(DataDelivery, event_time=d["t_usable"], **fields)
        if self.otel is not None:  # alternate pathway: OTLP endpoint
            self.otel.record_delivery(**fields)
        return True

    def _resolve_pending(self, final=False) -> None:
        self._pending = [
            d for d in self._pending if not self._try_emit_delivery(d, final=final)
        ]

    # --------------------------------------------------------------------------
    def _sim_cmd(self, sim_idx: int) -> str:
        attempt = self._sim_attempts.get(sim_idx, 0)
        self._sim_attempts[sim_idx] = attempt + 1
        # a resubmitted sim is a new execution
        key = f"{self.name}.sim{sim_idx}.a{attempt}"
        self._sim_keys[sim_idx] = key
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {sim_idx} "
            f"--phase {self.phase} "
            f"--num_step {self.num_step} "
        )
        prefix = self._task_env_prefix(key)
        return f"{prefix}{self.sim_executable} {self.src_dir}/simulation.py {args}"

    def _training_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase} "
            f"--num_epochs {self.num_epochs} "
            f"--read_size {self.train_read_size}"
        )
        key = f"{self.name}.train.it{self.iteration}"
        prefix = self._task_env_prefix(key)
        return f"{prefix}{self.train_executable} {self.src_dir}/training.py {args}"

    def _prediction_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase} "
            f"--num_epochs {self.num_epochs} "
            f"--num_mult_outlier 1 "
            f"--num_mult {self.num_mult} "
            f"--output_file {self.prediction_file} "
            f"--read_size {self.agent_read_size}"
        )
        key = f"{self.name}.agent.it{self.iteration}"
        prefix = self._task_env_prefix(key)
        return f"{prefix}{self.predict_executable} {self.src_dir}/agent.py {args}"

    def _selection_cmd(self) -> str:
        args = (
            f"--data_root_dir {self.sim_output_dir} "
            f"--instance_index {self.iteration} "
            f"--phase {self.phase}"
        )
        return f"{self.selection_executable} {self.src_dir}/selection.py {args}"

    # --------------------------------------------------------------------------
    def register_tasks(self):
        # Build engine-specific task descriptions for R1 telemetry.
        # Dragon uses process_templates (ranks + cores/gpus per rank);
        # concurrent uses direct cpus/gpus keys.
        if self.engine == "dragon":
            _sim_desc   = {"process_templates": [[1, {}]], "cores_per_rank": 1,                    "gpus_per_rank": 0}
            _train_desc = {"process_templates": [[1, {}]], "cores_per_rank": self.training_cores,  "gpus_per_rank": 1}
            _pred_desc  = {"process_templates": [[1, {}]], "cores_per_rank": 1,                    "gpus_per_rank": 0}
        else:
            _sim_desc   = {"cpus": 1}
            _train_desc = {"cpus": self.training_cores, "gpus": 1}
            _pred_desc  = {"cpus": 1}

        # ── Simulation ────────────────────────────────────────────────────────
        @self.flow.executable_task(capture_stdio=True)
        async def simulation(task_description=_sim_desc, **kwargs):
            sim_idx = kwargs["sim_inputs"]["sim_idx"]
            return self._sim_cmd(sim_idx)

        self.simulation = simulation

        # ── Training ──────────────────────────────────────────────────────────
        @self.flow.executable_task(capture_stdio=True)
        async def training(task_description=_train_desc):
            return self._training_cmd()

        self.training = training

        # ── Prediction ────────────────────────────────────────────────────────
        @self.flow.executable_task(capture_stdio=True)
        async def prediction(task_description=_pred_desc):
            return self._prediction_cmd()

        self.prediction = prediction

        # ── Selection ─────────────────────────────────────────────────────────
        @self.flow.function_task
        async def selection(*args, **kwargs):
            return self._selection_cmd()

        self.selection = selection

    # --------------------------------------------------------------------------
    async def evaluate_simulations(self):
        t = time.time()
        out = await self.prediction()
        self._emit_stage("prediction", (time.time() - t) * 1000)
        it = self.iteration
        for read in self._ingest(out):
            # handoff 2: training's output (same data_0_<k>.h5 name) -> agent
            self._deliver(
                "train_to_agent",
                f"{self.name}.agent.it{it}",
                read,
                f"{self.name}.train.it{it}",
                _REPR_H5,
                it,
            )
        version = task_telemetry.file_version(self.prediction_file)
        with open(self.prediction_file) as f:
            predictions = yaml.safe_load(f)
        t_usable = time.time()
        self.sim_predictions = predictions
        # handoff 3: agent's predictions.yaml -> manager. The manager is already
        # waiting, so the delivery is created when it requested the prediction (t);
        # t_ready = max(t, agent write done) = agent write done. Usable = parsed into a
        # dict.
        manager_read = {
            "object_id": os.path.abspath(str(self.prediction_file)),
            "version": version,
            "payload_bytes": os.path.getsize(self.prediction_file),
            "t_start": t,
            "t": t_usable,
        }
        self._deliver(
            "agent_to_manager",
            f"{self.name}.manager",
            manager_read,
            f"{self.name}.agent.it{it}",
            _REPR_YAML,
            it,
            t_usable=t_usable,
            consumer_host=socket.gethostname(),
        )

    def _on_sim_done(self, task, sim_idx) -> None:
        # Collect the sim's reported writes (producer side of handoff 1) before the base
        # logic.
        if not task.cancelled() and task.exception() is None:
            self._ingest(task.result())
        super()._on_sim_done(task, sim_idx)

    def stop_simulation(self, *args, **kwargs):
        if random.random() < 0.5:
            return False
        return True

    async def init_sim_queue(self):
        for sim_idx, s in enumerate(range(self.total_num_sim)):
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.sim_inputs[sim_idx] = s

    async def add_sims_to_queue(self, resubmitted_sims):
        for sim_idx in resubmitted_sims:
            await self.sim_task_queue.put({"sim_idx": sim_idx})
            self.logger.info(
                f"Re-added Sim {sim_idx} back the queue", component=self.name
            )
            if sim_idx not in self.sim_inputs:
                raise ValueError(f"Unable to add sim {sim_idx} to queue")

    async def check_train_status(self):
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            ranks = comm.Get_size()
        except Exception:
            ranks = 1

        root_path = Path(self.sim_output_dir, "phase0")
        next_iter = self.iteration + 1
        filenames = [
            Path(root_path, f"data_{rank}_{next_iter}.h5") for rank in range(ranks)
        ]

        self.logger.info(
            f"Waiting for {len(filenames)} file to start training...",
            component=self.name,
        )
        start_trainig = False
        while True:
            if start_trainig:
                break
            start_trainig = True
            for filename in filenames:
                if not filename.exists():
                    if self.debug:
                        self.logger.info(
                            f"File {filename} not found yet, wait...",
                            component=self.name,
                        )
                    start_trainig = False
                    await asyncio.sleep(5)
                    break

        self.logger.info(
            "All required files are available. Starting training...",
            component=self.name,
        )
        return True

    async def train_model(self):
        self.iteration += 1
        if self.debug:
            self.logger.info(
                f"\nTraining Iteration {self.iteration}", component=self.name
            )
            self.logger.info(
                f"{len(self.registered_sims)} simulation(s) running....",
                component=self.name,
            )
        t = time.time()
        out = await self.training()
        self._emit_stage("training", (time.time() - t) * 1000)
        it = self.iteration
        for read in self._ingest(out):
            # handoff 1: sim <it>'s output data_0_<it>.h5 -> training iteration <it>
            self._deliver(
                "sim_to_train",
                f"{self.name}.train.it{it}",
                read,
                f"{self.name}.sim{it}.",
                _REPR_H5,
                it,
            )
        if self.debug:
            self.logger.task_completed("Training Completed", component=self.name)

    async def _signal_ready(self) -> None:
        if self._on_ready is not None:
            result = self._on_ready()
            if asyncio.iscoroutine(result):
                await result

    async def finalize_results(self):
        n_done = len(self.completed_sims)
        if not self._data_ready_signaled and n_done >= self.miniapps_data_ready:
            self._data_ready_signaled = True
            if self.debug:
                self.logger.info(
                    f"{n_done} sims completed"
                    " — signaling ready for downstream workflows",
                    component=self.name,
                )
            await self._signal_ready()
        if n_done >= self.total_num_sim:
            self.shutting_down.set()
            self.run_workflow = False
            self.logger.info("All sim have completed...", component=self.name)
        else:
            if self.debug:
                self.logger.info(
                    f"sim {n_done} out of {self.total_num_sim} have completed...",
                    component=self.name,
                )

    async def post_process_sim(self, sim_idx):
        pass

    async def close(self):
        # Deliveries whose producer write was never reported (e.g. producer
        # cancelled/killed before its stdout was collected) are emitted with status
        # "unknown".
        self._resolve_pending(final=True)
        # Temporary cleanup to test campaign manager and aid Disk quota exceeded.
        if self.home_dir.exists():
            shutil.rmtree(self.home_dir, ignore_errors=True)
            self.logger.info(
                f"Removed home directory: {self.home_dir}", component=self.name
            )

    async def stop(self):
        await self.close()

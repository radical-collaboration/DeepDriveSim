"""Agentic MiniApps workflow: the DeepDriveSim MiniApp with LLM steering.

The same workflow as :class:`MiniAppsWorkflow` (same simulations, training, batch sizes
and steering loop), except for one step: deciding what to do next. The stock workflow
runs ``agent.py`` (the stand-in for DeepDriveMD's outlier detection) and reads its
``predictions.yaml``. Here the manager instead collects the simulation data, sends it to
an LLM as one text prompt, and turns the answer into the per-simulation scores the
steering loop already uses.

In the spirit of a mini-app, the prompt and answer only need to be representative: the
prompt is padded with numeric rows to a configurable size, and the answer's length is
fixed by ``llm.max_tokens`` (with ``llm.ignore_eos``). The LLM request is an AsyncFlow
``prompt_task`` routed to the backend named ``llm`` (see ``llm_backends.py``).

Telemetry (custom ``ddsim.*`` events, manager side, each metric recorded independently):

* ``ddsim.MiniAppsStageTimer`` stage ``llm_steering``: the steering step's duration
  (counterpart of the stock ``prediction`` stage).
* ``ddsim.LLMRequest``: one per request (EQ3 latency, EQ4 status, prompt/answer size,
  AsyncFlow task uid).
* ``ddsim.LLMRequestSummary``: per replica at close (EQ1 count, EQ2 throughput).
* ``ddsim.InferenceService``: service start-up (EC5) and lifetime (EC2), emitted by
  ``run_agentic_workflow.py``.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

from workflows.miniapps_workflow.miniapps_workflow import MiniAppsWorkflow, make_event

try:
    from rhapsody.telemetry import define_event

    LLMRequest = define_event(
        "ddsim.LLMRequest",
        replica=str,
        iteration=int,
        request_id=str,
        task_uid=str,  # AsyncFlow task uid of the prompt_task
        llm_backend=str,
        t_submit=float,
        t_response=float,
        latency_ms=float,  # EQ3: request submitted -> answer available
        status=str,  # EQ4: completed | failed
        error=str,
        prompt_chars=int,
        prompt_tokens_est=int,
        response_chars=int,
        sims_in_prompt=int,
        sims_scored=int,
        stop_decisions=int,
    )
    LLMRequestSummary = define_event(
        "ddsim.LLMRequestSummary",
        replica=str,
        requests=int,  # EQ1
        completed=int,
        failed=int,
        window_s=float,  # first submit -> last answer
        throughput_rps=float,  # EQ2: completed / window_s
        mean_latency_ms=float,
    )
    InferenceService = define_event(
        "ddsim.InferenceService",
        service=str,
        llm_backend=str,
        model=str,
        num_gpus=int,
        phase=str,  # ready | stopped
        t_init_start=float,
        t_ready=float,
        t_stopped=float,
        startup_ms=float,  # EC5: init start -> ready to serve
        lifetime_ms=float,  # EC2: ready -> stopped (phase=stopped)
    )
except ImportError:  # pragma: no cover
    LLMRequest = LLMRequestSummary = InferenceService = None

CHARS_PER_TOKEN = 4  # rough English/numeric estimate, used only to size the prompt
_SCORE_RE = re.compile(r"sim_(\d+)\s*[:=]\s*([0-9]*\.?[0-9]+)")


def emit(telemetry, cls, backend: str, event_time: float | None = None, **fields):
    """Emit a custom ddsim.* event through RHAPSODY telemetry (no-op without it)."""
    if telemetry is None or cls is None:
        return
    extra = {} if event_time is None else {"event_time": event_time}
    telemetry.emit(
        make_event(
            cls, session_id=telemetry.session_id, backend=backend, **extra, **fields
        )
    )


class AgenticMiniAppsWorkflow(MiniAppsWorkflow):
    """MiniApps workflow whose steering step is an LLM prompt_task."""

    def __init__(self, config: dict = None, **kwargs):
        cfg = config or {}
        agentic = cfg.get("agentic", {}) or {}
        # set before super().__init__(), which calls register_tasks()
        self.llm_backend_name = kwargs.get("llm_backend", "llm")
        # the execution backend actually in use (the stock stage timer labels events
        # with cfg["engine"] instead, see DeepDriveSim issue #24)
        self.backend_label = kwargs.get(
            "backend_label", cfg.get("engine", "concurrent")
        )
        self.llm_backend_kind = (cfg.get("llm", {}) or {}).get("backend", "vllm")
        self.prompt_tokens = int(agentic.get("prompt_tokens", 2048))
        self.stop_threshold = float(agentic.get("stop_threshold", 0.5))
        self.decision = agentic.get("decision", "llm")  # llm | coinflip (stock rule)
        self._rng = random.Random(int(agentic.get("seed", 0)))
        self._requests: list[dict] = []
        super().__init__(config=config, **kwargs)

    # ------------------------------------------------------------------ tasks
    def register_tasks(self):
        super().register_tasks()

        @self.flow.prompt_task(backend=self.llm_backend_name)
        async def llm_steering(prompt: str):
            return prompt

        self.llm_steering = llm_steering

    # ------------------------------------------------------------------ prompt
    def _collect_data(self) -> list[dict]:
        """What the steering step knows: finished sims' outputs + running sims."""
        root = Path(self.sim_output_dir, f"phase{self.phase}")
        now = time.time()
        rows = {}
        for f in sorted(root.glob("data_0_*.h5")):
            try:
                k = int(f.stem.split("_")[-1])
                st = f.stat()
            except (ValueError, OSError):
                continue
            rows[k] = {
                "sim": k,
                "state": "finished",
                "bytes": st.st_size,
                "age_s": round(now - st.st_mtime, 1),
            }
        for k in self.registered_sims:
            rows.setdefault(
                int(k), {"sim": int(k), "state": "running", "bytes": 0, "age_s": 0.0}
            )
        return [rows[k] for k in sorted(rows)]

    def _build_prompt(self, data: list[dict]) -> str:
        head = (
            "You steer an ensemble of molecular-dynamics simulations. For every "
            "simulation listed below, rate how promising it is to keep exploring, "
            "from 0 (stop it) to 1 (keep it). Answer with one line per simulation, "
            "exactly 'sim_<id>: <score>'.\n\nSimulations:\n"
        )
        lines = [
            f"sim_{d['sim']}: state={d['state']} output_bytes={d['bytes']} "
            f"age_s={d['age_s']}"
            for d in data
        ]
        text = head + "\n".join(lines) + "\n\nObservations (collective variables):\n"
        # pad with numeric rows to the configured size (content is representative only)
        target = self.prompt_tokens * CHARS_PER_TOKEN
        obs = []
        sims = [d["sim"] for d in data] or [0]
        while len(text) + sum(len(o) + 1 for o in obs) < target:
            k = sims[len(obs) % len(sims)]
            vals = " ".join(f"{self._rng.uniform(-5, 5):.4f}" for _ in range(8))
            obs.append(f"sim_{k} t={len(obs)} cv=[{vals}]")
        return text + "\n".join(obs) + "\n"

    @staticmethod
    def _parse_scores(response: str, sims: list[int]) -> dict:
        wanted = set(sims)
        scores = {}
        for k, v in _SCORE_RE.findall(response or ""):
            k = int(k)
            if k in wanted and k not in scores:
                scores[k] = min(max(float(v), 0.0), 1.0)
        return scores

    # ------------------------------------------------------------------ steering
    async def evaluate_simulations(self):
        """LLM steering: replaces the stock agent.py prediction task."""
        it = self.iteration
        data = self._collect_data()
        prompt = self._build_prompt(data)
        t_submit = time.time()
        fut = self.llm_steering(prompt)
        task_uid = str(getattr(fut, "uid", "") or "")
        response, status, error = "", "completed", ""
        try:
            response = await fut
            response = response if isinstance(response, str) else str(response)
        except Exception as e:  # a failed request leaves the steering decision empty
            status, error = "failed", f"{type(e).__name__}: {e}"[:500]
        t_response = time.time()
        self._emit_stage("llm_steering", (t_response - t_submit) * 1000)

        sims = [d["sim"] for d in data]
        scores = self._parse_scores(response, sims)
        self.sim_predictions = scores
        stops = sum(
            1
            for k, s in scores.items()
            if k in self.registered_sims and self.stop_simulation(prediction=s)
        )
        req = dict(
            replica=self.name,
            iteration=it,
            request_id=f"{self.name}.llm.it{it}.{len(self._requests)}",
            task_uid=task_uid,
            llm_backend=self.llm_backend_kind,
            t_submit=t_submit,
            t_response=t_response,
            latency_ms=(t_response - t_submit) * 1000,
            status=status,
            error=error,
            prompt_chars=len(prompt),
            prompt_tokens_est=len(prompt) // CHARS_PER_TOKEN,
            response_chars=len(response),
            sims_in_prompt=len(sims),
            sims_scored=len(scores),
            stop_decisions=stops,
        )
        self._requests.append(req)
        emit(
            self.telemetry, LLMRequest, self.backend_label, event_time=t_response, **req
        )

    def stop_simulation(self, *args, prediction=None, **kwargs):
        if self.decision == "coinflip":  # the stock rule, for comparisons
            return super().stop_simulation(*args, prediction=prediction, **kwargs)
        return prediction is not None and prediction < self.stop_threshold

    # ------------------------------------------------------------------ shutdown
    def _emit_summary(self):
        if not self._requests or getattr(self, "_summary_done", False):
            return
        self._summary_done = True
        done = [r for r in self._requests if r["status"] == "completed"]
        window = max(r["t_response"] for r in self._requests) - min(
            r["t_submit"] for r in self._requests
        )
        emit(
            self.telemetry,
            LLMRequestSummary,
            self.backend_label,
            replica=self.name,
            requests=len(self._requests),
            completed=len(done),
            failed=len(self._requests) - len(done),
            window_s=window,
            throughput_rps=(len(done) / window) if window > 0 else 0.0,
            mean_latency_ms=(sum(r["latency_ms"] for r in done) / len(done))
            if done
            else -1.0,
        )

    async def close(self):
        self._emit_summary()
        await super().close()

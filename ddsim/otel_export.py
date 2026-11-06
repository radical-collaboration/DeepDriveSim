"""Optional OpenTelemetry (OTLP/HTTP) export for ddsim telemetry.

An *alternate pathway* next to RHAPSODY's JSONL checkpoint file: when an OTLP endpoint
is configured, metrics are also posted to it. Nothing changes when none is configured.

Endpoint resolution (first match wins):
  1. ``otel_endpoint`` in the workflow config, e.g. ``http://collector:4318``
  2. the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable
The endpoint is the OTLP/HTTP *base* URL; ``/v1/traces`` and ``/v1/metrics`` are
appended.

What is exported
----------------
* RHAPSODY's own telemetry (task lifecycle spans, task/resource metrics), by passing
  OTLP span processors / metric readers through ``start_telemetry()``
  (:func:`rhapsody_hooks`).
* ddsim metrics recorded via :class:`OtelExporter`. RHAPSODY's OTel pipeline only turns
  its built-in task events into spans/metrics; custom ``ddsim.*`` events reach the
  JSONL file but not OTel, so they are recorded here explicitly.

Export is not real time: spans are batched and metrics exported periodically and at
shutdown. Timestamps are the recorded event times, not export times.
"""

from __future__ import annotations

import os

ENV_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"


def resolve_endpoint(configured: str | None = None) -> str | None:
    """Return the OTLP/HTTP base endpoint to use, or None if export is disabled."""
    endpoint = (configured or os.environ.get(ENV_ENDPOINT) or "").strip()
    return endpoint.rstrip("/") or None


def warn_if_unreachable(endpoint: str | None, timeout_s: float = 2.0) -> bool:
    """Print a warning if nothing accepts connections at ``endpoint``.

    Export stays enabled (a collector may come up later), but with no collector the
    OTel SDK retries each export with backoff, which adds time at shutdown.
    """
    if not endpoint:
        return True
    import socket
    from urllib.parse import urlparse

    u = urlparse(endpoint)
    port = u.port or (443 if u.scheme == "https" else 4318)
    try:
        with socket.create_connection((u.hostname or "localhost", port), timeout_s):
            return True
    except OSError as e:
        print(
            f"[ddsim.otel_export] WARNING: OTLP endpoint {endpoint} is not reachable "
            f"({e}); exports will fail and be retried (slower shutdown)."
        )
        return False


def _exporters(endpoint: str):
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return (
        OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"),
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
    )


def rhapsody_hooks(endpoint: str | None, export_interval_ms: int = 5000) -> dict:
    """kwargs for ``WorkflowEngine.start_telemetry()`` that also send RHAPSODY's own
    telemetry to ``endpoint``. Returns {} if disabled or the exporter is unavailable."""
    if not endpoint:
        return {}
    try:
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        span_exporter, metric_exporter = _exporters(endpoint)
    except ImportError as e:
        print(f"[ddsim.otel_export] OTLP export disabled (missing package): {e}")
        return {}
    return {
        "span_processors": [BatchSpanProcessor(span_exporter)],
        "metric_readers": [
            PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=export_interval_ms
            )
        ],
    }


class OtelExporter:
    """Posts ddsim metrics to an OTLP/HTTP endpoint.

    Uses its own Tracer/MeterProvider (RHAPSODY's are private), tagged with the RHAPSODY
    ``session_id`` so both streams can be joined. Methods never raise.
    """

    def __init__(
        self,
        endpoint: str,
        service_name: str = "ddsim",
        resource_attributes: dict | None = None,
        export_interval_ms: int = 5000,
    ):
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        self.endpoint = endpoint
        resource = Resource.create(
            {"service.name": service_name, **(resource_attributes or {})}
        )
        span_exporter, metric_exporter = _exporters(endpoint)
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    metric_exporter, export_interval_millis=export_interval_ms
                )
            ],
        )
        self._tracer = self._tracer_provider.get_tracer("ddsim")
        meter = self._meter_provider.get_meter("ddsim")
        self._delivery_latency = meter.create_histogram(
            "ddsim.data_delivery.latency",
            unit="ms",
            description="ED6 producer-to-consumer latency per data handoff",
        )
        self._delivery_count = meter.create_counter(
            "ddsim.data_delivery.count",
            description="Data handoffs (ED6 deliveries) by exchange and status",
        )

    @classmethod
    def create(cls, endpoint: str | None, **kwargs) -> OtelExporter | None:
        """Exporter for ``endpoint``, or None if disabled/unavailable (never raises)."""
        if not endpoint:
            return None
        try:
            return cls(endpoint, **kwargs)
        except Exception as e:
            print(f"[ddsim.otel_export] OTLP export disabled: {e!r}")
            return None

    def record_delivery(self, **d) -> None:
        """Record one ED6 delivery: a span from t_ready to t_usable + metrics."""
        try:
            attrs = {
                f"ddsim.{k}": v
                for k, v in d.items()
                if isinstance(v, (str, bool, int, float)) and v is not None
            }
            known = d["status"] != "unknown"
            # The span carries the recorded times (not the export time). Unknown
            # deliveries have no t_ready: zero-length span at t_usable.
            start = d["t_ready"] if known else d["t_usable"]
            span = self._tracer.start_span(
                "ddsim.DataDelivery",
                start_time=int(start * 1e9),
                attributes=attrs,
            )
            span.end(end_time=int(d["t_usable"] * 1e9))
            labels = {
                "exchange": d["exchange"],
                "status": d["status"],
                "replica": d.get("replica", ""),
            }
            self._delivery_count.add(1, labels)
            if known:
                self._delivery_latency.record(d["latency_ms"], labels)
        except Exception as e:  # telemetry must never break the workflow
            print(f"[ddsim.otel_export] dropped delivery: {e!r}")

    def shutdown(self) -> None:
        """Flush pending spans/metrics and stop the exporters."""
        for provider in (self._tracer_provider, self._meter_provider):
            try:
                provider.shutdown()
            except Exception as e:
                print(f"[ddsim.otel_export] shutdown: {e!r}")

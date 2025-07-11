# Monitoring Setup

This project exposes metrics via Prometheus using the OpenTelemetry SDK.
Grafana can be used to visualize these metrics with the provided dashboard.

## Import the Dashboard

1. Start Grafana and ensure the Prometheus data source is configured.
2. Navigate to **Dashboards → Import**.
3. Upload `monitoring/dashboards/cort_dashboard.json` or paste its contents.
4. Select your Prometheus data source if prompted and click **Import**.

The dashboard displays request latency, convergence details and other
metrics from `metrics_v2.py`.


## Tracing Requests

Every call to the thinking engine is tagged with a short `request_id`.
This ID appears in all logs produced by the controller, providers and
strategies. To trace a request across components, search your logs for the
generated ID:

```json
{"event": "loop_start", "request_id": "a1b2c3d4", "prompt": "..."}
...
{"event": "llm_request_success", "request_id": "a1b2c3d4"}
```

Use this identifier to correlate metrics and log messages for a single
session or API call.

## Enabling Tracing

Tracing is initialized in `recthink_web_v2.py` using functions from
`monitoring/telemetry.py`. Call `initialize_telemetry` and then
`instrument_fastapi(app)` to attach tracing middleware. Ensure the
`opentelemetry-instrumentation-fastapi` package is installed.

## Environment Variables

Set the following variables to enable trace and metric exporters:

- `JAEGER_ENDPOINT` – host and port of the Jaeger agent. When provided,
  traces will be sent to Jaeger.
- `PROMETHEUS_PORT` – port to expose the Prometheus metrics endpoint.
  Metrics are served at `http://localhost:${PROMETHEUS_PORT}/metrics`.


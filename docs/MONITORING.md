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

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

## Enabling Tracing

Tracing is initialized in `recthink_web_v2.py` using functions from
`monitoring/telemetry.py`. Call `initialize_telemetry` and then
`instrument_fastapi(app)` to attach tracing middleware. Ensure the
`opentelemetry-instrumentation-fastapi` package is installed.

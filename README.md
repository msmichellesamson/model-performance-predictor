# model-performance-predictor

> Can we predict LLM quality degradation from inference-side signals alone, before users notice?

## The question I'm exploring

Detecting that a deployed model has gotten worse is genuinely hard. You usually
don't have ground-truth labels at serving time — you only get the user's
implicit feedback (regenerate, abandon, thumbs-down) days later. By then, your
top-of-funnel metrics have already moved.

The Anthropic Claude system cards repeatedly note that model behaviour can
shift across deployments, fine-tunes, and prompt changes. OpenAI's GPT-4 system
card discusses similar drift around tool use and refusals. Both labs invest
heavily in evals — but evals run periodically, not continuously, and they
don't tell you about drift between eval runs.

So my question: **using only signals you can compute from the inference
request/response itself** (no labels, no human feedback), how well can you
predict that quality is degrading?

## Why I care

This is the observability problem for LLMs that classical APM tools don't
solve. Latency, throughput, error rate — those are necessary but not
sufficient. A model that's fast, available, and quietly hallucinating is the
worst possible failure mode and the hardest to alert on.

I also care about **what counts as a feature** here. Token-level entropy,
output length distribution shifts, refusal rate drift, repetition statistics —
these are all weak proxies for quality. Combining them well is an interesting
mini ML problem in its own right.

## What's in here

- `src/core/predictor.py` — degradation predictor that takes streaming
  inference metrics and outputs a degradation probability
- `src/core/drift_detector.py` — distributional drift on response-level
  features (length, entropy proxies, refusal patterns)
- `src/core/metrics_collector.py` — wraps inference calls to extract
  features without modifying the model
- `src/api/` — FastAPI service that exposes predictions + Prometheus metrics
- `src/alerts/` — alerting integrations gated by predicted severity

Plus the production wrapping: Dockerfile, docker-compose for local dev,
terraform for GCP, k8s manifests, Prometheus rules + Grafana dashboards.

## What I'm finding (so far)

I don't have a real production trace yet, so I've been working with
synthetic degradations injected into a baseline. From that:

- Output length is a deceptively strong signal. When model quality
  degrades, response length distributions shift before user-visible
  metrics move. But the direction of the shift depends on the failure
  mode (refusal-heavy degradation → shorter; repetition-loop degradation
  → longer), which means a single threshold doesn't work.
- Refusal rate is the cleanest signal for safety/policy drift but useless
  for quality drift on benign queries.
- The predictor as currently structured is **overfit to the kinds of
  degradations I synthesised**. I'm essentially measuring my own injection
  patterns.
- Calibration matters more than accuracy here. A miscalibrated predictor
  that fires too often will get muted; one that fires too rarely is
  worse than nothing.

## What I'd do next

- Move from synthetic injections to a real before/after dataset (one
  obvious source: behaviour changes around documented model version
  bumps for any major API)
- Add a small **judge model** as a secondary signal — not as ground truth
  but as a noisy label generator to validate the unlabelled features against
- Treat this as a calibration problem, not a classification problem.
  Report ECE, not just AUC.
- Specifically replicate the kinds of drift discussed in the Anthropic
  and OpenAI system cards rather than generic ones

## Status

Service runs end-to-end. The predictor is a placeholder model; the
features are real but the validation isn't honest yet.

## References

- Anthropic, *Claude 3 Model Card* and ongoing system cards (2024)
- OpenAI, *GPT-4 System Card* (2023)
- Liang et al., *Holistic Evaluation of Language Models (HELM)* (2022)
- Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017)

# Metrics

## RMSE

Root mean squared error across all predicted dimensions:

```
rmse = sqrt(mean((pred - target)^2))
```

## VRMSE

Variance-normalized RMSE:

```
vrmse = sqrt(mean((pred - target)^2) / (var(target) + eps))
```

## Windowed VRMSE

Windowed VRMSE evaluates specific time windows, matching the paper's reported ranges:

- `vrmse_6_12`
- `vrmse_13_30`

These metrics are only computed when `output_steps` covers the requested window.

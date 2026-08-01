# Cross-benchmark message coverage

This study compares confirmed natural WildDelusion user messages with released
user turns from Spiral-Bench, Psychosis-Bench, and SIM-VAIL on four frozen
axes. The analysis plan was committed before corpus construction or coding.

Lost in Delusion is preregistered, but its reported synthetic user turns were
not present in an official turn-level release found on 2026-07-31. The pipeline
records this as unavailable rather than reconstructing them.

```bash
python coverage.py build
python coverage.py code --model gpt-5.4-mini
python coverage.py code --audit --model gpt-5.4-nano
python coverage.py analyze
```

The `work/` directory is gitignored because it contains sensitive message text
and the provenance key. Aggregate outputs can be copied to `results/` after the
run.

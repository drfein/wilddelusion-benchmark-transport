# Cross-benchmark message coverage

This composition-only study compares 522 validated WildDelusion target messages
with 14 Spiral-Bench starters, 192 Psychosis-Bench user turns, and 6,330
SIM-VAIL phenotype-generated user turns on four frozen axes. The corrected
analysis plan was committed before corrected-corpus coding or analysis.

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
and the provenance key. It also contains a 300-item blind human-audit packet;
automated-coder agreement is not a substitute for that pending human audit.

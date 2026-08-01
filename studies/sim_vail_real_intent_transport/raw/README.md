# Raw evidence

These deterministic gzip archives contain the complete final evidence used by
the SIM-VAIL real-context transport analysis:

- selected real WildDelusion conversation contexts and screening decisions;
- all target/control intervention arms;
- all 960 final Qwen3 responses with generation provenance;
- every raw GPT-5.2 judge response and parsed score;
- all three disjoint judge-calibration sets, including the two failed compact
  prompt attempts; and
- exact calibration and API manifests.

`manifest.json` records SHA-256 digests for both each compressed archive and
its uncompressed JSONL bytes. Verify without writing decompressed data:

```bash
python studies/sim_vail_real_intent_transport/reproduce.py verify
```

Materialized files are written under the ignored `.repro/` directory. The
archives contain sensitive public conversation text and inherit the source
dataset licenses described in the repository's `DATA_LICENSE.md`.

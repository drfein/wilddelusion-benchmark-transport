# DelusionScore construct transport to real conversations

This study stress-tests the central measurement construct in Shimgekar et al.,
"AI Psychosis: Does Conversational AI Amplify Delusion-Related Language?"
(arXiv:2603.19574) on WildDelusion conversations.

This is **not an exact classifier replication**. The paper reports a regularized
logistic regression over 384-dimensional `all-MiniLM-L6-v2` embeddings, but does
not release the fitted model or its 3,000 Reddit training posts. We reproduce that
model class and explicitly test whether its conclusions survive harder, real-world
contrasts.

## Frozen design

- Unit of split: complete source conversation; no conversation enters both train
  and test.
- Primary language: English, matching the paper's Reddit setting.
- Classifier: `all-MiniLM-L6-v2`, default mean pooling, unnormalized embeddings,
  L2 logistic regression with `C=1`.
- Endpoint test: real delusion endpoint versus a minimally edited grounded
  counterfactual; the control side is synthetic and is never called real data.
- External specificity: fully natural role-play, fiction, jokes, quoted beliefs,
  and text-transformation near misses.
- Longitudinal test: out-of-fold scores on every original user turn through the
  flagged endpoint.
- Artifact check: repeat the trajectory analysis without the selected endpoint.
- Inference: conversation-cluster bootstrap and conversation-cluster-robust OLS.

No paid API is used.

The main findings and their limitations are in [REPORT.md](REPORT.md). Public
score tables omit message text; `export_public.py` creates those tables from a
full local run.

## Run

```bash
python prepare_data.py \
  --paired-inputs /path/to/grounded_model_inputs.jsonl \
  --natural-controls /path/to/high_confidence_strict_controls.jsonl \
  --broad-controls /path/to/all_contextual_rejections.jsonl \
  --output-dir work/inputs

python run_analysis.py \
  --input-dir work/inputs \
  --output-dir work/results
```

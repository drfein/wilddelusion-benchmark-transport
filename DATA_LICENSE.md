# Data and upstream licenses

The MIT license applies to code authored for this repository. It does not
relicense datasets, model weights, paper supplements, or third-party code.

- `danielfein/WildDelusionCombined` combines WildChat (ODC-By) and ShareChat
  (CC BY-NC 4.0). Treat the combined data as non-commercial unless separate
  permission applies.
- Some natural near-miss controls originate in LMSYS-Chat-1M. Its terms do not
  permit this repository to redistribute conversation text. The repository
  therefore includes aggregate results and code, but never those raw rows.
- Psychosis-Bench is fetched from its upstream Git commit rather than vendored.
  Users are responsible for the upstream repository's terms.
- Hugging Face model checkpoints retain their own licenses and access terms.
- `studies/sim_vail_real_intent_transport/raw` redistributes the selected
  WildDelusion conversation text under the same source terms above. Generated
  model responses and study annotations are included solely to reproduce the
  reported research analysis.
- The vendored SIM-VAIL judge prompt is from upstream commit
  `f968ede3cb5a1d8791f7308dc44c35373d64069f` under its MIT license, reproduced
  beside the file in the study's `upstream/` directory.

The conversations may contain sensitive material. Do not use them for
diagnosis, person-level inference, surveillance, moderation, or attempts to
identify or contact source users.

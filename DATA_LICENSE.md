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

The conversations may contain sensitive material. Do not use them for
diagnosis, person-level inference, surveillance, moderation, or attempts to
identify or contact source users.

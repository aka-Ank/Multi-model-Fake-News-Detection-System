# Limitations

- The primal optimizer uses mini-batch subgradients; its objective need not decrease monotonically and its stopping heuristic is not a proof of optimum.
- SMO uses simple working-pair heuristics and an explicit iteration cap. A `max_iter` stop is recorded as incomplete rather than convergence.
- Synthetic and bounded text samples are small. They establish implementation behavior, not deployment performance.
- The Fakeddit experiment reads one local validation TSV and makes a new deterministic demonstration split; it does not replace the project's preserved train/validation/test protocol.
- Fakeddit source taxonomy labels are not independently fact-checked truth labels. NewsCLIPpings mismatch labels were not combined with them.
- Repeated seeds on the same test rows do not create independent test datasets; no unsupported significance claim is made.

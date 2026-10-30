# Mathematical validation

The primal objective is `0.5 ||w||² + C sum max(0, 1-y f(x))`. A central finite-difference check at a point away from the hinge kink gave a maximum absolute gradient error of `1.59e-10`. Unit tests also verify vector operations, kernels, label conversion, objective prediction signs, and TF-IDF train/test isolation.

For the linear dual check, the final dual objective was `0.15999999`; equality residual was `0`, maximum box-constraint violation was `0`, maximum KKT residual was `1.79e-4`, and three support vectors were identified. This is below the solver's `1e-3` KKT tolerance. SMO reports KKT convergence only after the constraint check remains clear for its configured number of passes. Primal's `objective_change_heuristic` is deliberately not represented as a certificate of optimality.

The full per-epoch primal trace and per-pass dual trace are embedded in [the suite raw log](raw/svc-scratch-research-v1.json). No primal-dual gap is reported because the stochastic primal trajectory and the SMO dual run are not matched iterates of one shared optimization path.

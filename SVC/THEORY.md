# Mathematical Foundation

For a feature vector `x` and weights `w`, a linear classifier uses the hyperplane `w.x + b = 0`. Its two half-spaces are predicted by the sign of `f(x) = w.x + b`. A dataset is linearly separable when one such hyperplane gives every signed target `y_i in {-1,+1}` the correct sign.

The functional margin of one point is `y_i f(x_i)`; it scales if `w` and `b` scale. The geometric margin is `y_i f(x_i) / ||w||`, which is scale-invariant. Maximum-margin classification chooses a separating boundary with the largest minimum geometric margin.

## Soft-margin primal

Hard-margin SVM requires `y_i f(x_i) >= 1` for every point, which fails with overlap or noise. Soft margin introduces slack variables `xi_i >= 0` and minimizes `1/2 ||w||² + C sum xi_i`, subject to `y_i f(x_i) >= 1-xi_i`. Eliminating slack gives the hinge-loss objective used here:

`1/2 ||w||² + C sum max(0, 1-y_i(w.x_i+b))`.

`C` controls the trade-off: greater `C` penalizes margin violations more strongly, typically shrinking tolerance for training errors; smaller `C` prioritizes a wider, more regularized boundary. At the hinge kink, margin exactly one, the loss is nondifferentiable. `PrimalLinearSVM` consistently selects the zero loss-side subgradient.

Its subgradient away from the kink is `w - C sum_(violators) y_i x_i` for weights and `-C sum_(violators) y_i` for bias. Mini-batches are rescaled by `n/batch_size`, so their expected loss component matches the stated summed-loss convention. The cost per epoch is O(n d) and model memory O(d).

## Dual and KKT conditions

The dual replaces direct weights with nonnegative Lagrange multipliers:

`max_alpha sum alpha_i - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i,x_j)`

subject to `0 <= alpha_i <= C` and `sum alpha_i y_i = 0`. KKT conditions link multipliers to margins: alpha zero implies a point outside/on the margin, alpha strictly between bounds implies margin one, and alpha at C permits a margin violation. Nonzero-alpha observations are support vectors.

SMO updates two multipliers at a time, preserving the equality constraint. It computes feasible lower/upper bounds, clips the update, and recalculates the bias from non-bound multipliers when possible. Near-zero curvature `eta` is handled by evaluating the two feasible bounds in the dual objective. The implementation reports KKT residual counts rather than claiming convergence merely because an iteration cap was reached.

## Kernels

The kernel trick substitutes `K(x,z)` for a feature-space dot product without explicitly building that feature space. This project implements `K_linear=x.z`, `K_poly=(gamma*x.z+coef0)^degree`, and `K_rbf=exp(-gamma||x-z||²)`. RBF gamma controls locality: high gamma produces narrow local influence, while low gamma produces smoother broad influence. Kernel training stores an O(n²) Gram matrix and can be slow or memory-heavy on large text datasets.

## Relation to LinearSVC

LinearSVC-style methods optimize a linear primal/dual objective without a full kernel matrix and scale more suitably to high-dimensional sparse TF-IDF. A kernel SVC uses the dual representation and is useful for small nonlinear experiments. Their intercept treatment, normalization, stopping thresholds, and regularization scaling can differ, so exact agreement with other implementations is neither expected nor used here.

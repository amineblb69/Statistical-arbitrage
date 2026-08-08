"""
Sanity check: reproduce Section 3.3.1 of the paper.

We generate OU paths, train the signature stopping policy to MAXIMIZE
E[X_τ] (payoff Y_t = X_t, no discount), and verify that the resulting
average stopped value beats the long-term mean θ = 10.

Paper reports ~10.23 with (mu=10, theta=10, sigma=1), N=100 train, 10 test.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from strategies.signature_sot import SignatureOptimalStopping


def simulate_ou(x0, mu, theta, sigma, T, n_steps, n_paths, rng):
    """Euler-Maruyama:  dX = mu*(theta - X) dt + sigma dW."""
    dt = T / n_steps
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for j in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        X[:, j + 1] = X[:, j] + mu * (theta - X[:, j]) * dt + sigma * dW
    return X


def main():
    rng = np.random.default_rng(42)

    # Paper's parameters
    mu, theta, sigma = 10.0, 10.0, 1.0
    T, n_steps       = 1.0, 100
    x0               = 10.0     # start at the long-term mean

    # ---- Training data: 100 OU paths ----
    X_train = simulate_ou(x0, mu, theta, sigma, T, n_steps, 100, rng)
    Y_train = X_train                   # payoff = the path itself (maximise X_τ)

    # ---- Fit the linear stopping policy ----
    sot = SignatureOptimalStopping(N=3, k=1.0, mu=20.0, T=T)
    sot.fit(X_train, Y_train, verbose=False)
    print(f"trained, l* length = {len(sot.l_star)}, "
          f"final loss = {sot._last_result.fun:.4f}")

    # ---- Testing data: 10 fresh OU paths ----
    X_test = simulate_ou(x0, mu, theta, sigma, T, n_steps, 10, rng)

    stopped_values = []
    for m in range(10):
        j = sot.stopping_time(X_test[m])
        stopped_values.append(X_test[m, j])
        print(f"  path {m}: stop at step {j:3d}, X_τ = {X_test[m, j]:.3f}, "
              f"path max = {X_test[m].max():.3f}")

    print(f"\naverage stopped value = {np.mean(stopped_values):.3f}   "
          f"(paper reports ≈ 10.23; long-term mean = 10.0)")


if __name__ == "__main__":
    main()
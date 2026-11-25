"""Conjugate Bayesian updating of the next-day return distribution per market type."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def posterior_update(n, xbar, sd, k, mu0, p0, n0=250):
    """Posterior for one market type after n observed days.

    Prior: mu ~ N(mu0, sd^2 / n0), i.e. n0 pseudo-observations at the overall mean.
    Returns the posterior mean, its standard deviation, P(mu > 0) and the
    Beta-Binomial posterior probability that a day rises.
    """
    mu_hat = (n * xbar + n0 * mu0) / (n + n0)
    se = sd / np.sqrt(n + n0)
    return {"mu_hat": mu_hat, "se": se, "p_mu_pos": float(norm.cdf(mu_hat / se)), "p_up": (k + n0 * p0) / (n + n0)}


def sequential_posterior(types, n0=250, burn_in=100, min_cell=30):
    """Posterior for each day's market type, using only outcomes already observed.

    `types` is the output of regime.market_types. At day t the known outcomes are
    the next-day returns of all days before t (the return of day t itself, which
    is the outcome of day t-1, has just been observed at the close).
    Returns a frame with mu_hat, p_mu_pos, p_up and the half-Kelly weight w.
    """
    labels = types["type"].to_numpy()
    outcome = types["r_next"].to_numpy()
    n_days = len(types)
    count, total, squares, rises = {}, {}, {}, {}
    all_n = all_sum = all_sq = all_k = 0.0
    out = np.full((n_days, 4), np.nan)
    for t in range(n_days):
        if t > 0 and not np.isnan(outcome[t - 1]):
            s, x = labels[t - 1], outcome[t - 1]
            count[s] = count.get(s, 0) + 1
            total[s] = total.get(s, 0.0) + x
            squares[s] = squares.get(s, 0.0) + x * x
            rises[s] = rises.get(s, 0) + (x > 0)
            all_n += 1; all_sum += x; all_sq += x * x; all_k += x > 0
        if all_n < burn_in:
            continue
        mu0, p0 = all_sum / all_n, all_k / all_n
        sd0 = np.sqrt((all_sq - all_sum**2 / all_n) / (all_n - 1))
        s = labels[t]
        n = count.get(s, 0)
        if n >= min_cell:
            sd = np.sqrt((squares[s] - total[s] ** 2 / n) / (n - 1))
        else:
            sd = sd0
        post = posterior_update(n, total.get(s, 0.0) / n if n else 0.0, sd, rises.get(s, 0), mu0, p0, n0)
        weight = float(np.clip(0.5 * post["mu_hat"] / sd**2, 0, 1))
        out[t] = post["mu_hat"], post["p_mu_pos"], post["p_up"], weight
    result = pd.DataFrame(out, index=types.index, columns=["mu_hat", "p_mu_pos", "p_up", "w"])
    result.insert(0, "type", labels)
    return result

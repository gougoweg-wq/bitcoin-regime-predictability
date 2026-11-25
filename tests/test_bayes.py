import numpy as np
import pandas as pd
import pytest

from btcregime import market_types, posterior_update, sequential_posterior


def test_worked_example_from_the_paper():
    post = posterior_update(n=645, xbar=0.004752, sd=0.03277, k=363, mu0=0.001709, p0=0.5318)
    assert post["mu_hat"] == pytest.approx(0.003902, abs=2e-6)
    assert post["mu_hat"] / post["se"] == pytest.approx(3.56, abs=0.01)
    assert post["p_mu_pos"] == pytest.approx(0.9998, abs=1e-4)
    assert post["p_up"] == pytest.approx(0.554, abs=5e-4)


def test_prior_and_data_limits():
    no_data = posterior_update(n=0, xbar=0.0, sd=0.03, k=0, mu0=0.002, p0=0.53)
    assert no_data["mu_hat"] == pytest.approx(0.002) and no_data["p_up"] == pytest.approx(0.53)
    much_data = posterior_update(n=10**7, xbar=0.005, sd=0.03, k=6 * 10**6, mu0=0.002, p0=0.53)
    assert much_data["mu_hat"] == pytest.approx(0.005, rel=1e-3) and much_data["p_up"] == pytest.approx(0.6, rel=1e-3)


def test_shrinkage_lies_between_prior_and_sample():
    post = posterior_update(n=100, xbar=0.01, sd=0.03, k=70, mu0=0.0, p0=0.5)
    assert 0.0 < post["mu_hat"] < 0.01 and 0.5 < post["p_up"] < 0.7


def test_library_matches_the_research_script(root, yahoo_long):
    """The packaged model reproduces the daily posteriors saved by the original experiment."""
    saved = pd.read_csv(f"{root}/results/phase4/exp10_bayes_daily.csv", index_col=0, parse_dates=True)
    ours = sequential_posterior(market_types(yahoo_long[:"2024-12-31"]))
    common = saved.index.intersection(ours.dropna().index)
    assert len(common) > 2800
    for column in ("mu_hat", "p_up", "p_mu_pos"):
        assert np.allclose(saved.loc[common, column], ours.loc[common, column], atol=1e-9), column

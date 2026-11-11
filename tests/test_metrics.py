import numpy as np
import pandas as pd

from snapme_eval.score import metrics, tokens


def test_ccc_perfect_and_shifted():
    x = np.array([100.0, 250, 400, 800])
    assert metrics.lins_ccc(x, x) == 1.0
    assert metrics.lins_ccc(x + 200, x) < 1.0


def test_agreement_basic():
    true = np.array([100.0, 200, 400])
    a = metrics.agreement(true * 1.1, true)
    assert abs(a["median_ape"] - 0.1) < 1e-9
    assert a["within_20pct"] == 1.0
    assert a["bias"] > 0


def test_cluster_bootstrap_is_deterministic():
    df = pd.DataFrame({"p": np.repeat(list("abcdefgh"), 5), "e": np.arange(40.0)})
    a = metrics.cluster_bootstrap(df, "p", lambda d: d["e"].mean(), reps=200)
    b = metrics.cluster_bootstrap(df, "p", lambda d: d["e"].mean(), reps=200)
    assert a == b and a[0] <= df["e"].mean() <= a[1]


def test_token_f1_matches_paper_definition():
    truth = tokens.clean("Bread, whole wheat, toasted")
    assert tokens.f1(truth, truth) == 1.0
    assert tokens.f1(truth, tokens.clean("Breads, wheat")) > 0
    assert tokens.f1(truth, tokens.clean("Coffee")) == 0.0

"""Unit tests for membership functions, the fuzzy layer and the network."""
import pytest
import torch

from src.fuzzy_layer import FuzzyLayer, cos_defuzz_trap, weighted_avg_defuzz
from src.fuzzy_network import FuzzyNetwork
from src.membership import gaussmf, trapmf

torch.set_default_dtype(torch.float32)


# ------------------------------------------------------------ membership
def test_trapmf_piecewise_values():
    p = torch.tensor([0.0, 1.0, 2.0, 4.0])
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    expected = torch.tensor([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    assert torch.allclose(trapmf(x, p), expected, atol=1e-6)


def test_trapmf_unsorted_params_are_sorted():
    x = torch.linspace(-1, 5, 50)
    assert torch.allclose(trapmf(x, torch.tensor([4.0, 0.0, 2.0, 1.0])),
                          trapmf(x, torch.tensor([0.0, 1.0, 2.0, 4.0])))


def test_trapmf_degenerate_triangle_and_crisp():
    tri = torch.tensor([0.0, 1.0, 1.0, 2.0])
    assert torch.isclose(trapmf(torch.tensor(1.0), tri), torch.tensor(1.0))
    assert torch.isclose(trapmf(torch.tensor(0.5), tri), torch.tensor(0.5))


def test_gaussmf_values_and_negative_sigma():
    p = torch.tensor([0.5, 0.2])
    assert torch.isclose(gaussmf(torch.tensor(0.5), p), torch.tensor(1.0))
    assert torch.isclose(gaussmf(torch.tensor(0.7), p), torch.exp(torch.tensor(-0.5)))
    assert torch.isclose(gaussmf(torch.tensor(0.7), torch.tensor([0.5, -0.2])),
                         gaussmf(torch.tensor(0.7), p))


# ------------------------------------------------------- defuzzification
def numeric_center_of_sums(w, cons, n=200_001):
    """Brute force: sum_i of clipped trapezoids, then centroid of that sum."""
    lo, hi = cons.min().item() - 1, cons.max().item() + 1
    y = torch.linspace(lo, hi, n, dtype=torch.float64)
    total = torch.zeros_like(y)
    for i in range(cons.shape[0]):
        total += torch.minimum(trapmf(y, cons[i].double()), torch.tensor(w[i], dtype=torch.float64))
    return (torch.trapezoid(y * total, y) / torch.trapezoid(total, y)).item()


@pytest.mark.parametrize("w", [[1.0, 0.0], [0.3, 0.8], [0.05, 0.5], [1.0, 1.0]])
def test_closed_form_cos_matches_numeric_integration(w):
    cons = torch.tensor([[0.0, 1.0, 2.0, 4.0], [1.0, 3.0, 3.5, 6.0]])
    closed = cos_defuzz_trap(torch.tensor([w]), cons.unsqueeze(1)).item()
    assert closed == pytest.approx(numeric_center_of_sums(w, cons), abs=1e-4)


def test_cos_single_symmetric_rule_gives_its_centre():
    cons = torch.tensor([[[1.0, 2.0, 4.0, 5.0]]])  # symmetric around 3
    for w in (0.1, 0.5, 1.0):
        assert cos_defuzz_trap(torch.tensor([[w]]), cons).item() == pytest.approx(3.0, abs=1e-5)


def test_weighted_average():
    w = torch.tensor([[1.0, 3.0]])
    centres = torch.tensor([[0.0], [4.0]])
    assert weighted_avg_defuzz(w, centres).item() == pytest.approx(3.0, abs=1e-5)


# ------------------------------------------------------------------ layer
def test_firing_strength_is_min_over_inputs():
    layer = FuzzyLayer(2, 1, 1, ante="trap", cons="trap")
    with torch.no_grad():
        layer.antes.copy_(torch.tensor([[[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0]]]))
    x = torch.tensor([[0.5, 3.0]])      # memberships 0.5 and 0.5
    assert layer.firing_strengths(x).item() == pytest.approx(0.5)
    x = torch.tensor([[0.25, 1.5]])     # 0.25 and 1.0 -> min 0.25
    assert layer.firing_strengths(x).item() == pytest.approx(0.25)


def test_layer_known_output_two_rules():
    """Hand-checkable example: x fires R0 fully, R1 not at all -> output = centre of C0."""
    layer = FuzzyLayer(1, 1, 2, ante="trap", cons="trap")
    with torch.no_grad():
        layer.antes.copy_(torch.tensor([[[0.0, 0.0, 0.4, 0.6]], [[0.4, 0.6, 1.0, 1.0]]]))
        layer.cons.copy_(torch.tensor([[[0.0, 0.1, 0.1, 0.2]], [[0.8, 0.9, 0.9, 1.0]]]))
    out = layer(torch.tensor([[0.2], [0.8], [0.5]]))
    assert out[0].item() == pytest.approx(0.1, abs=1e-5)
    assert out[1].item() == pytest.approx(0.9, abs=1e-5)
    assert out[2].item() == pytest.approx(0.5, abs=1e-5)   # both rules at 0.5, symmetric


@pytest.mark.parametrize("ante,cons", [("trap", "trap"), ("gauss", "gauss"),
                                       ("gauss", "trap"), ("trap", "gauss")])
def test_layer_shapes_and_gradients(ante, cons):
    layer = FuzzyLayer(3, 2, 5, ante=ante, cons=cons, generator=torch.Generator().manual_seed(0))
    x = torch.rand(16, 3)
    y = layer(x)
    assert y.shape == (16, 2) and torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert layer.antes.grad is not None and torch.isfinite(layer.antes.grad).all()


def test_param_counts_match_paper_formula():
    assert FuzzyLayer(3, 1, 7, "trap", "trap").n_params() == 112         # Table 2, [3,1,7]
    assert FuzzyLayer(1, 1, 20, "gauss", "gauss").n_params() == 80
    assert FuzzyLayer(1, 1, 20, "gauss", "gauss").n_params(effective=True) == 60
    # Ablation model [2,1,10]: 10*(2*2 + 1*2) = 60. The paper reports 40
    # (2 inputs x 1 output x 10 rules x 2), i.e. it omits the consequent MFs.
    assert FuzzyLayer(2, 1, 10, "gauss", "gauss").n_params() == 60


# ---------------------------------------------------------------- network
def test_network_chains_layers_and_counts_params():
    net = FuzzyNetwork([[3, 1, 7], [1, 1, 7], [1, 1, 4]], seed=0)
    assert net(torch.rand(10, 3)).shape == (10, 1)
    assert net.n_params() == 112 + 56 + 32                  # Table 2 row 3 reports 200


def test_network_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        FuzzyNetwork([[3, 2, 4], [1, 1, 4]])


def test_network_seed_is_reproducible():
    a, b = FuzzyNetwork([[2, 1, 4]], seed=7), FuzzyNetwork([[2, 1, 4]], seed=7)
    x = torch.rand(5, 2)
    assert torch.equal(a(x), b(x))


def test_rules_printout():
    net = FuzzyNetwork([[2, 1, 3]], seed=0)
    rules = net.rules()
    assert len(rules) == 3 and rules[0].startswith("R0: IF x1 is trap(")

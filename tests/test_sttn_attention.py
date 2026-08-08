import math

import pytest

torch = pytest.importorskip("torch")
Attention = pytest.importorskip("videowipe.models.sttn").Attention


def test_attention_matches_previous_formula_without_returning_weights():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 6)

    expected_weights = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1)),
        dim=-1,
    )
    expected = torch.matmul(expected_weights, value)

    actual = Attention()(query, key, value)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (2, 3, 6)
    torch.testing.assert_close(actual, expected)

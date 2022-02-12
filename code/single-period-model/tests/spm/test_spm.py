import math
import pytest
from spm.spm import (
    State,
    SinglePeriodEconomy,
)


class TestState:
    @pytest.mark.parametrize("left, right, is_equal", [
        (State(1, 0.10), State(1, 0.10), True),
        (State(2, 0.50), State(2, 0.50), True),
        (State(1, 0.10), State(2, 0.20), False),
        (State(1, 0.10), State(1, 0.20), False),
        (State(1, 0.10), State(2, 0.10), False),
    ])
    def test_compare(self, left, right, is_equal):
        assert (left == right) == is_equal

    @pytest.mark.parametrize("state", [
        State(1, 0.10),
        State(2, 0.20)
    ])
    def test_is_hashable(self, state):
        assert isinstance(hash(state), int)

    @pytest.mark.parametrize("state, future_value, expected", [
        (State(1, 0.10), 0.20, 0.02),
    ])
    def test_present_value(self, state, future_value, expected):
        assert math.isclose(state.present_value(future_value), expected)


class TestSinglePeriodEconomy:
    @pytest.mark.parametrize("economy_probabilities, error_match", [
        ([0.5, 0.4], "must sum to 1.0"),
        ([0.2, 0.9, -0.1], "must be between 0 and 1"),
    ])
    def test_invalid_probability(self, economy_probabilities, error_match):
        economy = {State(i, 0.1): p for i, p in enumerate(economy_probabilities)}
        with pytest.raises(ValueError, match=error_match):
            SinglePeriodEconomy(economy)

    @pytest.mark.parametrize("economy, expected", [
        ({State(1, 0.50): 0.5, State(2, 0.48): 0.5}, 0.98),
        ({State(1, 0.35): 0.3, State(2, 0.35): 0.3, State(3, 0.20): 0.4}, 0.90),
    ])
    def test_discount_factor(self, economy, expected):
        economy = SinglePeriodEconomy(economy)
        assert math.isclose(economy.discount_factor, expected)

    @pytest.mark.parametrize("economy, expected", [
        ({State(1, 0.50): 0.5, State(2, 0.48): 0.5}, 1.0/0.98),
    ])
    def test_risk_free_rate(self, economy, expected):
        economy = SinglePeriodEconomy(economy)
        assert math.isclose(economy.risk_free_rate, expected)

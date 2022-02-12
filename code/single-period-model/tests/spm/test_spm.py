import math
import pytest
from itertools import repeat
from spm.spm import (
    Equity,
    State,
    SinglePeriodEconomy,
    Asset,
    DebtPariPassu,
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


test_states_1 = [State(1, 0.07), State(2, 0.25), State(3, 0.28), State(4, 0.27), State(5, 0.11)]
test_asset_values_1 = [120, 110, 100, 95, 60]
test_asset_1 = Asset(dict(zip(test_states_1, test_asset_values_1)))
test_face_value_1 = [105, 80, 60]
test_present_values_1 = [93.85, 76.20, 58.80]

test_states_2 = [State(1, 0.25), State(2, 0.30)]
test_asset_values_2 = [120, 90]
test_asset_2 = Asset(dict(zip(test_states_2, test_asset_values_2)))
test_face_value_2 = [120, 90]
test_present_values_2 = [57.00, 49.50]


class TestAsset:
    @pytest.mark.parametrize("asset_values", [
        {State(1, 0.10): 0.10, State(2, 0.20): 0.20},
    ])
    def test_getitem(self, asset_values):
        asset = Asset(asset_values)
        for state, value in asset_values.items():
            assert asset[state] == value

    @pytest.mark.parametrize("asset, expected", [
        (test_asset_1, 96.15),
        (test_asset_2, 57.00),
    ])
    def test_present_value(self, asset, expected):
        assert math.isclose(asset.present_value, expected)


class TestDebtPariPassu:
    @pytest.mark.parametrize("asset, face_value, expected", [
        *zip(repeat(test_asset_1), test_face_value_1, test_present_values_1),
        *zip(repeat(test_asset_2), test_face_value_2, test_present_values_2),
    ])
    def test_present_value(self, asset, face_value, expected):
        debt = DebtPariPassu(asset, face_value=face_value)

        assert math.isclose(debt.present_value, expected)

    @pytest.mark.parametrize("asset, present_value, expected", [
        *zip(repeat(test_asset_1), test_present_values_1, test_face_value_1),
        *zip(repeat(test_asset_2), test_present_values_2, test_face_value_2),
    ])
    def test_face_value(self, asset, present_value, expected):
        debt = DebtPariPassu(asset, present_value=present_value)

        assert math.isclose(debt.face_value, expected)

    @pytest.mark.parametrize("asset, face_value, other_face_value, expected", [
        (test_asset_1, 1.0, 110, 0.860541),
        (test_asset_1, 1.0, 100, 0.916535),
        (test_asset_1, 1.0,  80, 0.951481),
    ])
    def test_pari_passu_present_value(self, asset, face_value, other_face_value, expected):
        debt = DebtPariPassu(asset, face_value=face_value, other_face_value=other_face_value)

        assert math.isclose(debt.present_value, expected, rel_tol=1e-5)

    @pytest.mark.parametrize("asset, present_value, other_face_value, expected", [
        (test_asset_1, 0.860541, 110, 1.0),
        (test_asset_1, 0.916535, 100, 1.0),
        (test_asset_1, 0.951481,  80, 1.0),
    ])
    def test_pari_passu_face_value(self, asset, present_value, other_face_value, expected):
        debt = DebtPariPassu(asset, present_value=present_value, other_face_value=other_face_value)

        assert math.isclose(debt.face_value, expected, rel_tol=1e-5)


class TestEquity:
    @pytest.mark.parametrize("asset, debt_face_value, expected", [
        (test_asset_1, 120,  0.00),
        (test_asset_1, 105,  2.30),
        (test_asset_1,  80, 19.95),
    ])
    def test_present_value(self, asset, debt_face_value, expected):
        equity = Equity(asset, debt_face_value)

        assert math.isclose(equity.present_value, expected)
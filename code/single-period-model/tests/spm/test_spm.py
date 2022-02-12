import math
import pytest
from spm.spm import (
    State,
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

import abc
import logging
import functools
from scipy import optimize

from typing import Dict


class State:
    def __init__(self, state_id, state_price):
        self.state_id = state_id
        self.state_price = state_price

    def __str__(self):
        return "State: {}, Price: {}".format(self.state_id, self.state_price)

    def __repr__(self):
        return "State({}, {})".format(self.state_id, self.state_price)

    def __eq__(self, other):
        return self.state_id == other.state_id and self.state_price == other.state_price

    def __hash__(self):
        return hash((self.state_id, self.state_price))

    def present_value(self, value: float) -> float:
        return self.state_price * value


class SinglePeriodEconomy:
    def __init__(self, states: Dict[State, float]):
        self.validate_states(states)
        self.states = states

    @staticmethod
    def validate_states(states: Dict[State, float]) -> None:
        total_probability = 0.0
        for state, probability in states.items():
            if not 0 <= probability <= 1:
                raise ValueError(f"Probability must be between 0 and 1, but was {probability} for state {state}")
            total_probability += probability

        if total_probability != 1.0:
            raise ValueError(f"Probability distribution must sum to 1.0, but sum to {total_probability}")

    def __str__(self):
        string = "States:\n"
        for state, probability in self.states.items():
            string += f"\t{state}, Probability: {probability}\n"

        return string

    @property
    def discount_factor(self):
        return sum(state.state_price for state in self.states.keys())

    @property
    def risk_free_rate(self):
        return 1.0 / self.discount_factor


class Payoffs:
    def __init__(self, payoffs: Dict[State, float]):
        self.payoffs = payoffs

    def __getitem__(self, item):
        return self.payoffs[item]


class Asset:
    def __init__(self, values: Dict[State, float]):
        self.values = values

    def __getitem__(self, item):
        return self.values[item]

    def __iter__(self):
        return iter(self.values.items())

    @property
    def states(self):
        return self.values.keys()

    def __str__(self):
        string = "States Values:\n"
        for state, value in self:
            string += f"\t{state}, Value: {value}\n"

        return string


class Derivative(abc.ABC):
    def __init__(self, underlying: Asset):
        self.underlying = underlying

    @abc.abstractmethod
    def payoff(self, state: State) -> float:
        pass


class Equity(Derivative):
    def __init__(self, underlying: Asset, debt_face_value: float):
        super().__init__(underlying)
        self.debt_face_value = debt_face_value

    @property
    def present_value(self) -> float:
        return sum(state.present_value(self.payoff(state)) for state in self.underlying.states)

    def payoff(self, state: State) -> float:
        return max(self.underlying[state] - self.debt_face_value, 0)


class DebtPariPassu(Derivative):
    def __init__(self, underlying: Asset, present_value: float = None, face_value: float = None,
                 other_face_value: float = 0):
        super().__init__(underlying)
        self._present_value = present_value
        self._face_value = face_value
        self.other_face_value = other_face_value

    @functools.cached_property
    def face_value(self) -> float:
        if self._face_value is not None:
            return self._face_value

        def objective(face_value):
            present_value = self._calculate_present_value(face_value)
            return present_value - self.present_value

        logging.debug(f"Solving for face value of {self}")
        sol = optimize.fsolve(func=objective, x0=self.present_value)
        logging.debug(f"Obtained solution {sol}")

        self._face_value = sol[0]
        return self._face_value

    @functools.cached_property
    def present_value(self) -> float:
        if self._present_value is not None:
            return self._present_value

        present_value = self._calculate_present_value(self.face_value)

        self._present_value = present_value
        return self._present_value

    def _calculate_present_value(self, face_value):
        present_value = face_value * sum(state.present_value(self.recovery_rate(underlying_value, face_value))
                                         for state, underlying_value in self.underlying)
        return present_value

    def recovery_rate(self, underlying_value, face_value) -> float:
        return min(1.0, underlying_value / (face_value + self.other_face_value))

    def payoff(self, state: State) -> float:
        return self.face_value * self.recovery_rate(self.underlying[state], self.face_value)

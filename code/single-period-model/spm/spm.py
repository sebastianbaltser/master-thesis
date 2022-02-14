import abc
import logging
import functools
from scipy import optimize

from typing import (
    Collection,
    Dict,
    Any,
    Union,
)


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


class States:
    """
    A mapping from instances of ``State`` to values.

    Args:
        states (Dict[State, Any] | Collection[State]):
            A dictionary mapping instances of ``State`` to values. If a collection the
            parameter ``values`` should be specified.
        values (Collection[Any]):
            A collection of values. If a dictionary is specified in the parameter ``states``
            the parameter ``values`` should not be specified.
    """
    def __init__(self, states, values=None):
        if values is not None:
            states = dict(zip(states, values))

        self._states = states

    def __getitem__(self, item):
        return self._states[item]

    def __iter__(self):
        return iter(self._states.items())

    @property
    def states(self):
        return self._states.keys()

    def __eq__(self, other: "States") -> bool:
        return self._states == other._states

    def __add__(self, other: "States") -> "States":
        states = self._states.keys() | other._states.keys()
        return self.__class__({state: self._states.get(state, 0) + other._states.get(state, 0) for state in states})

    def __str__(self):
        string = ''
        for state, value in self._states.items():
            string += f"State: {state.state_id}, Value: {value}\n"

        return string

    def __repr__(self):
        return f"States({self._states})"


class SinglePeriodEconomy:
    def __init__(self, states: States):
        self.validate_states(states)
        self.states = states

    @staticmethod
    def validate_states(states: States) -> None:
        total_probability = 0.0
        for state, probability in states:
            if not 0 <= probability <= 1:
                raise ValueError(f"Probability must be between 0 and 1, but was {probability} for state {state}")
            total_probability += probability

        if total_probability != 1.0:
            raise ValueError(f"Probability distribution must sum to 1.0, but sum to {total_probability}")

    def __str__(self):
        string = "States:\n"
        for state, probability in self.states:
            string += f"\t{state}, Probability: {probability}\n"

        return string

    @property
    def discount_factor(self):
        return sum(state.state_price for state in self.states.states)

    @property
    def risk_free_rate(self):
        return 1.0 / self.discount_factor

    def risk_neutral_probability(self, state: State) -> float:
        return self.risk_free_rate * state.state_price

    def risk_neutral_expectation(self, values: States) -> float:
        return sum(self.risk_neutral_probability(state) * value for state, value in values)


class Payoffs:
    def __init__(self, payoffs: States):
        self.payoffs = payoffs

    def __getitem__(self, item):
        return self.payoffs[item]


class Asset:
    def __init__(self, values: States):
        self.values = values

    def __getitem__(self, item):
        return self.values[item]

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other):
        return self.values == other.values

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if other.states <= self.states:
                return self.__class__(self.values + other.values)
            else:
                raise ValueError("States in right operand must be a subset of states in left operand")
        else:
            new_values = {state: value + other for state, value in self.values}
            return self.__class__(States(new_values))

    @property
    def states(self):
        return self.values.states

    @property
    def present_value(self) -> float:
        return sum(state.present_value(value) for state, value in self)

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return f"Asset({self.values!r})"


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

    @functools.cached_property
    def bond_yield(self) -> float:
        return self.face_value / self.present_value - 1

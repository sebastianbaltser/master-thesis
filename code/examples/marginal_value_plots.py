import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np

from spm import (
    State,
    States,
    Asset,
    Firm,
    Equity,
    DebtPariPassu,
    SinglePeriodEconomy,
)


def get_states():
    states = [State(1, 0.06),
              State(2, 0.24),
              State(3, 0.29),
              State(4, 0.28),
              State(5, 0.12)]
    return states


def get_economy():
    states = get_states()
    return SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))


def get_firm_assets():
    states = get_states()
    return Asset(States(states, [120, 110, 100, 95, 60]))


def get_legacy_debt():
    firm_assets = get_firm_assets()
    return DebtPariPassu(firm_assets, face_value=80)


def get_legacy_equity():
    firm_assets = get_firm_assets()
    debt = get_legacy_debt()
    Equity(firm_assets, debt_face_value=debt.face_value)


def get_new_firm_from_debt_financed_option(option, option_price):
    firm_assets = get_firm_assets()
    new_firm_assets = firm_assets + option
    debt = get_legacy_debt()
    new_debt = DebtPariPassu(new_firm_assets, other_face_value=debt.face_value, present_value=option_price)

    total_face_value = debt.face_value + new_debt.face_value
    return Firm(new_firm_assets, debt_face_value=total_face_value)


def marginal_shareholder_value_of_debt_financing(option, option_price):
    economy = get_economy()
    firm = get_new_firm_from_debt_financed_option(option, option_price)
    default_probability = economy.risk_neutral_expectation(economy.map(firm.is_default_state))
    expected_payoff = economy.risk_neutral_expectation(option)

    expected_profit = (1 - default_probability) * (expected_payoff * economy.discount_factor - option_price)

    payoff_default_covariance = economy.risk_neutral_covariance(option, economy.map(firm.is_default_state))
    payoff_default_covariance *= economy.discount_factor

    expected_loss_rate = economy.risk_neutral_expectation(economy.map(firm.loss_rate))
    credit_spread = expected_loss_rate*economy.risk_free_rate / (1 - expected_loss_rate)
    marginal_excess_return = (1-default_probability) * economy.discount_factor * option_price * credit_spread

    return expected_profit - payoff_default_covariance - marginal_excess_return


def plot_marginal_shareholder_value_debt_financing():
    option = Asset(States(get_states(), [10, 10, 10, 10, 10]))
    option_price_range = np.arange(1, 30, 0.01)

    marginal_shareholder_values = [marginal_shareholder_value_of_debt_financing(option, option_price)
                                   for option_price in option_price_range]

    fig, ax = plt.subplots()
    ax.plot(option_price_range, marginal_shareholder_values)
    ax.set_xlabel("Option Price")
    ax.set_ylabel("Marginal Shareholder Value")
    ax.set_title("Marginal Shareholder Value of Debt Financing")

    return fig


if __name__ == '__main__':
    option = Asset(States(get_states(), [10, 10, 10, 10, 10]))
    option_price = 9.90
    marginal_shareholder_value = marginal_shareholder_value_of_debt_financing(option, option_price)
    print(f"Marginal Shareholder Value Debt Financing: {marginal_shareholder_value:.4f}")
    marginal_shareholder_value = marginal_shareholder_value_of_equity_financing(option, option_price)
    print(f"Marginal Shareholder Value Equity Financing: {marginal_shareholder_value:.4f}")

    fig = plot_marginal_shareholder_value_debt_financing()
    plt.show()
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

from spm import (
    State,
    States,
    Asset,
    Firm,
    SinglePeriodEconomy,
)
from spm.marginal_value_plots import (
    marginal_shareholder_value_of_debt_financing,
    marginal_shareholder_value_of_equity_financing,
    plot_marginal_shareholder_value,
)


def main():
    states_ids = np.arange(-4, 4.001, 0.1)
    state_densities = np.array([stats.t.pdf(state_id, 1, 0) for state_id in states_ids])
    state_probabilities = state_densities / np.sum(state_densities)
    state_prices = state_probabilities * 0.99
    states = [State(state_id, state_price) for state_id, state_price in zip(states_ids, state_prices)]
    true_probabilities = np.array([1/len(states)] * len(states))
    true_probabilities /= np.sum(true_probabilities)
    economy = SinglePeriodEconomy(States(states, true_probabilities))

    asset_value_low, asset_value_high = 10, 12
    asset_value_slope = (asset_value_high - asset_value_low) / (len(states) - 1)
    asset_values = [asset_value_low + asset_value_slope*x for x in range(0, len(states))]

    firm = Firm(Asset(States(states, asset_values)), debt_face_value=11)
    option_payoff = [1 for x in range(0, len(states))]
    option = Asset(States(states, option_payoff))
    option_price = 0.99
    marginal_shareholder_value = marginal_shareholder_value_of_debt_financing(economy, firm, option, option_price)
    print(f"Marginal Shareholder Value Debt Financing: {marginal_shareholder_value:.4f}")
    marginal_shareholder_value = marginal_shareholder_value_of_equity_financing(economy, firm, option, option_price)
    print(f"Marginal Shareholder Value Equity Financing: {marginal_shareholder_value:.4f}")

    option_price_range = np.arange(0, 2, 0.01)
    fig = plot_marginal_shareholder_value(economy, firm, option, option_price_range)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(states_ids, state_prices*100, label="State Prices * 100")
    ax.plot(states_ids, asset_values, label="Asset Value")
    ax.plot(states_ids, option_payoff, label="Option Payoff")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
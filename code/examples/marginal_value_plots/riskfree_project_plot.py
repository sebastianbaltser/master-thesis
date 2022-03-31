import matplotlib.pyplot as plt

import numpy as np

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
    states = [State(1, 0.06),
              State(2, 0.24),
              State(3, 0.29),
              State(4, 0.28),
              State(5, 0.12)]

    economy = SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))
    firm = Firm(Asset(States(states, [120, 110, 100, 95, 60])), debt_face_value=80)
    option = Asset(States(states, [10, 10, 10, 10, 10]))
    option_price = 9.90
    marginal_shareholder_value = marginal_shareholder_value_of_debt_financing(economy, firm, option, option_price)
    print(f"Marginal Shareholder Value Debt Financing: {marginal_shareholder_value:.4f}")
    marginal_shareholder_value = marginal_shareholder_value_of_equity_financing(economy, firm, option, option_price)
    print(f"Marginal Shareholder Value Equity Financing: {marginal_shareholder_value:.4f}")

    option_price_range = np.arange(1, 30, 0.01)
    fig = plot_marginal_shareholder_value(economy, firm, option, option_price_range)
    plt.show()


if __name__ == '__main__':
    main()
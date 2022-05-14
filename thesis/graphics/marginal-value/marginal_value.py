import matplotlib.pyplot as plt
import matplotlib
import pathlib
import tqdm
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
    marginal_creditor_value_of_debt_financing,
    get_new_firm_from_debt_financed_option,
    get_new_firm_from_cash_financed_option,
    marginal_shareholder_value_of_equity_financing,
    marginal_shareholder_value_of_cash_financing,
    plot_marginal_shareholder_value
)

plt.style.use("ggplot")

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def debt_financing_plot():
    states = [State(1, 0.06),
              State(2, 0.24),
              State(3, 0.29),
              State(4, 0.28),
              State(5, 0.12)]

    economy = SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))
    base_firm = Firm(Asset(States(states, [120, 110, 100, 95, 60])), debt_face_value=80)
    option = Asset(States(states, [1, 1, 1, 1, 1]))

    option_price_range = np.arange(0.91, 1.01, 0.01)

    firms = [get_new_firm_from_debt_financed_option(base_firm, option, option_price)
             for option_price in option_price_range]
    firm_option_price_pairs = zip(firms, option_price_range)
    g_debt = [marginal_shareholder_value_of_debt_financing(economy, firm, option, option_price)
              for firm, option_price in tqdm.tqdm(firm_option_price_pairs, desc="Debt Funding", total=len(firms))]

    h_debt = [marginal_creditor_value_of_debt_financing(base_firm, option, option_price)
              for option_price in tqdm.tqdm(option_price_range, desc="Creditors", total=len(firms))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(option_price_range, g_debt, label="Shareholders")
    ax.plot(option_price_range, h_debt, label="Creditors")
    ax.set_xlim([option_price_range[0], option_price_range[-1]])
    ylim = [-0.045, 0.045]
    ax.set_ylim(ylim)
    ax.plot([option_price_range[0], option_price_range[-1]], [0, 0], "-.", color="grey", linewidth=0.5)

    marks = [(0.99, -0.026874, 0.026874),
             (0.97, -0.008734, 0.028734),
             (0.960370, 0, 0.99-0.960370)]
    for price, shareholder_value, creditor_value in marks:
        ax.plot([price, price], [ylim[0]*0.9, ylim[1]*0.9], "--", color="black", linewidth=0.5)
        ax.scatter([price, price], [shareholder_value, creditor_value], marker="o", color="black", s=15, zorder=20)
        # ax.text(price, shareholder_value, f"{shareholder_value:.2f}", ha="right", va="bottom")
        annotate_kwargs = dict(xytext=(4, 4), textcoords="offset points", fontsize=8)
        for value in [shareholder_value, creditor_value]:
            ax.annotate(f"${value:.4f}$", (price, value), **annotate_kwargs)

    ax.set_xlabel("Project Price")
    ax.set_ylabel("Marginal Value")
    ax.set_title("Marginal shareholder and creditor value for different project prices")
    ax.legend()

    return fig


def equity_financing_plot():
    states = [State(1, 0.06),
              State(2, 0.24),
              State(3, 0.29),
              State(4, 0.28),
              State(5, 0.12)]

    economy = SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))
    base_firm = Firm(Asset(States(states, [120, 110, 100, 95, 60])), debt_face_value=80)
    option = Asset(States(states, [1, 1, 1, 1, 1]))

    theoretical_option_price = economy.risk_neutral_expectation(option.values) * economy.discount_factor
    option_price_range = np.arange(0.70, 1.05, 0.01)

    firms = [get_new_firm_from_debt_financed_option(base_firm, option, option_price)
             for option_price in option_price_range]
    firm_option_price_pairs = zip(firms, option_price_range)
    g_equity = [marginal_shareholder_value_of_equity_financing(economy, firm, option, option_price)
                for firm, option_price in tqdm.tqdm(firm_option_price_pairs, desc="Debt Funding", total=len(firms))]

    h_equity = [theoretical_option_price - option_price - g for g, option_price in zip(g_equity, option_price_range)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(option_price_range, g_equity, label="Shareholders")
    ax.plot(option_price_range, h_equity, label="Creditors")
    ax.set_xlim([option_price_range[0], option_price_range[-1]])
    ylim = [-0.18, 0.18]
    ax.set_ylim(ylim)
    ax.plot([option_price_range[0], option_price_range[-1]], [0, 0], "-.", color="grey", linewidth=0.5)

    marks = [(0.99, -0.12, 0.12),
             (0.87, 0, 0.99-0.87)]
    for price, shareholder_value, creditor_value in marks:
        ax.plot([price, price], [ylim[0]*0.9, ylim[1]*0.9], "--", color="black", linewidth=0.5)
        ax.scatter([price, price], [shareholder_value, creditor_value], marker="o", color="black", s=15, zorder=20)
        # ax.text(price, shareholder_value, f"{shareholder_value:.2f}", ha="right", va="bottom")
        annotate_kwargs = dict(xytext=(4, 4), textcoords="offset points", fontsize=8)
        for value in [shareholder_value, creditor_value]:
            ax.annotate(f"${value:.4f}$", (price, value), **annotate_kwargs)

    ax.set_xlabel("Project Price")
    ax.set_ylabel("Marginal Value")
    ax.set_title("Marginal shareholder and creditor value for different project prices")
    ax.legend()

    return fig


def cash_financing_plot():
    states = [State(1, 0.06),
              State(2, 0.24),
              State(3, 0.29),
              State(4, 0.28),
              State(5, 0.12)]

    economy = SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))
    base_firm = Firm(Asset(States(states, [120, 110, 100, 95, 60])), debt_face_value=80)
    option = Asset(States(states, [1, 1, 1, 1, 1]))

    theoretical_option_price = economy.risk_neutral_expectation(option.values)
    option_price_range = np.arange(0.95, 1.01, 0.01)

    firms = [get_new_firm_from_cash_financed_option(economy, base_firm, option, option_price)
             for option_price in option_price_range]
    firm_option_price_pairs = zip(firms, option_price_range)
    g_equity = [marginal_shareholder_value_of_cash_financing(economy, firm, option, option_price)
                for firm, option_price in tqdm.tqdm(firm_option_price_pairs, desc="Debt Funding", total=len(firms))]

    h_equity = [theoretical_option_price - option_price - g for g, option_price in zip(g_equity, option_price_range)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(option_price_range, g_equity, label="Shareholders")
    ax.plot(option_price_range, h_equity, label="Creditors")
    ax.set_xlim([option_price_range[0], option_price_range[-1]])
    ax.set_ylim([-0.15, 0.15])
    ax.plot([option_price_range[0], option_price_range[-1]], [0, 0], "--", color="black", linewidth=0.5)
    ax.set_xlabel("Project Price")
    ax.set_ylabel("Marginal Value")
    ax.set_title("Marginal shareholder and creditor value for different project prices")
    ax.legend()

    return fig


def marginal_shareholder_value_plot():
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


if __name__ == '__main__':
    PATH = pathlib.Path(__file__).parent.absolute()

    fig = debt_financing_plot()
    fig.savefig(PATH / "debt-financing.pgf")
    fig = equity_financing_plot()
    fig.savefig(PATH / "equity-financing.pgf")
    fig = cash_financing_plot()
    fig.savefig(PATH / "cash-financing.pgf")
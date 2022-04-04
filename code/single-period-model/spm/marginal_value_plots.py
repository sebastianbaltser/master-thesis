import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from spm import (
    Firm,
    DebtPariPassu,
    SinglePeriodEconomy,
)


def get_new_firm_from_debt_financed_option(firm: Firm, option, option_price):
    firm_assets = firm.assets
    new_firm_assets = firm_assets + option
    debt_face_value = firm.debt_face_value
    new_debt = DebtPariPassu(new_firm_assets, other_face_value=debt_face_value, present_value=option_price)

    total_face_value = debt_face_value + new_debt.face_value
    return Firm(new_firm_assets, debt_face_value=total_face_value)


def calculate_credit_spread(economy: SinglePeriodEconomy, firm: Firm):
    expected_loss_rate = economy.risk_neutral_expectation(economy.map(firm.loss_rate))
    credit_spread = expected_loss_rate*economy.risk_free_rate / (1 - expected_loss_rate)

    return credit_spread


def marginal_shareholder_value_of_debt_financing(economy: SinglePeriodEconomy, firm: Firm, option, option_price):
    firm = get_new_firm_from_debt_financed_option(firm, option, option_price)
    default_probability = economy.risk_neutral_expectation(economy.map(firm.is_default_state))
    expected_payoff = economy.risk_neutral_expectation(option)

    expected_profit = (1 - default_probability) * (expected_payoff * economy.discount_factor - option_price)

    payoff_default_covariance = economy.risk_neutral_covariance(option, economy.map(firm.is_default_state))
    payoff_default_covariance *= economy.discount_factor

    credit_spread = calculate_credit_spread(economy, firm)
    marginal_excess_return = (1-default_probability) * economy.discount_factor * option_price * credit_spread

    return expected_profit - payoff_default_covariance - marginal_excess_return


def get_new_firm_from_equity_financed_option(firm: Firm, option):
    firm_assets = firm.assets
    new_firm_assets = firm_assets + option
    debt_face_value = firm.debt_face_value

    return Firm(new_firm_assets, debt_face_value=debt_face_value)


def marginal_shareholder_value_of_equity_financing(economy: SinglePeriodEconomy, firm: Firm, option, option_price):
    firm = get_new_firm_from_equity_financed_option(firm, option)
    default_probability = economy.risk_neutral_expectation(economy.map(firm.is_default_state))
    expected_payoff = economy.risk_neutral_expectation(option)

    expected_profit = (1 - default_probability) * (expected_payoff * economy.discount_factor - option_price)

    payoff_default_covariance = economy.risk_neutral_covariance(option, economy.map(firm.is_default_state))
    payoff_default_covariance *= economy.discount_factor

    shareholder_direct_loss = default_probability * option_price

    return expected_profit - payoff_default_covariance - shareholder_direct_loss


def plot_marginal_shareholder_value_debt_financing(economy, firm, option, option_price_range):
    marginal_shareholder_values = [marginal_shareholder_value_of_debt_financing(economy, firm, option, option_price)
                                   for option_price in option_price_range]

    fig, ax = plt.subplots()
    ax.plot(option_price_range, marginal_shareholder_values)
    ax.set_xlabel("Option Price")
    ax.set_ylabel("Marginal Shareholder Value")
    ax.set_title("Marginal Shareholder Value of Debt Financing")

    return fig


def plot_marginal_shareholder_value(economy, firm, option, option_price_range) -> plt.Figure:
    g_debt = [marginal_shareholder_value_of_debt_financing(economy, firm, option, option_price)
              for option_price in tqdm.tqdm(option_price_range, desc="Debt Funding")]
    g_equity = [marginal_shareholder_value_of_equity_financing(economy, firm, option, option_price)
                for option_price in tqdm.tqdm(option_price_range, desc="Equity Funding")]

    fig, ax = plt.subplots()
    ax.plot(option_price_range, g_debt, label="Debt Funding")
    ax.plot(option_price_range, g_equity, label="Equity Funding")
    ax.set_xlabel("Option Price")
    ax.set_ylabel("Marginal Shareholder Value")
    ax.set_title("Marginal Shareholder Value of Different Financing Methods")
    ax.legend()

    return fig

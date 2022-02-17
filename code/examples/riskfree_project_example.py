from spm import (
    State,
    States,
    Asset,
    Equity,
    DebtPariPassu,
    SinglePeriodEconomy,
)


def main():
    states = [State(1, 0.06),
              State(2, 0.24),
              State(3, 0.29),
              State(4, 0.28),
              State(5, 0.12)]
    print(states)
    economy = SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))
    gross_risk_free_rate = economy.risk_free_rate
    risk_free_rate = gross_risk_free_rate - 1
    print(f"Risk free rate: {risk_free_rate:.4%}")
    firm = Asset(States(states, [120, 110, 100, 95, 60]))
    print("Firm asset values:")
    print(firm)
    legacy_debt_face_value = 80
    debt = DebtPariPassu(firm, face_value=legacy_debt_face_value)
    print("Legacy debt payoff:")
    print(debt.payoff)
    equity = Equity(firm, debt_face_value=debt.face_value)
    print("Equity payoff:")
    print(equity.payoff)

    print(f"Equity value: {equity.present_value:.3f}")
    print(f"Debt value: {debt.present_value:.3f}")

    expected_loss_rate = economy.risk_neutral_expectation(
        States({state: (1 - payoff / debt.face_value) for state, payoff in debt.payoff})
    )
    debit_value_adjustment = expected_loss_rate * debt.face_value / gross_risk_free_rate
    print(f"DVA: {debit_value_adjustment:.4f}")

    option_promised_payoff = 10
    option_price = option_promised_payoff / (gross_risk_free_rate + 0.00)
    option = Asset(States(states, [option_promised_payoff]*len(states)))
    print("\nNew option payoffs:")
    print(option)
    print(f"Option present value: {option.present_value:.3f}")
    print(f"Option price: {option_price:.3f}")

    legacy_firm = firm
    firm = legacy_firm + option
    new_debt = DebtPariPassu(firm, present_value=option_price, other_face_value=debt.face_value)
    legacy_debt = DebtPariPassu(firm, face_value=legacy_debt_face_value, other_face_value=new_debt.face_value)
    total_face_value = legacy_debt.face_value + new_debt.face_value
    new_equity = Equity(firm, debt_face_value=total_face_value)

    credit_spread = (legacy_debt.bond_yield - risk_free_rate)

    print(f"Firm present value: {firm.present_value:.3f}")
    print(f"\tChange in firm value: {firm.present_value:.4f} - {legacy_firm.present_value:.4f} "
          f"= {firm.present_value-legacy_firm.present_value:.4f}")
    print(f"Equity present value: {new_equity.present_value:.3f}")
    print(f"\tChange in equity value: {new_equity.present_value:.3f} - {equity.present_value:.3f} "
          f"= {new_equity.present_value - equity.present_value:.3f}")
    state_is_no_default = States({state: total_face_value <= value for state, value in firm})
    promised_return = (option_promised_payoff - option.present_value * (gross_risk_free_rate + credit_spread))
    expected_promised_return = economy.risk_neutral_expectation(
        States({state: promised_return*is_no_default for state, is_no_default in state_is_no_default})
    )
    marginal_value_to_shareholders = 1 / gross_risk_free_rate * expected_promised_return
    print(f"\tADS marginal value to shareholders of debt financing: {marginal_value_to_shareholders:.3f}")
    print(f"Legacy debt present value: {legacy_debt.present_value:.3f}")
    print(f"\tChange in legacy debt value: {legacy_debt.present_value - debt.present_value:.3f}")
    print(f"Equity payoff:\n{new_equity.payoff}")
    print(f"Legacy debt payoff:\n{legacy_debt.payoff}")
    print(f"New debt payoff:\n{new_debt.payoff}")
    print(f"New debt present value: {new_debt.present_value:.3f}")
    print(f"New debt face value: {new_debt.face_value:.3f}")

    print("\nFunding Value Adjustments")
    print(f"FVA 1: {(new_debt.face_value - new_debt.present_value*gross_risk_free_rate) / gross_risk_free_rate:.4f}")
    expected_loss_rate = economy.risk_neutral_expectation(
        States({state: (1 - payoff / new_debt.face_value) for state, payoff in new_debt.payoff})
    )
    debit_value_adjustment = expected_loss_rate * new_debt.face_value / gross_risk_free_rate
    print(f"DVA: {debit_value_adjustment:.4f}")

    print(f"FVA 2: {equity.present_value - new_equity.present_value:.4f}")
    option_pv_expected_payoff = economy.risk_neutral_expectation(option.values) / gross_risk_free_rate
    no_default_probability = economy.risk_neutral_expectation(state_is_no_default)
    wealth_transfer = option_pv_expected_payoff * credit_spread * no_default_probability / gross_risk_free_rate
    print(f"ADS wealth transfer: {wealth_transfer:.4f}")

    print("\nFund Transfer Pricing")
    new_debt = DebtPariPassu(firm, face_value=option_promised_payoff, other_face_value=legacy_debt.face_value)
    legacy_debt = DebtPariPassu(firm, face_value=legacy_debt.face_value, other_face_value=new_debt.face_value)
    credit_spread = (legacy_debt.bond_yield - risk_free_rate)
    total_face_value = legacy_debt.face_value + new_debt.face_value
    new_equity = Equity(firm, debt_face_value=total_face_value)

    print(f"New debt fair present value: {new_debt.present_value:.4f}")
    print(f"Donation size: {option.present_value - new_debt.present_value:.4f}")

    print(f"Equity present value: {new_equity.present_value:.4f}")
    print(f"\tChange in equity value: {new_equity.present_value - equity.present_value:.4f}")
    print(f"Legacy debt present value: {legacy_debt.present_value:.4f}")
    print(f"\tChange in legacy debt value: {legacy_debt.present_value - debt.present_value:.4f}")

    state_is_no_default = States({state: total_face_value <= value for state, value in firm})
    no_default_probability = economy.risk_neutral_expectation(state_is_no_default)
    option_pv_expected_payoff = economy.risk_neutral_expectation(option.values) / gross_risk_free_rate
    wealth_transfer = option_pv_expected_payoff * credit_spread * no_default_probability / gross_risk_free_rate

    donation_size = wealth_transfer/no_default_probability * gross_risk_free_rate/(gross_risk_free_rate+credit_spread)
    print(f"FVA 3: {donation_size:.4f}")
    expected_loss_rate = economy.risk_neutral_expectation(
        States({state: (1 - payoff / new_debt.face_value) for state, payoff in new_debt.payoff})
    )
    debit_value_adjustment = expected_loss_rate * new_debt.face_value / gross_risk_free_rate
    print(f"DVA: {debit_value_adjustment:.4f}")

    print("Equity Funding:")
    legacy_debt = DebtPariPassu(firm, face_value=legacy_debt_face_value)
    new_equity_share = option.present_value / (firm.present_value - legacy_debt.present_value)
    print(new_equity_share)
    legacy_equity = Equity(firm, debt_face_value=legacy_debt.face_value, equity_share=1-new_equity_share)
    new_equity = Equity(firm, debt_face_value=legacy_debt.face_value, equity_share=new_equity_share)

    print(f"\tLegacy debt present value: {legacy_debt.present_value:.4f}")
    print(f"\tLegacy equity present value: {legacy_equity.present_value:.4f}")
    print(f"\tNew equity present value: {new_equity.present_value:.4f}")
    print(f"\tChange in legacy debt value: {legacy_debt.present_value - debt.present_value:.4f}")
    print(f"\tChange in legacy equity value: {legacy_equity.present_value - equity.present_value:.4f}")


if __name__ == "__main__":
    main()
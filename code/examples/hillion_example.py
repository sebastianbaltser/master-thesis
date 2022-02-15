from spm import (
    State,
    States,
    Asset,
    Equity,
    DebtPariPassu,
    SinglePeriodEconomy,
)


def main():
    risk_free_rate = 0.05
    gross_risk_free_rate = 1 + risk_free_rate
    states = [State("u", 0.625/gross_risk_free_rate), State("d", 0.375/gross_risk_free_rate)]
    economy = SinglePeriodEconomy(States(states, [0.5, 0.5]))
    asset = Asset(States(states, [120, 80]))
    print("Asset values:")
    print(asset)

    debt_face_value = 90
    debt = DebtPariPassu(asset, face_value=debt_face_value)
    equity = Equity(asset, debt_face_value=debt_face_value)

    print(f"Equity value: {equity.present_value:.3f}")
    print(f"Debt value: {debt.present_value:.3f}")

    print(f"DVA: {90 / gross_risk_free_rate:.3f} - {debt.present_value:.3f} "
          f"= {90 / gross_risk_free_rate - debt.present_value:.3f}")

    option_promised_payoff = 15.75
    option = Asset(States(states, [option_promised_payoff, option_promised_payoff]))
    print("\nNew option payoffs:")
    print(option)
    print(f"Option present value: {option.present_value:.3f}")

    asset = asset + option
    new_debt = DebtPariPassu(asset, present_value=option.present_value, other_face_value=debt_face_value)
    legacy_debt = DebtPariPassu(asset, face_value=debt_face_value, other_face_value=new_debt.face_value)
    total_face_value = debt_face_value + new_debt.face_value
    new_equity = Equity(asset, debt_face_value=total_face_value)

    credit_spread = (legacy_debt.bond_yield - risk_free_rate)

    print(f"Equity present value: {new_equity.present_value:.3f}")
    print(f"\tChange in equity value: {new_equity.present_value - equity.present_value:.3f}")
    state_is_no_default = States({state: total_face_value <= value for state, value in asset})
    promised_return = (option_promised_payoff - option.present_value * (gross_risk_free_rate + credit_spread))
    expected_promised_return = economy.risk_neutral_expectation(
        States({state: promised_return*is_no_default for state, is_no_default in state_is_no_default})
    )
    marginal_value_to_shareholders = 1 / gross_risk_free_rate * expected_promised_return
    print(f"\tADS marginal value to shareholders of debt financing: {marginal_value_to_shareholders:.3f}")
    print(f"Legacy debt present value: {legacy_debt.present_value:.3f}")
    print(f"\tChange in legacy debt value: {legacy_debt.present_value - debt.present_value:.3f}")
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
    new_debt = DebtPariPassu(asset, face_value=option_promised_payoff, other_face_value=legacy_debt.face_value)
    legacy_debt = DebtPariPassu(asset, face_value=legacy_debt.face_value, other_face_value=new_debt.face_value)
    credit_spread = (legacy_debt.bond_yield - risk_free_rate)
    total_face_value = legacy_debt.face_value + new_debt.face_value
    new_equity = Equity(asset, debt_face_value=total_face_value)

    print(f"New debt fair present value: {new_debt.present_value:.4f}")
    print(f"Donation size: {option.present_value - new_debt.present_value:.4f}")

    print(f"Equity present value: {new_equity.present_value:.4f}")
    print(f"\tChange in equity value: {new_equity.present_value - equity.present_value:.4f}")
    print(f"Legacy debt present value: {legacy_debt.present_value:.4f}")
    print(f"\tChange in legacy debt value: {legacy_debt.present_value - debt.present_value:.4f}")

    state_is_no_default = States({state: total_face_value <= value for state, value in asset})
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
    legacy_debt = DebtPariPassu(asset, face_value=debt_face_value)
    legacy_equity = Equity(asset, debt_face_value=legacy_debt.face_value, equity_share=1)
    new_equity_share = option.present_value / legacy_equity.present_value
    legacy_equity = Equity(asset, debt_face_value=legacy_debt.face_value, equity_share=1-new_equity_share)
    new_equity = Equity(asset, debt_face_value=legacy_debt.face_value, equity_share=new_equity_share)

    print(f"\tLegacy debt present value: {legacy_debt.present_value:.4f}")
    print(f"\tLegacy equity present value: {legacy_equity.present_value:.4f}")
    print(f"\tNew equity present value: {new_equity.present_value:.4f}")
    print(f"\tChange in legacy debt value: {legacy_debt.present_value - debt.present_value:.4f}")
    print(f"\tChange in legacy equity value: {legacy_equity.present_value - equity.present_value:.4f}")

if __name__ == "__main__":
    main()

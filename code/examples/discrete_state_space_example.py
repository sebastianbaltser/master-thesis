from spm import (
    State,
    States,
    Asset,
    Equity,
    DebtPariPassu,
    SinglePeriodEconomy,
)


def main():
    states = [State(1, 0.07),
              State(2, 0.25),
              State(3, 0.28),
              State(4, 0.27),
              State(5, 0.11)]
    economy = SinglePeriodEconomy(States(states, [0.1, 0.3, 0.3, 0.25, 0.05]))
    gross_risk_free_rate = economy.risk_free_rate + 1
    risk_free_rate = gross_risk_free_rate - 1
    print(f"Risk free rate: {risk_free_rate:.4%}")
    firm = Asset(States(states, [120, 110, 100, 95, 60]))
    print("Firm asset values:")
    print(firm)
    legacy_debt = DebtPariPassu(firm, face_value=80)
    print("Legacy debt")
    equity = Equity(firm, debt_face_value=legacy_debt.face_value)



if __name__ == "__main__":
    main()
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
    new_debt = DebtPariPassu(new_firm_assets, face_value=option_price)

    total_face_value = debt.face_value + new_debt.face_value
    return Firm(new_firm_assets, debt_face_value=total_face_value)


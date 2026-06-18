import pandas as pd

def get_sample_input():
    # minimal valid structure with ALL required columns
    return pd.DataFrame([{
        "AMT_INCOME_TOTAL": 0,
        "AMT_CREDIT": 0,
        "AMT_ANNUITY": 0,
        "DAYS_BIRTH": 0,
        "DAYS_EMPLOYED": 0,
        "EXT_SOURCE_1": 0,
        "EXT_SOURCE_2": 0,
        "EXT_SOURCE_3": 0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y"
        # add only required columns used by pipeline
    }])
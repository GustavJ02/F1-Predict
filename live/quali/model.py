"""Qualifying model for the runner: Q1/Q2 predict where the cutoff will end up, Q3 who takes pole.

Use it with:  MODEL=live.quali.model ./run_session.sh 2026 Austin quali
"""
import pandas as pd
from live.quali.predict_elimination import predict as predict_cutoff
from live.quali.predict_pole import predict as predict_pole


def predict(state, history, info):
    part = info["session_part"]
    if pd.isna(part):
        return None
    part = int(part)
    if part == 3:       # no elimination: pole probabilities for the three most likely drivers
        res = predict_pole(history, part, as_of=info["now"], verbose=False)
        if res is None:
            return None
        row = dict(part=part, current=res["current"], n_pending=res["n_pending"])
        for i, t in enumerate(res["top3"], 1):
            row[f"pole{i}"], row[f"p{i}"] = t["tla"], t["prob"]
        return row
    res = predict_cutoff(history, part, as_of=info["now"], verbose=False)
    if res is None:
        return None
    return dict(part=part, current=res["current"], pred=res["pred"],
                lo=res["lo"], hi=res["hi"], n_pending=res["n_pending"])

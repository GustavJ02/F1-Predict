"""Qualifying model for the runner: predicts where the Q1/Q2 cutoff will end up.

Use it with:  MODEL=quali.model ./run_session.sh 2026 Austin quali
"""
import pandas as pd
from quali.predict_elimination import predict as predict_cutoff


def predict(state, history, info):
    part = info["session_part"]
    if pd.isna(part) or int(part) == 3:        # Q3 has no elimination
        return None
    res = predict_cutoff(history, int(part), as_of=info["now"], verbose=False)
    if res is None:
        return None
    return dict(part=int(part), current=res["current"], pred=res["pred"],
                lo=res["lo"], hi=res["hi"], n_pending=res["n_pending"])

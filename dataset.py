import numpy as np
import pandas as pd 
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import io
import requests

def load_dataset(ticker):
    data = pd.read_csv(f"{ticker}.csv", index_col=0, parse_dates=True)
    return data


def download():
    BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    FILES = ["DEMO_J", "BMX_J", "BPX_J", "BIOPRO_J", "SMQ_J"]
    headers = {"User-Agent": "Mozilla/5.0"}

    def scarica(nome):
        r = requests.get(f"{BASE}{nome}.xpt", headers=headers, timeout=60)
        r.raise_for_status()
        # I file XPT iniziano con questa intestazione; se manca, è arrivato altro (es. HTML)
        if not r.content.startswith(b"HEADER RECORD"):
            raise ValueError(
                f"{nome}: la risposta non è un file XPT. Inizio contenuto: {r.content[:200]!r}"
            )
        return pd.read_sas(io.BytesIO(r.content), format="xport")

    demo, bmx, bpx, bio, smq = [scarica(f) for f in FILES]

    # Unione sulla chiave paziente SEQN
    df = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    for altro in (bmx[["SEQN", "BMXBMI"]],
                bpx[["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]],
                bio[["SEQN", "LBXSNASI"]],
                smq[["SEQN", "SMQ020", "SMQ040"]]):
        df = df.merge(altro, on="SEQN", how="inner")

    # Variabili del modello
    out = pd.DataFrame()
    out["arterial_pressure"] = df[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1)
    out["age"] = df["RIDAGEYR"]
    out["BMI"] = df["BMXBMI"]
    out["sodium"] = df["LBXSNASI"]

    # Dummy fumatore: 1 = fuma attualmente (ogni giorno o qualche giorno), 0 = altrimenti
    out["smoker"] = ((df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))).astype(float)
    out.loc[~df["SMQ020"].isin([1, 2]), "smoker"] = float("nan")

    # Dummy genere: 1 = maschio, 0 = femmina
    out["gender"] = (df["RIAGENDR"] == 1).astype(int)

    # Solo adulti, senza dati mancanti
    out = out[df["RIDAGEYR"] >= 18].dropna()
    out["smoker"] = out["smoker"].astype(int)

    out.to_csv("nhanes_regression.csv", index=False)
    print(out.shape)
    print(out.head())
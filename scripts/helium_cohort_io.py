from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def read_plan():
    paths=[ROOT/'data/helium_spectrum_download_plan.csv',ROOT/'data/mixed_helium_spectrum_download_plan.csv',ROOT/'data/helium_control_spectrum_download_plan.csv']
    return pd.concat([pd.read_csv(p) for p in paths if p.exists()],ignore_index=True).drop_duplicates('spec_file')

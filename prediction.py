# %%
import pandas as pd

total_data = pd.read_csv(
    "combined_energy_price_clean.csv",
    sep=";",            # German-style separator
    decimal=",",        # convert 1,23 -> 1.23
    low_memory=False     # avoids DtypeWarning

)
total_data
# %%

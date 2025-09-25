# %%
import pandas as pd

wind_data = pd.read_csv(
    "Hochrechnung Windenergie [2025-09-24 09-11-23].csv",
    sep=";",            # German-style separator
    decimal=",",        # convert 1,23 -> 1.23
    low_memory=False     # avoids DtypeWarning

)

print(wind_data.head())

# %%

solar_data = pd.read_csv(
    "Hochrechnung Solarenergie [2025-09-24 09-07-35].csv",
    sep=";",            # German-style separator
    decimal=",",        # convert 1,23 -> 1.23
    low_memory=False

)

print(solar_data.head())
# %%

spot_price = pd.read_csv(
    "Spotmarktpreis [2025-09-24 09-21-25].csv",
    sep=";",            # German-style separator
    decimal=",",        # convert 1,23 -> 1.23
)

print(spot_price.head())
# %%


tso_cols = ["50Hertz", "Amprion", "TenneT TSO", "TransnetBW"]

for col in tso_cols:
    # Replace comma with dot, then convert to float
    wind_data[col] = wind_data[col].astype(str).str.replace(',', '.')
    wind_data[col] = pd.to_numeric(wind_data[col], errors='coerce')

# Create total column
wind_data["Total_Wind"] = wind_data[tso_cols].sum(axis=1)

# Keep only Zeit and total
wind_total = wind_data[["Zeit", "Total_Wind"]]

print(wind_total.head())

# %%


# Automatically detect numeric columns for summing, excluding Zeit/Zeitzone
exclude_cols = ["Zeit", "Zeitzone"]  # adjust if other non-numeric columns exist
solar_cols = [col for col in solar_data.columns if col not in exclude_cols]

# Convert all columns to numeric (fix commas just in case)
for col in solar_cols:
    solar_data[col] = solar_data[col].astype(str).str.replace(',', '.')
    solar_data[col] = pd.to_numeric(solar_data[col], errors='coerce')

# Sum all solar columns into a total column
solar_data["Total_Solar"] = solar_data[solar_cols].sum(axis=1)

# Keep only Zeit and total
solar_total = solar_data[["Zeit", "Total_Solar"]]

print(solar_total.head())
# %%

# Merge wind and solar totals on 'Zeit'
energy_total = pd.merge(wind_total, solar_total, on="Zeit", how="outer")

print(energy_total.head())
# %%


price_data = pd.read_csv(
    "Spotmarktpreis [2025-09-24 09-21-25].csv",
    sep=";",            # German-style separator
    decimal=",",        # convert 1,23 -> 1.23
    low_memory=False     # avoids DtypeWarning

)

print(price_data.head())
# %%
# Convert spot price from ct/kWh to €/MWh
price_data["Price_MWh"] = price_data["Spotmarktpreis in ct/kWh"] * 10

print(price_data[["Datum", "von", "Zeitzone von", "Spotmarktpreis in ct/kWh", "Price_MWh"]].head())

# %%
print(energy_total.head())

# %%

# Step 1: Convert columns to datetime
price_data["Datetime"] = pd.to_datetime(price_data["Datum"] + " " + price_data["von"], dayfirst=True)
energy_total["Datetime"] = pd.to_datetime(energy_total["Zeit"], dayfirst=True)

# Step 2: Merge on the new datetime column
combined_data = pd.merge(
    energy_total, 
    price_data[["Datetime", "Price_MWh"]], 
    on="Datetime", 
    how="left"
)

print(combined_data.head())
# %%
# Keep only rows where minutes are 0 (full hours)
combined_data = combined_data[combined_data["Datetime"].dt.minute == 0].copy()

# Optionally, drop the original columns if you only want Datetime and Price_MWh
combined_data = combined_data[["Datetime", "Total_Wind", "Total_Solar", "Price_MWh"]]

print(combined_data.head())
# %%
# Save combined_data to a CSV file
combined_data.to_csv("combined_energy_price.csv", index=False, sep=";")  # use ; as separator if German-style

print("CSV file saved as 'combined_energy_price.csv'.")

# %%

combined_data.head()
# %%
combined_data.isna().any().any()
# %%
combined_data.isna().sum()
# %%
combined_data
# %%
# Remove rows with any NaN values
combined_data = combined_data.dropna().reset_index(drop=True)
# %%
combined_data
# %%
combined_data.head()
# %%
combined_data.to_csv("combined_energy_price_clean.csv", index=False, sep=";")  # use ; as separator if German-style

# %%
##    streamlit run dashboard.py


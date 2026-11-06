
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------
# 1. PROJECT SETUP
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = BASE_DIR / "Unemployment in India.csv"

sns.set_theme(style="whitegrid")

print("=" * 55)
print("       UNEMPLOYMENT ANALYSIS IN INDIA")
print("=" * 55)

# --------------------------------
# 2. LOAD DATASET
# --------------------------------

df = pd.read_csv(DATA_FILE)

# Clean column names
df.columns = df.columns.str.strip()

# Remove unwanted spaces in text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Convert date and numerical columns
df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=True, errors="coerce"
)

numeric_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing essential values
df.dropna(
    subset=["Region", "Date"] + numeric_columns,
    inplace=True
)

df["Month"] = df["Date"].dt.month_name()
df["Year"] = df["Date"].dt.year
df["Month_Number"] = df["Date"].dt.month

print("\nDataset loaded successfully!")
print("Total records:", len(df))
print("Total regions:", df["Region"].nunique())

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

# --------------------------------
# 3. BASIC STATISTICS
# --------------------------------

print("\nUnemployment Statistics:")
print(df["Estimated Unemployment Rate (%)"].describe())

average_rate = df["Estimated Unemployment Rate (%)"].mean()

print(f"\nAverage Unemployment Rate: {average_rate:.2f}%")

# --------------------------------
# 4. STATE-WISE ANALYSIS
# --------------------------------

region_data = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Regions with Highest Unemployment:")
print(region_data.head(10))

print("\nTop 10 Regions with Lowest Unemployment:")
print(region_data.tail(10))

plt.figure(figsize=(12, 6))
region_data.head(10).plot(kind="bar", color="tomato")

plt.title("Top 10 Regions with Highest Unemployment")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_unemployment_regions.png")
plt.show()

# --------------------------------
# 5. UNEMPLOYMENT TREND ANALYSIS
# --------------------------------

monthly_data = (
    df.groupby(df["Date"].dt.to_period("M"))[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

monthly_data.index = monthly_data.index.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_data.index,
    monthly_data.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_unemployment_trend.png")
plt.show()

# --------------------------------
# 6. COVID-19 IMPACT ANALYSIS
# --------------------------------

pre_covid = df[df["Date"] < "2020-04-01"]

during_covid = df[df["Date"] >= "2020-04-01"]

pre_rate = pre_covid["Estimated Unemployment Rate (%)"].mean()

covid_rate = during_covid["Estimated Unemployment Rate (%)"].mean()

print("\nCOVID-19 IMPACT ANALYSIS")
print("-" * 40)
print(f"Average before April 2020: {pre_rate:.2f}%")
print(f"Average from April 2020: {covid_rate:.2f}%")

if pre_rate > 0:
    change = ((covid_rate - pre_rate) / pre_rate) * 100
    print(f"Percentage change: {change:.2f}%")

plt.figure(figsize=(8, 5))

plt.bar(
    ["Before COVID-19", "During COVID-19"],
    [pre_rate, covid_rate],
    color=["skyblue", "salmon"]
)

plt.title("Impact of COVID-19 on Unemployment")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "covid_impact.png")
plt.show()

# --------------------------------
# 7. RURAL VS URBAN ANALYSIS
# --------------------------------

if "Area" in df.columns:

    area_data = (
        df.groupby("Area")["Estimated Unemployment Rate (%)"]
        .mean()
    )

    print("\nRural vs Urban Unemployment:")
    print(area_data)

    plt.figure(figsize=(7, 5))
    area_data.plot(kind="bar", color=["orange", "steelblue"])

    plt.title("Rural vs Urban Unemployment Rate")
    plt.xlabel("Area")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rural_urban_comparison.png")
    plt.show()

# --------------------------------
# 8. MONTHLY / SEASONAL ANALYSIS
# --------------------------------

seasonal_data = (
    df.groupby("Month_Number")["Estimated Unemployment Rate (%)"]
    .mean()
)

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(11, 5))

plt.plot(
    month_labels,
    seasonal_data.reindex(range(1, 13)).values,
    marker="o",
    color="purple"
)

plt.title("Monthly Seasonal Unemployment Patterns")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "seasonal_trends.png")
plt.show()

# --------------------------------
# 9. FINAL INSIGHTS
# --------------------------------

print("\n" + "=" * 55)
print("                 FINAL INSIGHTS")
print("=" * 55)

print(f"1. Overall average unemployment: {average_rate:.2f}%")

print(
    f"2. Highest unemployment region: {region_data.index[0]}"
)

print(
    f"3. Lowest unemployment region: {region_data.index[-1]}"
)

print(
    f"4. Average before COVID-19: {pre_rate:.2f}%"
)

print(
    f"5. Average during COVID-19: {covid_rate:.2f}%"
)

print("\nAll analysis completed successfully!")
print("Graphs saved inside the outputs folder.")
print("=" * 55)

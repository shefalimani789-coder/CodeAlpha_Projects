# ============================================================
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------

file_name = "Unemployment_Rate_upto_11_2020.csv"

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("ERROR: Dataset file not found!")
    print("Make sure the CSV file is in the same folder as this Python file.")
    exit()


print("=" * 60)
print("        UNEMPLOYMENT ANALYSIS WITH PYTHON")
print("=" * 60)


# ------------------------------------------------------------
# STEP 2: Display Basic Information
# ------------------------------------------------------------

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 3: Clean Column Names
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# STEP 4: Convert Date Column
# ------------------------------------------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

elif "Date" not in df.columns:
    print("\nDate column was not found.")
    print("Please check your CSV column names.")


# ------------------------------------------------------------
# STEP 5: Remove Duplicate Rows
# ------------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicates)

df = df.drop_duplicates()


# ------------------------------------------------------------
# STEP 6: Handle Missing Values
# ------------------------------------------------------------

df = df.dropna()

print("\nDataset after cleaning:")
print(df.shape)


# ------------------------------------------------------------
# STEP 7: Find Important Numeric Columns
# ------------------------------------------------------------

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# STEP 8: Overall Unemployment Rate
# ------------------------------------------------------------

if "Estimated Unemployment Rate (%)" in df.columns:

    average_unemployment = df["Estimated Unemployment Rate (%)"].mean()

    maximum_unemployment = df["Estimated Unemployment Rate (%)"].max()

    minimum_unemployment = df["Estimated Unemployment Rate (%)"].min()

    print("\nUnemployment Statistics:")
    print(
        f"Average Unemployment Rate: "
        f"{average_unemployment:.2f}%"
    )

    print(
        f"Maximum Unemployment Rate: "
        f"{maximum_unemployment:.2f}%"
    )

    print(
        f"Minimum Unemployment Rate: "
        f"{minimum_unemployment:.2f}%"
    )


# ------------------------------------------------------------
# STEP 9: Unemployment Trend Over Time
# ------------------------------------------------------------

if "Date" in df.columns and \
   "Estimated Unemployment Rate (%)" in df.columns:

    monthly_data = (
        df.groupby("Date")["Estimated Unemployment Rate (%)"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_data["Date"],
        monthly_data["Estimated Unemployment Rate (%)"],
        marker="o"
    )

    plt.title("Unemployment Rate Trend in India")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")

    plt.xticks(rotation=45)

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 10: State-wise Average Unemployment Rate
# ------------------------------------------------------------

if "Region" in df.columns and \
   "Estimated Unemployment Rate (%)" in df.columns:

    state_average = (
        df.groupby("Region")["Estimated Unemployment Rate (%)"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage Unemployment Rate by State/Region:")
    print(state_average)


    # Bar graph
    plt.figure(figsize=(12, 7))

    state_average.plot(kind="bar")

    plt.title("Average Unemployment Rate by State/Region")
    plt.xlabel("State/Region")
    plt.ylabel("Average Unemployment Rate (%)")

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 11: COVID-19 Impact Analysis
# ------------------------------------------------------------

if "Date" in df.columns and \
   "Estimated Unemployment Rate (%)" in df.columns:

    # Before COVID period
    before_covid = df[
        df["Date"] < "2020-03-01"
    ]

    # During COVID period
    during_covid = df[
        df["Date"] >= "2020-03-01"
    ]

    before_average = (
        before_covid["Estimated Unemployment Rate (%)"].mean()
    )

    during_average = (
        during_covid["Estimated Unemployment Rate (%)"].mean()
    )

    print("\nCOVID-19 Impact Analysis:")
    print("-" * 40)

    print(
        f"Average unemployment before March 2020: "
        f"{before_average:.2f}%"
    )

    print(
        f"Average unemployment from March 2020 onward: "
        f"{during_average:.2f}%"
    )

    change = during_average - before_average

    print(
        f"Change in average unemployment: "
        f"{change:.2f} percentage points"
    )


# ------------------------------------------------------------
# STEP 12: COVID-19 Comparison Graph
# ------------------------------------------------------------

if "Date" in df.columns and \
   "Estimated Unemployment Rate (%)" in df.columns:

    plt.figure(figsize=(12, 6))

    plt.plot(
        monthly_data["Date"],
        monthly_data["Estimated Unemployment Rate (%)"],
        marker="o"
    )

    # Mark the beginning of COVID period
    plt.axvline(
        pd.Timestamp("2020-03-01"),
        linestyle="--",
        label="COVID-19 Period Begins"
    )

    plt.title("Impact of COVID-19 on Unemployment Rate")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")

    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 13: Identify Highest Unemployment Period
# ------------------------------------------------------------

if "Date" in df.columns and \
   "Estimated Unemployment Rate (%)" in df.columns:

    highest_row = df.loc[
        df["Estimated Unemployment Rate (%)"].idxmax()
    ]

    print("\nHighest Unemployment Observation:")
    print(highest_row)


# ------------------------------------------------------------
# STEP 14: Identify Lowest Unemployment Period
# ------------------------------------------------------------

if "Date" in df.columns and \
   "Estimated Unemployment Rate (%)" in df.columns:

    lowest_row = df.loc[
        df["Estimated Unemployment Rate (%)"].idxmin()
    ]

    print("\nLowest Unemployment Observation:")
    print(lowest_row)


# ------------------------------------------------------------
# STEP 15: Correlation Analysis
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

if len(numeric_columns) > 1:

    correlation = df[numeric_columns].corr()

    print("\nCorrelation Matrix:")
    print(correlation)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Between Economic Indicators")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 16: Final Insights
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("                 KEY INSIGHTS")
print("=" * 60)

if "Estimated Unemployment Rate (%)" in df.columns:

    print(
        f"\n1. Average unemployment rate: "
        f"{average_unemployment:.2f}%"
    )

    print(
        f"2. Maximum recorded unemployment rate: "
        f"{maximum_unemployment:.2f}%"
    )

    print(
        f"3. Minimum recorded unemployment rate: "
        f"{minimum_unemployment:.2f}%"
    )

    if during_average > before_average:
        print(
            "4. Unemployment increased after the beginning "
            "of the COVID-19 period."
        )
    else:
        print(
            "4. The average unemployment rate was lower "
            "during the selected COVID-19 period."
        )

print(
    "\n5. State-wise analysis helps identify regions "
    "with relatively higher unemployment."
)

print(
    "6. Time-series analysis helps identify changes "
    "and trends in unemployment."
)

print(
    "7. These findings can help policymakers design "
    "employment-support and economic recovery measures."
)

print("\nAnalysis completed successfully!")
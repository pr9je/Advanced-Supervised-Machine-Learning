# Core libraries for this notebook 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10,5)
pd.set_option("display.max_columns", None)

# Shape of the dataset.
df.shape

# Data types of every column.
df.info()

# Statistical overview of numerical columns.

df.describe(include='all')

# convert true categorical columns to 'category' dtypes 

for col in ["company_hash", "job_position"]:
  df[col] = df[col].astype("category")

df.dtypes

# missing values per columns 
missing = df.isna().sum().to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
missing


# checking for duplicates learner records (same email appearing multiple times = multiple job updates.)

print("Total rows: ", len(df))
print("Unique learners (by email_has): ", df['email_hash'].nunique())
print("Learners with moer than one record: ", (df["email_hash"].value_counts() > 1).sum())


# orgyear should be realistrically fall between 1980 and the current CTC-update year,
valid_year_mask = df["orgyear"].between(1980,2021)

print(f"Rows with a plausibel orgyear: {valid_year_mask.sum() } / {len(df)}")
print(f"Rows dropped as invalid orgyear: {(~valid_year_mask).sum()}"
      f"({(~valid_year_mask).mean()*100:.2f} % of data)")

df_clean = df[valid_year_mask | df["orgyear"].isna()].copy()

# derive years_of_experience = ctc_updated_year - orgyear (engineered feature)

df_clean["years_of_experience"] = df_clean["ctc_updated_year"] - df_clean["orgyear"]

# A tiny number of rows can stil end up negative (orgyear after CTC update year)

df_clean = df_clean[(df_clean["years_of_experience"] >= 0) | (df_clean["years_of_experience"].isna())]
df_clean["years_of_experience"].describe()
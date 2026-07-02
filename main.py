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


# Fill missing job_position with an explicit "Unknown" category rather than dropping rows.

df_clean["job_position"] = df_clean["job_position"].cat.add_categories("Unknown").fillna("Unknown")
df_clean["job_position"].isna().sum()

# For plotiing convenience, cast the two categoricals baclt to plain straings.
# They have 1000s/10000s of categories; seaborn's hue=based grouping cn blow up-memory on the full-categpry index even when a plot only uses a handful of them.
df_clean["job_position"] = df_clean["job_position"].astype(str)
df_clean["company_hash"] = df_clean["company_hash"].astype(str)


# Univariate Analysis

# - We look at each variable in isolvation: distribution for numerica variables(`ctc`, `orgyear`,`year_of_experience`) and frequency for categorical variables (`job_position`, `company_hash`)

# CTC distribution (raw) 

fig, ax = plt.subplots(1,2,figsize=(14,5))
sns.histplot(df_clean["ctc"], bins=100, ax=ax[0])
ax[0].set_title("CTC distribution (raw, linear scale)")
ax[0].set_xlabel("CTC")

sns.histplot(np.log1p(df_clean["ctc"]), bins=100, ax=ax[1], color="orange")
ax[1].set_title("CTC distribution (log scale)")
ax[1].set_xlabel("log(1 + CTC)")
plt.tight_layout()
plt.show()


# years_of_experience distribution.

plt.figure(figsize=(10,5))
sns.histplot(df_clean["years_of_experience"].dropna(), bins=30, kde=True)
plt.title("Years of experience distribution")
plt.xlabel("Years of experience")
plt.show()


# job_position frequency (top 15)

top_positions = df_clean["job_position"].value_counts().head(15)
top_positions.index = top_positions.index.astype(str) # avoid category-dtype cartesian blow-up in seaborn

plt.figure(figsize=(10,6))
sns.barplot(x=top_positions.values, y = top_positions.index, hue=top_positions.index, palette="viridis", legend=False)
plt.title("Top 15 Job Positions by count")
plt.title("Number of learners")
plt.show()


# compan_hash frequency (top 15) -- companies are hashed, but frequency is still meaningful

plt.figure(figsize=(8,5))
sns.histplot(df_clean["orgyear"].dropna(), bins = 30)
plt.title("Org Join year distribution (cleaned)")
plt.xlabel("Org year")
plt.show()


# CTC vs Years of Experience (continous-continuous) -> scatter plot
plot_df = df_clean[(df_clean["ctc"] <= df_clean["ctc"].quantile(0.99)) & (df_clean["years_of_experience"] <= 20)]

plt.figure(figsize=(9,6))
sns.scatterplot(data=plot_df, x="years_of_experience", y="ctc", alpha=0.15, s=15)
plt.title("CTC vs Years of Experience (99th Percentile CTC cap, <=20 yrs exp, for readability)")
plt.xlabel("Years of Experience")
plt.ylabel("CTC")
plt.show()



# CTC vs Job Position (Categorical-continous)
top10_positions = df_clean["job_position"].value_counts().head(10).index
box_df = df_clean[df_clean["job_position"].isin(top10_positions) & (df_clean["ctc"] <= df_clean["ctc"].quantile(0.99))]

plt.figure(figsize=(11,6))
order = box_df.groupby("job_position", observed=True)["ctc"].median().sort_values(ascending=False).index
sns.boxplot(data=box_df, x="ctc", y="job_position", order=order, hue="job_position", palette="Set2", legend=False)
plt.title("CTC distribution by Job Position (top 10 positions, 99th pct CTC cap)")
plt.xlabel("CTC")
plt.show()


# CTC vs Company tier (categorical-continuous) -> compare top 10 companies by learner count
top10_companies = df_clean["company_hash"].value_counts().head(10).index
comp_df = df_clean[df_clean["company_hash"].isin(top10_companies) & (df_clean["ctc"] <= df_clean["ctc"].quantile(0.99))]

plt.figure(figsize=(11, 6))
order = comp_df.groupby("company_hash", observed=True)["ctc"].median().sort_values(ascending=False).index
sns.boxplot(data=comp_df, x="ctc", y="company_hash", order=order, hue="company_hash",
            palette="coolwarm", legend=False)
plt.title("CTC distribution across the 10 most-represented companies (99th pct CTC cap)")
plt.xlabel("CTC")
plt.show()

# Job Position vs Company (categorical-categorical) -> crosstab heatmap, top 8 x top 8 for readability
top8_positions = df_clean["job_position"].value_counts().head(8).index
top8_companies = df_clean["company_hash"].value_counts().head(8).index

ct = pd.crosstab(
    df_clean[df_clean["company_hash"].isin(top8_companies)]["company_hash"],
    df_clean[df_clean["job_position"].isin(top8_positions)]["job_position"]
)
ct = ct.loc[top8_companies, ct.columns.intersection(top8_positions)]

plt.figure(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
plt.title("Learner count: Top companies x Top job positions")
plt.show()


# years of Experience vs Job Position (categorical -continuous)
exp_box_df = df_clean[df_clean["job_position"].isin(top10_positions) & (df_clean["years_of_experience"] <= 20)]

plt.figure(figsize=(11,6))
order = exp_box_df.groupby("job_position", observed=True)["years_of_experience"].median().sort_values(ascending=False).index
sns.boxplot(data=exp_box_df, x="years_of_experience", y = "job_position", order=order, hue="job_position", palette="Spectral", legend=False)
plt.title(" Years of Experience by Job Position (top 10 positions)")
plt.xlabel("years of Experience")

# IQR-based outliers detectioon for CTC
Q1, Q3 = df_clean["ctc"].quantile([0.25,0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

n_outliers = ((df_clean["ctc"] < lower) | (df_clean["ctc"] > upper)).sum()
print(f"CTC IQR bounds: [{lower:,.0f}, {upper:,.0f}]")
print(f"Rows flagged as CTC outliers (1.5*IQR rule): {n_outliers} ({n_outliers / len(df)*100:.2f}%)")

plt.figure(figsize=(9,3))
sns.boxplot(x=df_clean["ctc"])
plt.title("CTC boxplot (raw) -- extreme right tail dominates the scale")
plt.show()

# Show the most extreme CTC values to eyeball plausibility.
df_clean.sort_values("ctc", ascending=False)[["company_hash", "job_position", "ctc", "orgyear"]].head(10)


# years_of_experience outliers.

plt.figure(figsize=(9,3))
sns.boxplot(x=df_clean["years_of_experience"])
plt.title("Years of Experience boxplot (cleaned orgyear)")
plt.show()

from scipy.stats import skew

for cols in ["ctc", "years_of_experience"]:
  s = skew(df_clean[cols].dropna())
  print(f"{cols}: skewness = {s:.2f}")

print(f"log1p(ctc): skewness = {skew(np.log1p(df_clean['ctc'].dropna())):.2f}")
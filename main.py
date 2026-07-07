# Core Libararies 

import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import skew

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
pd.set_option("display.max_columns",None)

# load the raw dataset
df = pd.read_csv('/content/scaler_hashed_for_students.csv')
df.head()

print("Shape: ", df.shape)
df.info()

df.describe(include="all").T

# Missing values.
missing = df.isna().sum().to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100 ).round(2)
missing

# Duplicate values
print("Fully duplicate rows:", df.duplicated().sum())
print("Duplicate on business key (email, company, orgyear, ctc, job_position):",
      df.duplicated(subset=["email_hash", "company_hash", "orgyear", "ctc", "job_position"]).sum())
print("Unique learners (email_hash):", df["email_hash"].nunique(), "vs total rows:", len(df))

# Univariate: CTC distribution (raw vs log)

fig, ax = plt.subplots(1,2, figsize=(14, 5))
sns.histplot(df["ctc"], bins=100, ax=ax[0], color="#4C72B0")
ax[0].set_title("CTC Distribution (Raw)")
ax[0].set_xlabel("CTC (INR)"); ax[0].set_ylabel("Count")

sns.histplot(np.log1p(df["ctc"]), bins=100, ax=ax[1], color="#DD8452")
ax[1].set_title("CTC Distribution (log1p scale)")
ax[1].set_xlabel("log(1 + CTC)"); ax[1].set_ylabel("Count")
plt.tight_layout(); plt.show()

print(f"CTC skewness (raw): {skew(df['ctc'].dropna()):.2f}")
print(f"CTC skewness (log1p): {skew(np.log1p(df['ctc'].dropna())):.2f}")


# Univariate: Orgyear distribution (raw, to expose the data-qulity problem)

plt.figure(figsize=(9,5))
sns.histplot(df["orgyear"].dropna(), bins=50, color="#55A868")
plt.title("Org Join year Distribution (Raw - before cleaning)")
plt.xlabel("Org Year"); plt.ylabel("Count")
plt.show()

print(df["orgyear"].describe())



# Univariate: Job Position frequency (top 15, including missingsness)

top_positions = df["job_position"].value_counts().head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_positions.values, y=top_positions.index, hue=top_positions.index, palette="viridis", legend=False)
plt.title("Top 15 Job Positions by Learner Count")
plt.xlabel("Number of Learners")
plt.ylabel("Job Position")
plt.show()

# Univariate: Company frequency (top 15)

top_companies = df["company_hash"].value_counts().head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=top_companies.values, y=top_companies.index, hue=top_companies.index, palette="mako", legend=False)
plt.title("Top 15 Companies by Learner Count")
plt.xlabel("Number of Learners"); plt.ylabel("Company (hashed)")
plt.show()

print("Total unique companies:", df["company_hash"].nunique())
print("Companies with only 1 learner:", (df["company_hash"].value_counts() == 1).sum())

# Bivariate: CTC vs Job Position (top 10 roles)
top10_roles = df["job_position"].value_counts().head(10).index
box_df = df[df["job_position"].isin(top10_roles) & (df["ctc"] <= df["ctc"].quantile(0.99))]
order = box_df.groupby("job_position", observed=True)["ctc"].median().sort_values(ascending=False).index

plt.figure(figsize=(11, 6))
sns.boxplot(data=box_df, x="ctc", y="job_position", order=order, hue="job_position", palette="Set2", legend=False)
plt.title("CTC Distribution by Job Position (Top 10 roles, 99th pct CTC cap for readability)")
plt.xlabel("CTC (INR)"); plt.ylabel("Job Position")
plt.show()

# Bivariate: CTC vs Years of Experience (using a quick unclean proxy for visualization only)
tmp = df.copy()
tmp["years_of_experience_raw"] = tmp["ctc_updated_year"] - tmp["orgyear"]
plot_df = tmp[(tmp["years_of_experience_raw"].between(0, 20)) &
              (tmp["ctc"] <= tmp["ctc"].quantile(0.99))]

plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_df, x="years_of_experience_raw", y="ctc", alpha=0.12, s=15, color="#4C72B0")
plt.title("CTC vs Years of Experience (99th pct CTC cap, 0-20 yrs)")
plt.xlabel("Years of Experience"); plt.ylabel("CTC (INR)")
plt.show()


# Bivariate: Job Position x Company heatmap (top 8 x top 8)
top8_roles = df["job_position"].value_counts().head(8).index
top8_companies = df["company_hash"].value_counts().head(8).index
ct = pd.crosstab(
    df[df["company_hash"].isin(top8_companies)]["company_hash"],
    df[df["job_position"].isin(top8_roles)]["job_position"]
).loc[top8_companies]
ct = ct[[c for c in top8_roles if c in ct.columns]]

plt.figure(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
plt.title("Learner Count: Top Companies x Top Job Positions")
plt.show()

# Data Cleaning:

df_clean = df.copy()
print("Starting rows:", len(df_clean))

# Invalid `orgyear` Handling.
valid_year = df_clean["orgyear"].between(1980, 2021)
invalid_count = (~valid_year & df_clean["orgyear"].notna()).sum()
print(f"Invalid orgyear values found: {invalid_count} ({invalid_count/len(df_clean)*100:.2f}% of rows)")

df_clean.loc[~valid_year, "orgyear"] = np.nan


# Missing Value Treatment
df_clean["job_position"] = df_clean["job_position"].fillna("Unknown")
df_clean["company_hash"] = df_clean["company_hash"].fillna("Unknown_Company")

print(df_clean[["job_position", "company_hash", "orgyear"]].isna().sum())


# Outlier Treatment (CTC)

LOWER_CTC, UPPER_CTC = 100_000, 15_000_000 # business-defined plausible ctc band (1L to 1.5Cr)

n_low = (df_clean["ctc"] < LOWER_CTC).sum()
n_high = (df_clean["ctc"] > UPPER_CTC).sum()

print(f"Rows below plausible CTC floor: {n_low} ({n_low/len(df_clean)*100:.2f}%)")
print(f"Rows above plausible CTC floor: {n_high} ({n_high/len(df_clean)*100:.2f}%)")

df_clean["ctc"] = df_clean["ctc"].clip(lower=LOWER_CTC, upper=UPPER_CTC)

# Visual check: CTC distribution after cleaning.

fig, ax = plt.subplots(1, 2, figsize=(14,5) )
sns.histplot(df_clean["ctc"], bins=100, ax=ax[0], color="#4C72B0")
ax[0].set_title("CTC Distribution (after winsorizing)")
sns.boxplot(x=df_clean["ctc"], ax=ax[1], color="#DD8452")
ax[1].set_title("CTC Boxplot (after winsorizing)")
plt.tight_layout(); plt.show()

# Data Consistency Checks.

print("Rows with CTC updated year before org year (impossible):",
      (df_clean["ctc_updated_year"] < df_clean['orgyear']).sum())
print("Negative or zero CTC:", (df_clean['ctc'] <= 0).sum())
print("Final shape after cleaning:", df_clean.shape)
df_clean.describe(include="all").T

# Feature Engineering

# year_of_experience

df_clean['years_of_experience'] = df_clean['ctc_updated_year'] - df_clean['orgyear']
df_clean.loc[df_clean['years_of_experience'] < 0, "years_of_experience"] = np.nan
df_clean["years_of_experience"].describe()

# Experience Bucker
def bucket_experience(y):
  if pd.isna(y):
    return "Unknown"
  if y < 2:
    return "Fresher (0-2y)"
  elif y < 5:
    return "Junior (2-5y)"
  elif y < 8:
    return "Mid (5-8y)"
  elif y < 12:
    return "Senior (8-12y)"
  else:
    return "Veteran (12y+)"

df_clean["experience_bucket"] = df_clean["years_of_experience"].apply(bucket_experience)
df_clean["experience_bucket"].value_counts()


# job_family and job_seniority

def map_job_family(pos):
  p = str(pos).lower()
  if "lead" in p or "manager" in p or "director" in p or "head" in p:
    return "Leadership"
  if "data scientist" in p or "data analyst" in p or "analytics" in p:
    return "Data"
  if "qa" in p or "sdet" in p or "test" in p:
    return "QA"
  if "devops" in p or "sre" in p or "infra" in p or "cloud" in p:
    return "DevOps/Infra"
  if "desgin" in p:
    return "Desgin"
  if "support" in p:
    return "Support"
  if "intern" in p:
    return "Intern"
  if "backend" in p or "frontend" in p or "fullstack" in p or "android" in p or "ios" in p or "engineer" in p or "developer" in p:
    return "Engineering"
  if p in ("nan", "unknown"):
    return "Unknown"
  return "Other" 

def map_job_seniority(pos):
  p = str(pos).lower()
  if "intern" in p:
    return 0
  if "lead" in p or "manager" in p or "director" in p or "head" in p or "leadership" in p:
    return 3
  if "senior" in p or "sr." in p or "sr" in p:
    return 2
  if p in ("nan", "unknown"):
    return -1 # unknown seniority, kept distinct from fresher
  return 1 # standard individual-contributor level.

df_clean["job_family"] = df_clean["job_position"].apply(map_job_family)
df_clean["job_seniority"] = df_clean["job_position"].apply(map_job_seniority)

print(df_clean['job_family'].value_counts())
print()
print(df_clean['job_seniority'].value_counts()) 


# company_employee_count, comapny_avg_ctc, company_tier
company_stats = df_clean.groupby("company_hash", observed=True).agg(
    company_employee_count = ("ctc", "size"),
    company_avg_ctc_raw = ("ctc", "mean")
)

global_mean_ctc = df_clean["ctc"].mean()
SHRINK_K = 10 # Shrinkage strength: companies with few samples pull towards global mean

company_stats["company_avg_ctc"] =(
    (company_stats["company_avg_ctc_raw"] * company_stats["company_employee_count"] + global_mean_ctc * SHRINK_K) /
    (company_stats["company_employee_count"] + SHRINK_K)
)

# Tier by quartile of the shrunk average CTC (only meaningful for companies with a few learners)
company_stats["company_tier"] = pd.qcut(
    company_stats["company_avg_ctc"], q=4,
    labels=["Tier 4 (Lower pay)", "Tier 3", "Tier 2", "Tier 1 (Top pay)"] 
)

df_clean = df_clean.merge(
    company_stats[["company_employee_count", "company_avg_ctc", "company_tier"]],left_on="company_hash", right_index=True, how="left"
)
df_clean[["company_employee_count", "company_avg_ctc", "company_tier"]].describe(include="all")


# Top_company_flag
df_clean["top_company_flag"] = (df_clean["company_tier"] == "Tier 1 (Top pay)").astype(int)
df_clean["top_company_flag"].value_counts()

# Avg_salary_by_role, median_salary, salary_percentile, salary_band, high_salary_flag.

role_avg_ctc = df_clean.groupby("job_position", observed=True)["ctc"].transform("mean")
df_clean["avg_salary_by_role"] = role_avg_ctc

df_clean["median_salary"] = df_clean["ctc"].median() #global reference point, used for salary bands below
df_clean["salary_percentile"] = df_clean["ctc"].rank(pct=True) *100

df_clean["salary_band"] = pd.qcut(
    df_clean["ctc"], q=4, labels=["Low", "Medium", "High", "Very High"]
)

df_clean["high_salary_flag"] = (df_clean["salary_percentile"] >= 75).astype(int)

df_clean[["avg_salary_by_role", "salary_percentile", "salary_band", "high_salary_flag"]].describe(include="all")


# Role_Frequency
df_clean["role_frequency"] = df_clean.groupby("job_position", observed=True)["job_position"].transform("count")
df_clean["role_frequency"].describe()


# Company_Growth

median_year = df_clean["orgyear"].median()
recent = df_clean[df_clean["orgyear"] >= median_year].groupby("company_hash", observed=True).size()
older = df_clean[df_clean["orgyear"] < median_year].groupby("company_hash", observed=True).size()

growth = ((recent - older) / (recent + older).replace(0, np.nan)).rename("company_growth_proxy")
df_clean = df_clean.merge(growth, left_on="company_hash", right_index=True, how="left")
df_clean["company_growth_proxy"] = df_clean["company_growth_proxy"].fillna(0)
df_clean["company_growth_proxy"].describe()

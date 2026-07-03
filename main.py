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

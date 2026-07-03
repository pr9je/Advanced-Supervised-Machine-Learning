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
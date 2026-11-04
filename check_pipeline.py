#%%
import pandas as pd
import numpy as np
from plotnine import *
from scipy.stats import spearmanr

#%%
# ---------------- load ----------------
mine = pd.read_csv("output/har_features.csv")            # 561 features + activity_Id + user_Id
theirs = pd.read_csv("dataset/Train/X_train.txt", sep=r"\s+", header=None)
y_tr = pd.read_csv("dataset/Train/y_train.txt", header=None)[0].values
s_tr = pd.read_csv("dataset/Train/subject_id_train.txt", header=None)[0].values
feat_names = pd.read_csv("dataset/features.txt", sep=r"\s+", header=None)[0].values
feat_names = [i.strip() for i in feat_names]
#%%

assert theirs.shape == (7767, 561), theirs.shape
theirs["activity_Id"], theirs["user_Id"] = y_tr, s_tr

# # ---------------- restrict mine to the authors' train subset ----------------
mine = mine[mine.user_Id.isin(np.unique(s_tr))].copy()
print(f"mine: {len(mine)} windows, theirs: {len(theirs)} windows")
#%%
# # ---------------- align windows: k-th window of each (user, activity) group ----------------
keys = ["user_Id", "activity_Id"]
for df in (mine, theirs):
    df["k"] = df.groupby(keys).cumcount()

cnt = (mine.groupby(keys).size().rename("mine")
       .to_frame().join(theirs.groupby(keys).size().rename("theirs"), how="outer").fillna(0))
bad = cnt[cnt.mine != cnt.theirs]
print(f"{len(bad)} of {len(cnt)} (user, activity) groups have different window counts:")
print(bad.astype(int).to_string() if len(bad) else "none")

good_idx = cnt[cnt.mine == cnt.theirs].index
mine = mine.set_index(keys + ["k"]).sort_index()
theirs = theirs.set_index(keys + ["k"]).sort_index()
keep = [i for i in mine.index if (i[0], i[1]) in good_idx]
mine, theirs = mine.loc[keep], theirs.loc[keep]
assert mine.index.equals(theirs.index)
print(f"comparing {len(mine)} aligned windows")

#%%
# ---------------- column-wise Spearman (rank, then Pearson) ----------------
A = mine.iloc[:, :561].rank().values
B = theirs.iloc[:, :561].values
B = pd.DataFrame(B).rank().values
A = (A - A.mean(0)) / A.std(0)
B = (B - B.mean(0)) / B.std(0)
with np.errstate(invalid="ignore", divide="ignore"):   # constant columns -> NaN
    rho = (A * B).mean(0)

res = pd.DataFrame({"idx": np.arange(1, 562), "feature": feat_names, "rho": rho})
res["family"] = res.feature.str.extract(r"^([A-Za-z]+)-?")[0]
res["domain"] = np.where(res.feature.str.startswith("t"), "time", "frequency")
res.to_csv("output/spearman_vs_uci.csv", index=False)

print(res.rho.describe())
print("NaN (constant) columns:", res.rho.isna().sum())
print("Worst 100 features:\n", res.nsmallest(100, "rho")[["idx", "feature", "rho"]].to_string(index=False))

#%%
# ---------------- plots ----------------
p1 = (ggplot(res.dropna(), aes("rho"))
      + geom_histogram(bins=50, fill="steelblue", color="white")
      + geom_vline(xintercept=res.rho.median(), linetype="dashed", color="firebrick")
      + labs(x="Spearman ρ (mine vs. UCI X_train)", y="Number of features",
             title="Agreement per feature (dashed = median)")
      + theme_bw())
p1.save("output/rho_hist.png", width=7, height=4.5, dpi=200)

p2 = (ggplot(res.dropna(), aes("family", "rho"))
      + geom_boxplot(outlier_alpha = 0)
      + geom_jitter(aes(color="domain"), width=0.15, size=1, alpha=0.6)
      + coord_flip()
      + labs(x="", y="Spearman ρ", color="Domain",
             title="Agreement by signal family")
      + theme_bw())
p2.save("output/rho_by_family.png", width=7.5, height=7, dpi=200)
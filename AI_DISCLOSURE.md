For the assignment I prepared, I got help from Chatgpt on several topics such as how to organize loops, how to plot model results, and how to evaluate clustering results and I used the code it provided below.

Codes;

How do you calculate and plot the means and standard deviations of data?
    for name in ["unscaled", "scaled"]:
        part = summary[
            (summary["algorithm"] == alg) &
            (summary["representation"] == name)
        ].set_index("parameter").loc[params]

        plt.errorbar( params, part["val_f1_mean"],
            yerr=part["val_f1_std"],
            marker="o", capsize=4, label=name + " validation" )
--------------------------------------------------------------------------------
Run k-means with different seed values and print the silhouette values.
km_models = {}
km_rows = []

for s in seeds:
    for name, data in [("unscaled", unscaled), ("scaled", scaled)]:
        x = data[s]["train"]
        v = data[s]["val"]
        km = KMeans(n_clusters=3, random_state=s, n_init=10)
        km.fit(x)
        pred = km.predict(v)
        n = len(set(pred))
        ari = adjusted_rand_score(y_val[s], pred)
        sil = silhouette_score(v, pred) if 1 < n < len(v) else np.nan

        km_models[(name, s)] = km
        km_rows.append([ "wine_kmeans_" + name + "_s" + str(s),"Wine", name, s, 3, km.inertia_, ari, sil])

km_df = pd.DataFrame(km_rows, columns=["id", "dataset", "representation", "seed","clusters", "train_inertia", "val_ari", "val_silhouette"])
display(km_df)

for name in ["unscaled", "scaled"]:
    km = km_models[(name, 42)]
    x = X_test if name == "unscaled" else X_test_scaled

    pred = km.predict(x)
    km_preds[name] = pred
    ari = adjusted_rand_score(y_test, pred)
    sil = silhouette_score(x, pred) if 1 < len(set(pred)) < len(x) else np.nan
    counts = pd.Series(pred).value_counts()

    km_test.append([ "Wine", name, 42, ari, sil,counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)])

----------------------------------------------------------------------------------------------------
How can I calculate the Pearson correlations between the features in a dataset and display them as a heatmap using Seaborn?
corr = X_bc.corr(method="pearson")
plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1)
-----------------------------------------------------------------------
How do you save files as CSV in JupyterLab?
results_df.to_csv("wine_classifier_runs.csv", index=False)
summary.to_csv("wine_classifier_summary.csv", index=False)
final_table.to_csv("wine_classifier_test.csv", index=False)

km_df.to_csv("wine_kmeans_runs.csv", index=False)
km_summary.to_csv("wine_kmeans_summary.csv", index=False)
km_test_df.to_csv("wine_kmeans_test.csv", index=False)

print("6 wine csv files saved")

bc_runs = pd.read_csv("bc_classifier_runs.csv")
bc_summary = pd.read_csv("bc_classifier_summary.csv")
bc_test = pd.read_csv("bc_classifier_test.csv")

bc_km_runs = pd.read_csv("bc_kmeans_runs.csv")
bc_km_summary = pd.read_csv("bc_kmeans_summary.csv")
bc_km_test = pd.read_csv("bc_kmeans_test.csv")

wine_runs = pd.read_csv("wine_classifier_runs.csv")
wine_summary = pd.read_csv("wine_classifier_summary.csv")
wine_test = pd.read_csv("wine_classifier_test.csv")

wine_km_runs = pd.read_csv("wine_kmeans_runs.csv")
wine_km_summary = pd.read_csv("wine_kmeans_summary.csv")
wine_km_test = pd.read_csv("wine_kmeans_test.csv")

bc_summary["dataset"] = "Breast Cancer"
wine_summary["dataset"] = "Wine"

print("all csv files loaded")
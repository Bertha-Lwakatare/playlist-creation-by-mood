# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


from sklearn import set_config
set_config(transform_output='pandas')

import os
print(os.getcwd())

# Data Importing, Exploration and Cleaning
spotify5000_df = pd.read_csv('3_spotify_5000_songs.csv')
spotify5000_df.columns                                                                                                           # column names have white space
spotify5000_df.columns = spotify5000_df.columns.str.strip()                                                                      # strip white space from column names
spotify5000_df.info()
spotify5000_df['Unnamed: 0']                                                                                                     # not necessary to keep
spotify5000_df['name'] = spotify5000_df['name'].str.strip()                                                                      # strip white space from song names
spotify5000_df['name'] = spotify5000_df['name'] + ' - ' + spotify5000_df['artist']                                               # concatente name of song and arist
spotify5000_df = spotify5000_df.set_index('name')                                                                                # change the index of the df to song name concatenated with artist name
spotify5000_df['type'].head(50)
spotify5000_df['type'].unique()                                                                                                  # there seems to only be one uniwue value with spaces so no information to be used in here
spotify5000_df = spotify5000_df.drop(columns=['id','time_signature','mode','key','Unnamed: 0','artist','type'])                  # drop unnecessary columns which do not inform on meaningful musical/ mood features (for now I will keep html)
spotify5000_df = spotify5000_df.drop_duplicates()                                                                                # drop the duplicates 75 were found
spotify5000_df_clean = spotify5000_df.drop(columns=['html'])                                                                     # drop html as it is not a value to use in analysis
spotify5000_df_clean.head(20)

# Scaling the data
scaler = MinMaxScaler()                                                                                                          # select the scaler to use
spotify5000_df_scaled = scaler.fit_transform(spotify5000_df_clean)                                                               # fit and transform
spotify5000_df_scaled.head(10)                                                                                                   # preview the data

# K means application
random_seed = 1234
kmeans = KMeans(n_clusters = 52,                                                                                                 # explore clusters by instinct
                      random_state = random_seed,
                      n_init='auto')

kmeans.fit(spotify5000_df_scaled)                                                                                                # fit the model to the data
cluster = kmeans.labels_                                                                                            
results_df = spotify5000_df_scaled.copy()
results_df["cluster"] = cluster
results_df.sort_values(by="cluster")
counts = results_df["cluster"].value_counts().sort_index()
print(counts.max(), counts.min())                                                                                                # print to see how many songs minimum and maximum playlists in this n_clusters has


# try using a loop to see if there is n_clusters that meets the boundaries
n_values = []
min_values = []
max_values = []

for n in range(35, 90):
    kmeans = KMeans(n_clusters=n,
                     random_state=random_seed,
                     n_init='auto')
    
    kmeans.fit(spotify5000_df_scaled.drop(columns=["cluster"], errors="ignore"))


    
    cluster = kmeans.labels_
    counts = pd.Series(cluster).value_counts()
    
    min_count = counts.min()
    max_count = counts.max()
    
    n_values.append(n)
    min_values.append(min_count)
    max_values.append(max_count)

plt.figure(figsize=(11, 6.5), dpi=150)
ax = plt.gca()

ax.plot(n_values, min_values, label="Min songs per playlist",
        color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
ax.plot(n_values, max_values, label="Max songs per playlist",
        color="#ff7f0e", linewidth=2.5, marker="o", markersize=4)

ax.axhspan(50, 250, color="green", alpha=0.06, zorder=0)
ax.axhline(50, color="green", linestyle="--", linewidth=1.5, label="Lower business boundary (50)")
ax.axhline(250, color="red", linestyle="--", linewidth=1.5, label="Upper business boundary (250)")

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("#333333")
    ax.spines[spine].set_linewidth(1.2)

ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.7)
ax.set_axisbelow(True)  # grid behind the lines/markers
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.tick_params(axis="both", which="major", labelsize=11, length=5, color="#333333")
ax.set_xlabel("Number of playlists (k)", fontsize=13, labelpad=10)
ax.set_ylabel("Playlist size (songs)", fontsize=13, labelpad=10)
#ax.set_title("Cluster Size vs. Business Requirements (k = 35–90)",
#             fontsize=15, fontweight="bold", pad=15)
ax.legend(fontsize=10.5, frameon=True, framealpha=0.9, loc="upper right")

plt.tight_layout()
plt.savefig("images/cluster_size_vs_requirements.png", dpi=300, bbox_inches="tight")
plt.show()


# Try elbow method to observe if there is any k-value
inertia_list = []
i_values = []

for i in range(35,90):
    spotify5000_df_scaled_kmeans = KMeans(n_clusters=i)
    spotify5000_df_scaled_kmeans.fit(spotify5000_df_scaled)
    inertia_list.append(round(spotify5000_df_scaled_kmeans.inertia_))

    i_values.append(i)

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

ax.plot(i_values, inertia_list, color="#1f77b4", linewidth=2.5)

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("#333333")
    ax.spines[spine].set_linewidth(1.2)


ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.7)
ax.set_axisbelow(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.tick_params(axis="both", which="major", labelsize=11, length=5, color="#333333")
ax.set_xlabel("k (number of clusters)", fontsize=13, labelpad=10)
ax.set_ylabel("Inertia", fontsize=13, labelpad=10)
#ax.set_title("Elbow Method Showing the Optimal k", fontsize=15, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("images/elbow_method_optimal_k.png", dpi=300, bbox_inches="tight")
plt.show()

# Check silhouette score
sil_scores = []
k_values = []

for k in range(35, 90):

    kmeans = KMeans(n_clusters = k,
                    random_state = random_seed,
                    n_init='auto')
    kmeans.fit(spotify5000_df_scaled)
    labels = kmeans.labels_
    sil_score = silhouette_score(spotify5000_df_scaled, labels)
    sil_scores.append(sil_score)
    k_values.append(k)

sns.set_theme(style='ticks')

g = sns.relplot(y=sil_scores,
                 x=k_values,
                 kind='line',
                 color="#1f77b4",
                 linewidth=2.5,
                 height=6.5,
                 aspect=1.7)

ax = g.ax

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("#333333")
    ax.spines[spine].set_linewidth(1.2)


ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.7)
ax.set_axisbelow(True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.tick_params(axis="both", which="major", labelsize=11, length=5, color="#333333")
g.set_axis_labels("Number of clusters", "Silhouette score", fontsize=13)
#ax.set_title(f"Silhouette Score from {k_values[0]} to {k_values[-1]} Clusters",
#             fontsize=15, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("images/silhouette_score_by_k.png", dpi=300, bbox_inches="tight")
plt.show()

# Create a radar chart
scatter_objects = []

categories = results_df.columns.drop("cluster")

cluster_feat_means = results_df.groupby(by="cluster").mean()

for cluster in sorted(results_df['cluster'].unique()):                                                                          # iterate over the unique clusters and add an object for each cluster to the list
    cluster_means = cluster_feat_means.loc[cluster, categories]                                                                 # find the mean value for each column of the cluster (only feature columns)

    cluster_scatter = go.Scatterpolar(
        r=cluster_means,
        theta=categories,
        fill='toself',
        name=f'Cluster {cluster}'
    )

    scatter_objects.append(cluster_scatter)

fig = go.Figure()                                                                                                               # create the figure (the white area)
fig.add_traces(scatter_objects)

fig.update_layout(
    title_text='Radar chart of Music Features Per Playlist',
    height=600,
    width=800,
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True
)

fig.show()

# Observe the variane of features
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

cluster_feat_means.var().sort_values(ascending=False).plot(
    kind='bar', ax=ax, color="#1f77b4", width=0.7
)

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("#333333")
    ax.spines[spine].set_linewidth(1.2)

ax.grid(True, which="major", axis="y", alpha=0.3, linewidth=0.7)
ax.set_axisbelow(True)  # grid behind the bars
ax.tick_params(axis="both", which="major", labelsize=11, length=5, color="#333333")
plt.xticks(rotation=45, ha="right")

ax.set_xlabel("Feature", fontsize=13, labelpad=10)
ax.set_ylabel("Variance of cluster means", fontsize=13, labelpad=10)
#ax.set_title("Feature Variance Across Cluster Means", fontsize=15, fontweight="bold", pad=15)

plt.tight_layout()
#plt.savefig("images/feature_variance_across_clusters.png", dpi=300, bbox_inches="tight")
plt.show()


# Clustering further based on musical mood features after clustering only to 8 playlists
k_broad = 8                                                              

kmeans_broad = KMeans(n_clusters=k_broad,
                       random_state=random_seed,
                       n_init='auto')

kmeans_broad.fit(spotify5000_df_scaled)                                  

results_df_broad = spotify5000_df_scaled.copy()
results_df_broad['broad_cluster'] = kmeans_broad.labels_
results_df_broad['broad_cluster'].value_counts().sort_index()

weights = {'valence': 1, 'energy': 1, 'acousticness': 1}
results_df_broad['mood_score'] = (
    results_df_broad['valence'] * weights['valence'] +
    results_df_broad['energy'] * weights['energy'] +
    results_df_broad['acousticness'] * weights['acousticness']
) / sum(weights.values())


target_size = 200                                                                                                                # songs per playlist plus oder minus           
playlist_labels = pd.Series(index=results_df_broad.index, dtype=object)

for cluster_id, group in results_df_broad.groupby('broad_cluster'):                                                              # loop over every broad cluster in results_df_broaad
    group_sorted = group.sort_values('mood_score')                                                                               # sort by the mood score for every group
    n_sub = max(1, round(len(group_sorted) / target_size))                                                                       # divides the number of songs in playlist to make sure we have target value. if number of songs is laes than target value then 1

    chunks = np.array_split(group_sorted.index, n_sub)                                                                           # splits one cluster into the number of sub categories

    for i, chunk_index in enumerate(chunks):
        playlist_labels.loc[chunk_index] = f"{cluster_id}_{i}"

results_df_broad['playlist'] = playlist_labels
results_df_broad['playlist'].nunique()                                                                                           # gives the number of new clusters

playlist_counts = results_df_broad['playlist'].value_counts().sort_index()
min_songs = playlist_counts.min()
max_songs = playlist_counts.max()
print(f'The least songs in a playlist is {min_songs} and the most in a playlist is {max_songs}')

# Find the average mood_score per playlist, then pick the highest and lowest scoring playlists
playlist_avg_mood = results_df_broad.groupby('playlist')['mood_score'].mean()

high_mood_playlist = playlist_avg_mood.idxmax()
low_mood_playlist = playlist_avg_mood.idxmin()

print(f"High mood playlist: {high_mood_playlist}  (avg score: {playlist_avg_mood.max():.3f})")
print(f"Low mood playlist: {low_mood_playlist}  (avg score: {playlist_avg_mood.min():.3f})")

results_df_broad[results_df_broad['playlist'] == high_mood_playlist].head(5)
results_df_broad[results_df_broad['playlist'] == low_mood_playlist].head(5)

# print a playlist as an example
playlist_label = '4_0'                                    
example_playlist = results_df_broad[results_df_broad['playlist'] == playlist_label]
example_playlist.to_csv(f'playlist_{playlist_label}.csv')
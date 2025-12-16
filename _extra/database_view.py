import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Load db file

db_path = "C:/Users/sebas/Downloads/02b369b7-c039-4c04-8221-09bd8871b5df_1761658779643.db"  # Replace with your database file path
output_dir = os.path.join(BASE_DIR, "csv_exports")
files = glob.glob(os.path.join(output_dir, "*.csv"))

dfs = {}

for file in files:
    filename = os.path.basename(file)   # <-- Clean key name
    dfs[filename] = pd.read_csv(file)
    print(f"Loaded {filename} ({len(dfs[filename])} rows)")

#moved file build up to stop rebuilding files each time
# files = glob.glob('*.csv', recursive=True)


#enhanced file name pull with Ai help (ChatGPT5)
#Now will look in specified directory without appending directory to name
#giving only CSV file name
#os walk probably easier but cool to learn about glob
#checks names specifically, breaks when used with dfs
file_name_check = [os.path.basename(f) for f in glob.glob(os.path.join(output_dir, "**/*.csv"), recursive=True)]

#make a folder, dont yell at me if its already there please
os.makedirs(output_dir, exist_ok=True)
conn = sqlite3.connect(db_path)

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
tables = [t[0] for t in tables]
#debug print found tables
# print("Tables found:", tables)

#change to CSV

for table in tables:

    csv_name = f"{table}.csv"
    print(f"{csv_name}")

    if csv_name in file_name_check:
        print(f" Skipping {table} already exported.") 
        continue


    df = pd.read_sql_query(f"SELECT * FROM {table}", conn) 

    csv_path = os.path.join(output_dir, f"{table}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Exported {table} to {csv_path}")

#close connection
conn.close()

#load files into df and read out head of each
print(f"files found {files}")

for file in files:
    name = file
    dfs[name] = pd.read_csv(file)
    print(f"Loaded {name} ({len(dfs[name])} rows)")

keys = dfs.keys()

for key in keys:
    print(f"\nHead and info of Key {key}")
    print(f"{dfs[key].head()}\n")
    print(f"{dfs[key].info()}")

print("\n")

# debug for keys
# print(dfs.keys())

# load data
df_playerSummaries = dfs["PlayerSummaries.csv"].copy()
df_cityInformation = dfs["CityInformations.csv"].copy()


# get players

player_cols = [f"Player{i}" for i in range(6)]

def determine_active_player(row):
    for i, col in enumerate(player_cols):
        if row[col] == 2:
            return i
    return None

df_playerSummaries["ActivePlayer"] = df_playerSummaries.apply(determine_active_player, axis=1)

# Only 4 players
df_playerSummaries = df_playerSummaries[df_playerSummaries["ActivePlayer"].isin([0, 1, 2, 3])]


# initial EDA on specific metrics vs turn

metrics = [
    "Cities", "Population", "Territory",
    "Gold", "GoldPerTurn",
    "HappinessPercentage",
    "SciencePerTurn", "CulturePerTurn",
    "FaithPerTurn", "TourismPerTurn",
    "Technologies"
]

# Ensure numeric
for m in metrics:
    df_playerSummaries[m] = pd.to_numeric(df_playerSummaries[m], errors="coerce")

def plot_metric(df, metric, use_scatter=False):
    plt.figure(figsize=(10, 5))
    
    for player in sorted(df["ActivePlayer"].unique()):
        sub = df[df["ActivePlayer"] == player]
        x = sub["Turn"]
        y = sub[metric]

        if use_scatter:
            plt.scatter(x, y, s=10, alpha=0.7, label=f"Player {player}")
        else:
            plt.plot(x, y, label=f"Player {player}")
    
    plt.title(f"{metric} vs Turn")
    plt.xlabel("Turn")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    # plt.show()

# for metric in metrics:
#     plot_metric(df_playerSummaries, metric, use_scatter=False)


# HEATMAP Correalation
df = df_playerSummaries


def plot_heatmap(data, title):
    corr = data.corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    # plt.show()

# Build per-player heatmaps
players = [0, 1, 2, 3]

for p in players:
    df_p = df_playerSummaries[df_playerSummaries["ActivePlayer"] == p][metrics].dropna()
    # plot_heatmap(df_p, f"Correlation Heatmap — Player {p}")

# Separate group heatmap: Players 2–4 (Normal AI)
df_normal_ai = df_playerSummaries[df_playerSummaries["ActivePlayer"].isin([1,2,3])][metrics].dropna()
# plot_heatmap(df_normal_ai, "Correlation Heatmap — Normal AI (Players 1 2 3)")


##Analyze CityInformation.csv

#plot city size by player per turn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

city_df = pd.read_csv("csv_exports/CityInformations.csv")

city_df["Turn"] = pd.to_numeric(city_df["Turn"], errors="coerce")
city_unique = city_df.drop_duplicates(subset=["Turn", "Name", "Owner"])

print(city_df["Owner"].unique()[:20])

# Cities per player per turn
cities_per_turn = (
    city_unique.groupby(["Turn", "Owner"])
               .size()
               .reset_index(name="CityCount")
)


# sns.lineplot(data=cities_per_turn, x="Turn", y="CityCount", hue="Owner")
# plt.title("Cities Controlled Per Turn")
# plt.show()

city_pivot = cities_per_turn.pivot(index="Turn", columns="Owner", values="CityCount")
print(city_pivot)

# Debug prints
# print(city_df.columns)
# strat_df = pd.read_csv("csv_exports/StrategyChanges.csv")
# print("\n")
# print(strat_df.columns)

## Analysis on city economics to player economics
# Average city yields per player per turn
player_cols = [f"Player{i}" for i in range(22)]  # Player0..Player21

def get_active_player(row):
    for i, col in enumerate(player_cols):
        if row[col] == 2:        # 2 = it's that player's turn
            return i
    return None

df_playerSummaries["ActivePlayer"] = df_playerSummaries.apply(get_active_player, axis=1)
print(df_playerSummaries["ActivePlayer"].unique())

city_player_cols = [f"Player{i}" for i in range(22)]

def get_city_owner(row):
    for i, col in enumerate(city_player_cols):
        if row[col] == 1:      # 1 = city is owned by this player
            return i
    return None

city_df["OwnerID"] = city_df.apply(get_city_owner, axis=1)
city_yields = (
    city_df
    .groupby(["Turn", "OwnerID"])[[
        "Population", "FoodPerTurn", "ProductionPerTurn",
        "GoldPerTurn", "SciencePerTurn", "CulturePerTurn",
        "FaithPerTurn", "TourismPerTurn",
        "BuildingCount", "WonderCount"
    ]]
    .mean()
    .reset_index()
)
df_playerSummaries["ActivePlayer"] = df_playerSummaries["ActivePlayer"].astype(int)
city_yields["OwnerID"] = city_yields["OwnerID"].astype(int)
df_playerSummaries["Turn"] = df_playerSummaries["Turn"].astype(int)
city_yields["Turn"] = city_yields["Turn"].astype(int)



merged = pd.merge(
    df_playerSummaries,
    city_yields,
    left_on=["Turn", "ActivePlayer"],
    right_on=["Turn", "OwnerID"],
    how="left"
)


# Detect players that appear in the match
valid_players = sorted(df_playerSummaries["ActivePlayer"].dropna().unique())

print("Detected valid players:", valid_players)

# Filter to only valid players
df_filtered = merged[merged["ActivePlayer"].isin(valid_players)].copy()

# Drop unused Player5+ columns
cols_to_drop = [
    c for c in df_filtered.columns 
    if c.startswith("Player") and int(c.replace("Player","")) not in valid_players
]
df_filtered = df_filtered.drop(columns=cols_to_drop)

# Build heatmap
# Clean numeric columns to include only meaningful gameplay features
keep_cols = [
    # index
    "Player0","Player1","Player2","Player3","Turn",

    # player-level (from PlayerSummaries)
    "Score", "Cities", "Population", "Territory",
    "Gold", "GoldPerTurn_x", "HappinessPercentage",
    "SciencePerTurn_x", "CulturePerTurn_x", "FaithPerTurn",
    "TourismPerTurn", "Technologies",

    # city-level (from city_yields)
    "FoodPerTurn", "ProductionPerTurn", "GoldPerTurn_y",
    "SciencePerTurn_y", "CulturePerTurn_y", "FaithPerTurn_y",
    "TourismPerTurn_y", "BuildingCount", "WonderCount",
]

# Build heatmap from df_filtered, NOT merged
df_corr = df_filtered[[c for c in keep_cols if c in df_filtered.columns]].copy()

# Generate correlation matrix
corr = df_corr.select_dtypes("number").corr()

# plt.figure(figsize=(15,12))
# sns.heatmap(corr, cmap="coolwarm", annot=False)
# plt.title("Correlation Heatmap — Cleaned and Corrected")
# plt.show()


## Clustering


# ============================================================
#                  STRATEGY CLUSTERING MODULE
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


# ------------------------------------------------------------
# 1. Metrics that actually exist in merged dataframe
# ------------------------------------------------------------
metrics = [
    # Player-level yields (PlayerSummaries → _x)
    "Cities",
    "Population_x",
    "Territory",
    "GoldPerTurn_x",
    "SciencePerTurn_x",
    "CulturePerTurn_x",
    "FaithPerTurn_x",

    # Removed (always 0)
    # "TourismPerTurn_x",

    # City-level yields (CityInfo → _y)
    "FoodPerTurn",
    "ProductionPerTurn",
    "GoldPerTurn_y",
    "SciencePerTurn_y",
    "CulturePerTurn_y",
    "FaithPerTurn_y",
    "TourismPerTurn_y",
    "BuildingCount",
    "WonderCount",
]



# ------------------------------------------------------------
# 2. FUNCTION: Build one-row-per-turn table for a single player
# ------------------------------------------------------------
def build_player_turn_table(player_id, merged, metrics):
    """Returns clean 1-row-per-turn summary for one player."""

    # Select rows for this player
    df_p = merged[merged["ActivePlayer"] == player_id].copy()

    # Keep only metrics that actually exist
    existing = [m for m in metrics if m in df_p.columns]

    # Group to ONE ROW per turn (averaging multiple city rows)
    df_turn = (
        df_p.groupby("Turn")[existing]
        .mean()
        .reset_index()
    )

    # Add Player label
    df_turn["Player"] = player_id

    # Fill NaNs → in early turns, faith/tourism = 0, cities = 0, etc
    df_turn = df_turn.fillna(0)
    

    return df_turn


# ------------------------------------------------------------
# 3. Build tables for all players (0–3)
# ------------------------------------------------------------
player_tables = []
for p in [0, 1, 2, 3]:
    df_player = build_player_turn_table(p, merged, metrics)
    player_tables.append(df_player)

df_all_players = pd.concat(player_tables, ignore_index=True)

print("Combined per-turn data:", df_all_players.shape)
print(df_all_players.head())


# ------------------------------------------------------------
# 4. CLUSTERING PER PLAYER
# ------------------------------------------------------------
def cluster_player(df_player, metrics, k=4):
    """Run KMeans clustering on one player's turn-level summary."""

    X = df_player[metrics].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    df_player["Cluster"] = kmeans.fit_predict(X_scaled)

    return df_player, kmeans, X_scaled


# ------------------------------------------------------------
# 5. Run clustering for Player 1 (LLM)
# ------------------------------------------------------------
df_p1 = df_all_players[df_all_players["Player"] == 0].copy()
df_p1, kmeans_p1, X_scaled_p1 = cluster_player(df_p1, metrics, k=4)


# ------------------------------------------------------------
# 6. SCATTER PLOT: Science vs Gold (Cluster Colors)
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_p1,
    x="SciencePerTurn_x",
    y="GoldPerTurn_x",
    hue="Cluster",
    palette="tab10"
)
plt.title("Player 1 (LLM) Clusters: Science vs Gold")
plt.show()


# ------------------------------------------------------------
# 7. BEHAVIORAL TIMELINE
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(df_p1["Turn"], df_p1["Cluster"], marker="o", linewidth=1)
plt.title("Behavioral State Timeline – Player 1 (LLM)")
plt.xlabel("Turn")
plt.ylabel("Cluster")
plt.show()


# ------------------------------------------------------------
# 8. HIERARCHICAL DENDROGRAM (OPTIONAL)
# ------------------------------------------------------------
Z = linkage(X_scaled_p1, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title("Player 1 – Hierarchical Clustering Dendrogram")
plt.xlabel("Turn Index")
plt.ylabel("Distance")
plt.show()

print("cluster centroid\n")
centroids = kmeans_p1.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=metrics)
print(centroids_df)


# ============================================================
# 9. DISPLAY CENTROID TABLE (nicely formatted)
# ============================================================
print("\n=== Cluster Centroids (Scaled Space) ===")
print(centroids_df.round(3))


# ============================================================
# 10. NORMALIZE CENTROIDS FOR RADAR CHART VISUALIZATION
# ============================================================
# Radar charts look best when all values scaled 0–1
centroids_norm = (centroids_df - centroids_df.min()) / (centroids_df.max() - centroids_df.min())

# categories for radar chart
categories = list(centroids_norm.columns)
N = len(categories)

# angle setup for radar
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # repeat first point to close the circle


# ============================================================
# 11. RADAR / SPIDER CHART FOR ALL CLUSTERS
# ============================================================
plt.figure(figsize=(10, 10))
plt.suptitle("Player 1 (LLM) – Cluster Profiles (Radar Chart)", fontsize=16)

for idx in range(len(centroids_norm)):
    values = centroids_norm.iloc[idx].tolist()
    values += values[:1]  # close the shape
    
    ax = plt.subplot(2, 2, idx + 1, polar=True)
    ax.plot(angles, values, linewidth=2, label=f"Cluster {idx}")
    ax.fill(angles, values, alpha=0.25)
    ax.set_title(f"Cluster {idx}", size=14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yticklabels([])  # cleaner look

plt.tight_layout()
plt.show()


print(df_p1[["GoldPerTurn_x", "GoldPerTurn_y", "Cities"]].corr())

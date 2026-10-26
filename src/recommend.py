import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = ["danceability", "energy", "tempo", "acousticness", "instrumentalness", "valence"]

def recommend(df, scaled, seed_index, count=19, use_cluster=True):
    seed = df.loc[seed_index]

    # 같은 무드(같은 군집)만 후보로 사용
    if use_cluster:
        candidates = df[(df["mood"] == seed["mood"]) & (df["cluster"] == seed["cluster"])]
    else:
        candidates = df[df["mood"] == seed["mood"]]
    candidates = candidates.drop(index=seed_index).copy()

    if len(candidates) == 0:
        return candidates.assign(similarity=[])

    candidates["similarity"] = cosine_similarity(
        scaled[[seed_index]], scaled[candidates.index.to_numpy()]
    ).flatten()
    candidates = candidates.sort_values("similarity", ascending=False, kind="stable")

    # 같은 곡(리마스터 포함)과 같은 아티스트 반복 제한
    selected = []
    used_titles = [str(seed["track_name"]).split(" - ")[0].lower()]
    artist_count = {str(seed["artists"]).split(";")[0]: 1}

    for index, row in candidates.iterrows():
        title = str(row["track_name"]).split(" - ")[0].lower()
        artist = str(row["artists"]).split(";")[0]

        if title in used_titles or artist_count.get(artist, 0) >= 2:
            continue

        selected.append(index)
        used_titles.append(title)
        artist_count[artist] = artist_count.get(artist, 0) + 1

        if len(selected) == count:
            break

    return candidates.loc[selected]


# 곡 사이 거리 (표준화된 Feature 기준 유클리드 거리)
def distance_matrix(vectors):
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def total_cost(order, D):
    return float(sum(D[order[i], order[i + 1]] for i in range(len(order) - 1)))

def nearest_neighbor(D):
    order = [0]
    remaining = list(range(1, len(D)))

    while remaining:
        last = order[-1]
        nearest = min(remaining, key=lambda i: D[last, i])
        order.append(nearest)
        remaining.remove(nearest)

    return order


# Seed는 고정하고 구간을 뒤집어서 비용이 줄면 적용
def two_opt(order, D):
    best = order[:]
    best_cost = total_cost(best, D)
    improved = True

    while improved:
        improved = False

        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_order = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                new_cost = total_cost(new_order, D)

                if new_cost < best_cost - 1e-9:
                    best = new_order
                    best_cost = new_cost
                    improved = True

    return best


# Seed + 추천곡을 순서 최적화해서 (플레이리스트, 최적화 비용, 유사도순 비용)을 반환
# method: "nn_2opt" 또는 "similarity"(유사도 높은 순서)
def make_playlist(df, scaled, seed_index, size=20, method="nn_2opt", use_cluster=True):
    if size < 2:
        raise ValueError("size는 2 이상이어야 합니다.")
    if method not in ("nn_2opt", "similarity"):
        raise ValueError("method는 nn_2opt 또는 similarity만 가능합니다.")

    rec = recommend(df, scaled, seed_index, size - 1, use_cluster)
    index_list = [seed_index] + rec.index.tolist()
    D = distance_matrix(scaled[index_list])

    similarity_order = list(range(len(index_list)))
    if method == "similarity":
        order = similarity_order
    else:
        order = two_opt(nearest_neighbor(D), D)

    playlist = df.loc[[index_list[i] for i in order]].copy()
    similarity = [1.0] + rec["similarity"].tolist()
    playlist["similarity"] = [similarity[i] for i in order]
    playlist.insert(0, "order", range(1, len(playlist) + 1))

    return playlist, total_cost(order, D), total_cost(similarity_order, D)
"""Explainable segmentation and PCA, using only NumPy array operations.

Run from the repository root with ``python part3.py``. The outcome and value
are used only to describe the finished segments, never to create them.
"""

import numpy as np

from common import load_data, write_csv, write_results


COHORT_DISTANCES = np.array(
    [
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ],
    dtype=float,
)


def standardise(features):
    """Use the full-data mean and sample SD of each continuous predictor."""
    means = features.mean(axis=0)
    deviations = features - means
    sample_sd = np.sqrt((deviations**2).sum(axis=0) / (len(features) - 1))
    if np.any(sample_sd == 0):
        raise ValueError("Cannot standardise a constant predictor.")
    return deviations / sample_sd, means, sample_sd


def kmeans(features, k):
    """Lloyd's assignment/mean loop with pinned starts and no iteration cap.

    np.argmin returns the first minimum, so exact distance ties go to the
    lower-numbered centroid. An empty cluster retains its previous centroid.
    The pass count includes the final assignment pass that changes nothing.
    """
    n = len(features)
    starting_rows = np.arange(k) * (n // k)
    centroids = features[starting_rows].copy()
    labels = np.full(n, -1, dtype=int)
    passes = 0

    while True:
        squared_distances = (
            (features[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(axis=2)
        new_labels = squared_distances.argmin(axis=1)
        passes += 1
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)

    wcss = ((features - centroids[labels]) ** 2).sum()
    return labels, centroids, float(wcss), passes


def choose_elbow(wcss):
    """Return the first K whose next improvement is less than half the last."""
    for k in (3, 4, 5):
        previous_drop = wcss[k - 1] - wcss[k]
        next_drop = wcss[k] - wcss[k + 1]
        if next_drop < 0.5 * previous_drop:
            return k
    raise ValueError("None of K=3, 4, 5 satisfies the supplied elbow rule.")


def name_cluster(members):
    """Render zero-based member indices in the required human-readable form."""
    return "+".join(f"E{member + 1}" for member in sorted(members))


def linkage_distance(distances, members_a, members_b, linkage):
    """Always use original member-pair distances, not distances of centroids.

    In particular, average linkage weights every original pair equally; it is
    not the unweighted average of the two previously merged cluster distances.
    """
    member_pairs = distances[np.ix_(members_a, members_b)]
    if linkage == "single":
        return float(member_pairs.min())
    if linkage == "complete":
        return float(member_pairs.max())
    if linkage == "average":
        return float(member_pairs.sum() / member_pairs.size)
    raise ValueError(f"Unknown linkage rule: {linkage}")


def agglomerative(distances, linkage):
    """Merge the closest active clusters, recording every height and partition."""
    n = len(distances)
    clusters = [(member,) for member in range(n)]
    partitions = [clusters.copy()]
    rows = []
    heights = []

    while len(clusters) > 1:
        best = None
        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                height = linkage_distance(
                    distances, clusters[i], clusters[j], linkage
                )
                candidate = (height, i, j)
                if best is None or candidate < best:
                    best = candidate

        height, i, j = best
        members_a, members_b = clusters[i], clusters[j]
        # Active clusters are sorted by smallest member, hence A precedes B.
        rows.append(
            (
                linkage,
                len(rows) + 1,
                name_cluster(members_a),
                name_cluster(members_b),
                height,
            )
        )
        heights.append(height)
        merged = tuple(sorted(members_a + members_b))
        clusters = [
            cluster
            for position, cluster in enumerate(clusters)
            if position not in (i, j)
        ]
        clusters.append(merged)
        clusters.sort(key=lambda members: members[0])
        partitions.append(clusters.copy())

    gaps = np.diff(heights)
    # Gap index 0 is between merges 1 and 2: exactly one merge lies below it.
    merges_below_cut = int(gaps.argmax()) + 1
    cut_partition = partitions[merges_below_cut]
    return rows, n - merges_below_cut, cut_partition


def pca_seven_steps(standardised):
    """Perform the seven required PCA steps, including deterministic signs."""
    # Step 1: the column mean vector is zero, up to floating-point arithmetic.
    mean_vector = standardised.mean(axis=0)
    # Step 2: explicitly centre, even though standardisation already did so.
    centred = standardised - mean_vector
    # Step 3: form the sample covariance matrix without a covariance helper.
    covariance = centred.T @ centred / (len(centred) - 1)
    # Step 4: use the permitted symmetric-matrix eigen-decomposition.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Step 5: reorder eigenpairs from the largest eigenvalue to the smallest.
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for column in range(eigenvectors.shape[1]):
        largest_entry = np.abs(eigenvectors[:, column]).argmax()
        if eigenvectors[largest_entry, column] < 0:
            eigenvectors[:, column] *= -1
    # Step 6: retain the top two eigenvectors as columns of W.
    weights = eigenvectors[:, :2]
    # Step 7: the two-column score matrix is Z = X W.
    scores = centred @ weights

    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    return {
        "mean_vector": mean_vector,
        "centred": centred,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "scores": scores,
        "explained": explained,
        "cumulative": cumulative,
        "components_for_80pct": int(np.flatnonzero(cumulative >= 0.8)[0] + 1),
    }


def main():
    predictors, outcome, value = load_data()
    continuous = predictors[:, :5]
    standardised, _, _ = standardise(continuous)
    fits = {k: kmeans(standardised, k) for k in range(2, 7)}
    wcss = {k: fit[2] for k, fit in fits.items()}
    results = {f"wcss_k{k}": wcss[k] for k in range(2, 7)}
    results["elbow_k"] = choose_elbow(wcss)

    # Use K=3 regardless of the elbow result; name segments by original X3.
    labels = fits[3][0]
    cluster_order = sorted(
        range(3), key=lambda cluster: continuous[labels == cluster, 2].mean()
    )
    segment_masks = {}
    segment_rows = []
    for segment, cluster in zip(("A", "B", "C"), cluster_order):
        mask = labels == cluster
        segment_masks[segment] = mask
        size = int(mask.sum())
        results[f"size_segment_{segment}"] = size
        segment_rows.append(
            (
                segment,
                size,
                *continuous[mask].mean(axis=0),
                outcome[mask].mean(),
                value[mask].mean(),
            )
        )

    hierarchical_rows = []
    for linkage in ("single", "complete", "average"):
        rows, clusters_at_cut, _ = agglomerative(COHORT_DISTANCES, linkage)
        hierarchical_rows.extend(rows)
        results[f"{linkage}_clusters_at_cut"] = clusters_at_cut

    pca = pca_seven_steps(standardised)
    eigenvalues = pca["eigenvalues"]
    results["eigenvalue_sum"] = eigenvalues.sum()
    for component in range(5):
        results[f"eigenvalue_{component + 1}"] = eigenvalues[component]
    for component in range(5):
        results[f"explained_ratio_{component + 1}"] = pca["explained"][component]
    for component in range(5):
        results[f"cumulative_ratio_{component + 1}"] = pca["cumulative"][component]
    results["components_for_80pct"] = pca["components_for_80pct"]
    for component in range(2):
        for feature in range(5):
            results[f"pc{component + 1}_loading_X{feature + 1}"] = (
                pca["eigenvectors"][feature, component]
            )
    for row in range(3):
        for component in range(2):
            results[f"row{row + 1}_pc{component + 1}"] = (
                pca["scores"][row, component]
            )
    for segment, mask in segment_masks.items():
        results[f"mean_pc1_segment_{segment}"] = pca["scores"][mask, 0].mean()

    # Numerical invariants do not substitute any printed checkpoint constants.
    assert sum(int(mask.sum()) for mask in segment_masks.values()) == len(value)
    assert np.allclose(standardised.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(np.diag(pca["covariance"]), 1, atol=1e-12)
    assert np.allclose(
        pca["eigenvectors"].T @ pca["eigenvectors"], np.eye(5), atol=1e-12
    )
    assert np.isclose(eigenvalues.sum(), np.trace(pca["covariance"]))

    write_results("part3_results.txt", results)
    write_csv(
        "part3_segments.csv",
        [
            "segment",
            "size",
            "mean_X1",
            "mean_X2",
            "mean_X3",
            "mean_X4",
            "mean_X5",
            "outcome_rate",
            "mean_value",
        ],
        segment_rows,
    )
    write_csv(
        "part3_hierarchical.csv",
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hierarchical_rows,
    )


if __name__ == "__main__":
    main()

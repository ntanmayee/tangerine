import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


def order_dataframe_by_linkage(df):
    """
    Computes hierarchical clustering on the fly to find the optimal
    visual ordering of rows (leaves) for a heatmap.
    """
    # Safety check
    if df.empty or len(df) < 2:
        return df

    # 1. Prepare the data matrix (Fill NaNs with 0 to prevent SciPy errors)
    data_matrix = df.fillna(0.0).values

    try:
        # 2. Compute pairwise Euclidean distances
        dist_matrix = pdist(data_matrix, metric="euclidean")

        # Safety catch: If all rows are completely identical (e.g., all 0s)
        if np.all(dist_matrix == 0):
            return df

        # 3. Compute Ward's linkage
        Z = linkage(dist_matrix, method="ward")

        # 4. Extract the optimal visual sorting of the leaves
        optimal_order = leaves_list(Z)

        # 5. Return the reordered dataframe
        return df.iloc[optimal_order]

    except Exception as e:
        # Failsafe: if math fails (e.g., infinite values), return original order
        print(f"Clustering failed: {e}")
        return df

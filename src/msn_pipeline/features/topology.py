from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def calculate_graph_metrics(
    G: nx.Graph,
    metrics: List[str],
    n_random: int = 50,
) -> Dict[str, Optional[float]]:
    """Thin wrapper around global graph metrics."""
    available_metrics = {
        "Eg": lambda: nx.global_efficiency(G) if G.number_of_nodes() > 1 else None,
        "Eloc": lambda: nx.local_efficiency(G) if G.number_of_nodes() > 1 else None,
        "Cp": lambda: nx.average_clustering(G, weight="weight") if G.number_of_edges() > 0 else 0,
        "Lp": lambda: get_average_path_length(G),
        "γ": lambda: get_smallworld_properties(G, n_random)[0],
        "λ": lambda: get_smallworld_properties(G, n_random)[1],
        "σ": lambda: get_smallworld_properties(G, n_random)[2],
    }
    return {m: available_metrics[m]() for m in metrics if m in available_metrics}


def get_average_path_length(G: nx.Graph) -> Optional[float]:
    if nx.is_empty(G):
        return None
    largest_cc = max(nx.connected_components(G), key=len)
    if len(largest_cc) < 2:
        return None
    return nx.average_shortest_path_length(G.subgraph(largest_cc))


def get_smallworld_properties(
    G: nx.Graph,
    n_random: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if nx.is_empty(G):
        return None, None, None

    Cp = nx.average_clustering(G, weight="weight")
    Lp = get_average_path_length(G)

    rand_cp, rand_lp = [], []
    for _ in range(n_random):
        Gr = nx.random_reference(G, niter=10)
        if Gr.number_of_edges() > 0:
            rand_cp.append(nx.average_clustering(Gr, weight="weight"))
            if nx.is_connected(Gr):
                rand_lp.append(nx.average_shortest_path_length(Gr))

    gamma = Cp / np.mean(rand_cp) if rand_cp else None
    lambda_ = Lp / np.mean(rand_lp) if Lp and rand_lp else None
    sigma = gamma / lambda_ if gamma and lambda_ else None

    return gamma, lambda_, sigma


def compute_nodal_topology(G: nx.Graph) -> pd.DataFrame:
    """Per-node topology metrics."""
    nodes = list(G.nodes())
    metrics = pd.DataFrame(index=nodes)

    metrics["degree_centrality"] = pd.Series(nx.degree_centrality(G))
    metrics["betweenness"] = pd.Series(nx.betweenness_centrality(G))
    metrics["clustering"] = pd.Series(nx.clustering(G))
    metrics["eigenvector"] = pd.Series(nx.eigenvector_centrality(G, max_iter=1000))
    metrics["pagerank"] = pd.Series(nx.pagerank(G))

    return metrics


def compute_graph_topology(G: nx.Graph) -> pd.DataFrame:
    """Graph-level topology metrics."""
    network_density = nx.density(G)

    if nx.is_connected(G):
        largest_component = G
    else:
        largest_nodes = max(nx.connected_components(G), key=len)
        largest_component = G.subgraph(largest_nodes)

    if nx.is_connected(largest_component):
        avg_path_length = nx.average_shortest_path_length(largest_component)
        graph_diameter = nx.diameter(largest_component)
    else:
        avg_path_length = None
        graph_diameter = None

    local_efficiency = nx.local_efficiency(G)

    from networkx.algorithms import community
    from networkx.algorithms.community.quality import modularity

    communities = list(community.greedy_modularity_communities(G))
    modularity_val = modularity(G, communities)

    metrics = pd.DataFrame(
        [
            {
                "density": network_density,
                "avg_path_length": avg_path_length,
                "diameter": graph_diameter,
                "local_efficiency": local_efficiency,
                "modularity": modularity_val,
            }
        ]
    )

    return metrics


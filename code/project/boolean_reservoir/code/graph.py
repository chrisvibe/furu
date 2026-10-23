import networkx as nx
import numpy as np
from project.boolean_reservoir.code.utils.utils import print_pretty_binary_matrix, make_rng
from scipy import sparse
import math

def generate_graph_w_k_avg_incoming_edges(n_nodes, k_min=None, k_avg=None, k_max=None, self_loops=None, rng=None):
    adj_matrix = generate_adjacency_matrix(n_nodes=n_nodes, k_min=k_min, k_avg=k_avg, k_max=k_max, self_loops=self_loops, rng=rng)
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return graph

def graph2adjacency_list_outgoing(graph: nx.Graph):
    # convention is typically outgoing edges
    adj_list = [[] for _ in range(graph.number_of_nodes())]
    for node, neighbors in graph.adjacency():
        adj_list[node] = list(neighbors)
    return adj_list

def calc_spectral_radius(graph: nx.DiGraph):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    eigenvalues = np.linalg.eigvals(adj_matrix)
    rho = max(abs(eigenvalues))
    return rho

def remove_isolated_nodes(graph: nx.Graph, remove_connected_to_self_only=False):
    in_degree = graph.in_degree
    non_isolated_nodes = {node for node in graph.nodes() if in_degree[node] > 0}
    if remove_connected_to_self_only:
        self_loops = {node for node in graph.nodes() if in_degree[node] == 1 and graph.has_edge(node, node)}
        non_isolated_nodes = non_isolated_nodes - self_loops
    return graph.subgraph(non_isolated_nodes).copy()

def constrain_degree_of_bipartite_mapping(a, b, min_degree, max_degree, p, in_degree=True, rng=None):
    '''
    a and b repesent a bipartite mapping a→b
    to build adjacency matrices for this we use an analogy of pigeons finding pigeon holes of a certain capacity
    in_degree: sets control of in-degree vs out-degree
    constraint: sets min max constrain on a or b
    p: probability of connection from max - min

    if in_degree = False we constrain out degree of a not b....
    '''
    rng = make_rng() if rng is None else rng
    capacity_range = max_degree - min_degree
    constrained_set = b if in_degree else a
    free_set = a if in_degree else b
    edge_range = capacity_range * constrained_set
    edge_range = (rng.random(edge_range) <= p).sum()
    k = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, constrained_set, capacity_range, rng=rng) # probabilistic connections
    k += min_degree # deterimistic connections

    # project 1D in-degree sequence to random 2D adjacency matrix
    w = random_projection_1d_to_2d(k, m=free_set, rng=rng)
    if in_degree:
        return w
    else:
        return w.T

def random_constrained_stub_matching(a, b, a_min, a_max, b_min, b_max, p, rng=None):
    rng = make_rng() if rng is None else rng
    # make ka:
    capacity_range_a = a_max - a_min
    capacity_range_b = b_max - b_min
    edge_range = min(capacity_range_a * b, capacity_range_b * a)
    edge_range = (rng.random(edge_range) <= p).sum()
    ka_out = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, a, capacity_range_a, rng=rng) # probabilistic connections
    ka_out += a_min # deteriministic connections

    # make kb:
    edge_range = ka_out.sum() 
    kb_in = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, b, capacity_range_b, rng=rng) # probabilistic connections
    kb_in += b_min # deteriministic connections
    
    # project 1D in-degree sequence to random 2D adjacency matrix
    w = random_boolean_adjancency_matrix_from_two_degree_sets(ka_out, kb_in)
    return w

def generate_adjacency_matrix(n_nodes, k_min: int=0, k_avg: float=None, k_max: int=None, self_loops: float=None, rows=None, fixed_edges: bool=True, rng=None):
    """
    Generate a random boolean directed adjacency matrix.

    fixed_edges=True  → G(n,m): exactly round(k_avg * n_nodes) edges, distributed via pigeonhole.
    fixed_edges=False → G(n,p): each node's degree drawn from Binomial(k_max-k_min, p); edge
                        count is a random variable with expectation k_avg * n_nodes.
    """
    rng = make_rng() if rng is None else rng
    actual_rows = n_nodes if rows is None else rows
    max_diag_len = min(actual_rows, n_nodes)

    k_max = actual_rows if k_max is None else k_max
    k_avg = rng.uniform(k_min, k_max) if k_avg is None else float(k_avg)

    assert 0 <= k_min <= k_avg <= k_max <= actual_rows, (
        f"Invalid k parameters: 0 <= k_min={k_min} <= k_avg={k_avg} <= k_max={k_max} <= actual_rows={actual_rows} must hold"
    )

    capacity_range = k_max - k_min

    if fixed_edges:
        # G(n,m): fix total edges and self-loop count upfront
        total_edges = round(k_avg * n_nodes) 
        max_possible_edges = actual_rows * n_nodes
        max_off_diagonal_edges = max_possible_edges - max_diag_len
        required_self_loops = max(0, total_edges - max_off_diagonal_edges)

        if self_loops is not None:
            n_self_loops = round(float(self_loops) * max_diag_len)
        else:
            n_self_loops = None

        assert total_edges <= max_possible_edges, (
            f"Theoretical limit on edges exceeded: total_edges={total_edges} > max_possible_edges={max_possible_edges} (n_nodes={n_nodes}, actual_rows={actual_rows})"
        )
        if self_loops is not None:
            assert total_edges >= n_self_loops, (
                f"Not enough total edges for requested self-loops: total_edges={total_edges} < n_self_loops={n_self_loops} (self_loops={self_loops}, max_diag_len={max_diag_len})"
            )
            assert n_self_loops >= required_self_loops, (
                f"Density conflict: n_self_loops={n_self_loops} < required_self_loops={required_self_loops} — matrix is too dense to have this few self-loops (total_edges={total_edges}, max_off_diagonal_edges={max_off_diagonal_edges})"
            )

        # 1. G(n,m): pigeonhole distributes exactly total_edges across nodes
        edge_range = total_edges - k_min * n_nodes
        k = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, n_nodes, capacity_range, rng=rng)
        k += k_min
    else:
        # G(n,p): each node draws its degree independently from Binomial(capacity_range, p)
        p_pool = (k_avg - k_min) / capacity_range if capacity_range > 0 else 0.0
        k = rng.binomial(capacity_range, p_pool, size=n_nodes) + k_min
        total_edges = k.sum()

        if self_loops is not None:
            n_self_loops = round(float(self_loops) * max_diag_len)
        else:
            # Expected diagonal density matches overall edge density
            n_self_loops = int(rng.binomial(max_diag_len, k_avg / actual_rows))

    # 2. Vectorized 1D Repair: Ensure enough k>0 on the diagonal
    if self_loops is not None:
        diag_k = k[:max_diag_len]
        self_loop_potential = (diag_k > 0).sum()
        
        if self_loop_potential < n_self_loops:
            diff = n_self_loops - self_loop_potential
            
            # Find indices where k == 0 on the diagonal to bump up
            zero_idx = np.where(diag_k == 0)[0]
            add_idx = rng.choice(zero_idx, diff, replace=False)
            
            # Find indices where we can steal an edge (k > max(1, k_min))
            min_k_allowed = max(1, k_min)
            steal_candidates = np.where(k > min_k_allowed)[0]
            
            # Create a weighted pool so columns with lots of edges can be chosen multiple times
            steal_amounts = k[steal_candidates] - min_k_allowed
            steal_pool = np.repeat(steal_candidates, steal_amounts)
            steal_idx = rng.choice(steal_pool, diff, replace=False)
            
            # Apply changes instantly. np.add.at safely handles duplicate steal indices
            k[add_idx] += 1
            np.add.at(k, steal_idx, -1)

    # 3. Project 1D array to 2D Matrix
    adj_matrix = random_projection_1d_to_2d(k, m=actual_rows, rng=rng)

    # 4. Vectorized 2D Repair: Force exact self-loop count without for-loops
    if self_loops is not None:
        diagonal = np.diag(adj_matrix)
        self_loop_diff = n_self_loops - diagonal.sum()
        
        if self_loop_diff != 0:
            diff = abs(self_loop_diff)
            add_edge_to_diagonal = self_loop_diff > 0
            
            if add_edge_to_diagonal:
                # Valid columns: Diagonal is 0, AND column has edges to give
                valid_cols = np.where((diagonal == 0) & (k[:max_diag_len] > 0))[0]
                chosen_cols = rng.choice(valid_cols, diff, replace=False)
                
                # Find an existing '1' in each chosen column
                col_mask = adj_matrix[:, chosen_cols] == 1
            else:
                # Valid columns: Diagonal is 1, AND column is not full
                valid_cols = np.where((diagonal == 1) & (k[:max_diag_len] < actual_rows))[0]
                num_to_change = min(diff, len(valid_cols)) # Safety check
                chosen_cols = rng.choice(valid_cols, num_to_change, replace=False)
                
                # Find an existing '0' in each chosen column
                col_mask = adj_matrix[:, chosen_cols] == 0
                
            # Exclude the diagonal cells themselves from being selected
            col_mask[chosen_cols, np.arange(len(chosen_cols))] = False
            
            # CRITICAL TRICK: Multiply by random noise so np.argmax doesn't systematically steal from row 0
            rand_weights = rng.random((actual_rows, len(chosen_cols)))
            weighted_mask = col_mask * rand_weights
            r_indices = np.argmax(weighted_mask, axis=0)
            
            # Execute all swaps instantly
            if add_edge_to_diagonal:
                adj_matrix[r_indices, chosen_cols] = 0
                adj_matrix[chosen_cols, chosen_cols] = 1
            else:
                adj_matrix[r_indices, chosen_cols] = 1
                adj_matrix[chosen_cols, chosen_cols] = 0

    return adj_matrix

def random_projection_1d_to_2d(k, m, rng=None):
    """
    Projects 1D degree sequence k into a 2D adjacency matrix.
    """
    rng = make_rng() if rng is None else rng
    n = len(k)
    
    # 1. Generate column indices (vectorized)
    # [0, 0, 0, 1, 1, 2, 2, 2, 2...] 
    cols = np.repeat(np.arange(n), k)
    
    # 2. Generate row indices (loop is necessary here because ki varies)
    # Use rng.choice instead of permutation for speed
    rows = np.concatenate([
        rng.choice(m, ki, replace=False)
        for ki in k
    ])

    # For smaller/dense graphs: vectorized assignment
    adj = np.zeros((m, n), dtype=bool)
    adj[rows, cols] = True
    return adj

def randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(pigeons, holes, capacity, rng=None):
    # this returns a normally distributed hole occupance by CLT constrained by capacity [0, capacity]
    rng = make_rng() if rng is None else rng
    max_occupance = holes * capacity
    if max_occupance == pigeons:
        return np.full((holes,), capacity, dtype=int)
    assert pigeons <= max_occupance, "Too many pigeons for the given number of holes and capacity"
    worst_case_capacity = min(capacity, pigeons)
    possible_hole_assignments = np.zeros((holes * worst_case_capacity), dtype=bool)
    possible_hole_assignments[:pigeons] = True
    rng.shuffle(possible_hole_assignments)
    hole_occupance = possible_hole_assignments.reshape(worst_case_capacity, holes).sum(axis=0)
    return hole_occupance

def random_boolean_adjancency_matrix_from_p(n: int, m: int, p: float, rng=None) -> np.ndarray:
    rng = make_rng() if rng is None else rng
    adj_matrix = rng.random((n, m))
    adj_matrix = adj_matrix <= p
    return adj_matrix

def random_boolean_adjancency_matrix_from_two_degree_sets(ka: np.ndarray, kb: np.ndarray, max_tries=100) -> np.ndarray:
    """
    Generate a boolean adjacency matrix from two degree sequences ka and kb.
    
    Why not do this?
    G = nx.bipartite.configuration_model(ka, kb, create_using=nx.Graph())
    Exact degree sequences may not be realized due to the rejection of non-simple elements
    creat_using=nx.graph() removes the multi-edges and self-loops, but then the edge count may be wrong

    Ok... and this?
    G = nx.bipartite.havel_hakimi_graph(ka, kb, create_using=nx.Graph())
    This is deterministic. I need random.

    ... And this?
    G = nx.bipartite.random_graph(len(ka), len(kb), p)
    Doesnt give fine grained control over degrees :(

    Conclusion:
    1. configuration_model
    2. attempt repair remaining after drop of non-simple elements

    Alternative: 
    1. deterministic solve: havel-hakimi
    2. scramble: double edge swap (problem: makes connections within sets) or curveball algorithm (problem: not available)

    :param ka: A 1D integer ndarray of degree sequence for set A (length n).
    :param kb: A 1D integer ndarray of degree sequence for set B (length n).
    :return: A boolean ndarray representing the adjacency matrix.
    """
    raise NotImplementedError('This is a hard problem and is not yet needed...')
    assert sum(ka) == sum(kb), "The sum of degrees in ka and kb must be equal (handshake lemma)."
    assert ka.max() <= len(kb), "No node in set A should have a degree greater than the number of nodes in set B."
    assert kb.max() <= len(ka), "No node in set B should have a degree greater than the number of nodes in set A."
    assert (ka >= 0).all(), "All elements of ka must be > 0 (nodes should have non-negative degree)."
    assert (kb >= 0).all(), "All elements of kb must be > 0 (nodes should have non-negative degree)."

    # Create initial probabilisitic graph
    A = range(len(ka))
    B = range(len(ka), len(ka) + len(kb))
    for i in range(max_tries):
        G = nx.bipartite.configuration_model(ka, kb, create_using=nx.Graph())
        adj_matrix = nx.bipartite.biadjacency_matrix(G, row_order=A, column_order=B).toarray()
        currrent_ka = adj_matrix.sum(axis=1)
        currrent_kb = adj_matrix.sum(axis=0)
        if (currrent_ka == ka).all() and (currrent_kb == kb).all():
            return adj_matrix 

    # # Brute Force Repair (wont always work...)
    # currrent_ka = adj_matrix.sum(axis=1)
    # deficit_a = ka - currrent_ka
    # missing = deficit_a.sum()
    # if missing: # handshake lemma implies kb is satisdied when ka is
    #     currrent_kb = adj_matrix.sum(axis=0)
    #     deficit_b = kb - currrent_kb
    #     candidates_a = np.where(deficit_a > 0)[0]
    #     candidates_b = np.where(deficit_b > 0)[0]
    #     # make all possible edges (ignore edges that are already 1)
    #     candidate_edges = [(i, j) for i in candidates_a for j in candidates_b if adj_matrix[i, j] == 0]
    #     if not candidate_edges:
    #         if recursion <= max_recursion:
    #             return random_boolean_adjancency_matrix_from_two_degree_sets(ka, kb, recursion=recursion+1)
    #         UserWarning('random_boolean_adjancency_matrix_from_two_degree_sets: Could not satisfy degree set constraints')
    #         return adj_matrix
    #     # randomly choose missing
    #     candidate_edges = np.stack(candidate_edges, axis=0)
    #     idx = np.random.choice(len(candidate_edges), size=missing, replace=False)
    #     candidate_edges = candidate_edges[idx]
    #     # set missing (without checking)
    #     adj_matrix[candidate_edges[:, 0], candidate_edges[:, 1]] = 1

    # assert (ka == adj_matrix.sum(axis=1)).all(), "row sum mismatch (ka)"
    # assert (kb == adj_matrix.sum(axis=0)).all(), "column sum mismatch (kb)"
    # return adj_matrix

# This can be used to avoid case where u want G(n, m) but m % k_avg != 0 which causes systematic bias in the degree distribution if n is small
def generate_k_avg_series(n_nodes, from_val, to_val, approx_skip):
    """
    Generate a series of k_avg values for G(n, m) where m = k_avg * n_nodes is always an integer.
    The spacing between consecutive k_avg values is approximately `approx_skip`.

    Args:
        n_nodes: Number of nodes in the reservoir (e.g., 25).
        from_val: Start value for k_avg (e.g., 0).
        to_val: End value for k_avg (e.g., 6).
        approx_skip: Approximate spacing between k_avg values (e.g., 0.1).

    Returns:
        List of k_avg values where k_avg * n_nodes is integer, spaced ~approx_skip apart.
    """
    target_step_m = approx_skip * n_nodes
    lower_step = max(1, int(math.floor(target_step_m)))
    upper_step = max(1, int(math.ceil(target_step_m)))

    # Use a single step if target_step_m is integer, else alternate
    if lower_step == upper_step:
        step_pattern = [lower_step]
    else:
        step_pattern = [lower_step, upper_step]

    start_m = int(math.ceil(from_val * n_nodes))
    end_m = int(math.floor(to_val * n_nodes))

    result = []
    m = start_m
    pattern_index = 0

    while m <= end_m:
        k_avg = m / n_nodes
        if k_avg > to_val:
            break
        result.append(k_avg)
        m += step_pattern[pattern_index % len(step_pattern)]
        pattern_index += 1

    return result   

if __name__ == '__main__':
    w = generate_adjacency_matrix(n_nodes=1000, k_avg=3, k_max=6, k_min=2, self_loops=0)
    k = w.sum(axis=0)
    print(k)
    print(k.min(), k.max())
    eigenvalues = np.linalg.eigvals(w)
    rho = max(abs(eigenvalues))
    print(rho)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    eigenvalue_magnitudes = eigenvalue_magnitudes[np.argsort(-eigenvalue_magnitudes,)]
    print("Eigenvalues (first 10):\n", eigenvalue_magnitudes[:10])
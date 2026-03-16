"""
Forest Fire Route Blocking — Task 5 (GreedyReplace Version)
Algorithm : GreedyReplace adapted for Edge Blocking (IMIN-EB)
Reference : Xie et al. (2024) INFORMS Journal on Computing
            "Influence Minimization via Blocking Strategies"

Key design:
  Phase 1 — Greedy over SEED OUT-EDGES only (OutNeighbors baseline)
  Phase 2 — Replace in reverse insertion order from ALL candidates
             (recovers any better edge missed in Phase 1)
  DESCE   — Dominator tree simultaneous gain for all edges per sample
             Virtual node w=(u,v) represents edge (u,v) in g^E
             Blocking w ≡ blocking edge (u,v) — Theorem 4/5 Xie et al.
"""

import sys
import random
import time
import heapq
from collections import defaultdict, deque
import numpy as np

"""
 I/O
"""

def read_graph(path):
    adj = defaultdict(list)
    nodes = set()
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            p = float(parts[2]) if len(parts) > 2 else 1.0
            edges.append((u, v, p))
            adj[u].append((v, p))
            nodes.add(u); nodes.add(v)
    return edges, adj, nodes


def read_seeds(path):
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(int(line.split()[0]))
    return seeds


def flush_edge(path, u, v):
    """Write one blocked edge immediately for getting partial credit on timeout."""
    with open(path, 'a') as f:
        f.write(f"{u} {v}\n")
        f.flush()
"""
CSR precomputation
"""

def build_csr(adj, node2idx, blocked_set):
    """Build CSR format adjacency for fast traversal."""
    N = len(node2idx)
    # Count out-degrees
    out_deg = np.zeros(N, dtype=np.int32)
    for u, neighbors in adj.items():
        if u not in node2idx:
            continue
        uidx = node2idx[u]
        for (v, p) in neighbors:
            if v in node2idx and (u,v) not in blocked_set:
                out_deg[uidx] += 1
    
    # Build indptr (CSR row pointers)
    indptr = np.zeros(N+1, dtype=np.int32)
    for i in range(N):
        indptr[i+1] = indptr[i] + out_deg[i]
    
    # Fill indices and probs
    total_edges = indptr[N]
    indices = np.zeros(total_edges, dtype=np.int32)
    probs = np.zeros(total_edges, dtype=np.float32)
    pos = indptr.copy()
    
    for u, neighbors in adj.items():
        if u not in node2idx:
            continue
        uidx = node2idx[u]
        for (v, p) in neighbors:
            if v in node2idx and (u,v) not in blocked_set:
                vidx = node2idx[v]
                indices[pos[uidx]] = vidx
                probs[pos[uidx]] = p
                pos[uidx] += 1
    
    return indptr, indices, probs


def mc_spread_csr(indptr, indices, probs, seed_indices, N, r):
    """Fast MC spread using CSR format — no dict lookups inside loop."""
    total = 0
    for _ in range(r):
        burned = np.zeros(N, dtype=np.bool_)
        queue = []
        for s in seed_indices:
            burned[s] = True
            queue.append(s)
        qi = 0
        while qi < len(queue):
            u = queue[qi]; qi += 1
            start, end = indptr[u], indptr[u+1]
            # Vectorized random check for all neighbors at once
            rand_vals = np.random.random(end - start)
            for j in range(end - start):
                v = indices[start + j]
                if not burned[v] and rand_vals[j] < probs[start + j]:
                    burned[v] = True
                    queue.append(v)
        total += burned.sum()
    return total / r

"""
 GRAPH UTILITIES
"""

def multisource_bfs(adj, seeds, h=None):
    """BFS from all seeds; stop at depth h if given."""
    visited = set(seeds)
    queue = deque((s, 0) for s in seeds)
    while queue:
        u, d = queue.popleft()
        if h is not None and d >= h:
            continue
        for (v, _) in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append((v, d + 1))
    return visited


def get_all_candidates(adj, seeds, v_filter):
    cands = []
    for u in v_filter:  
        for (v, p) in adj.get(u, []):
            if v in v_filter:
                cands.append((u, v, p))
    cands.sort(key=lambda x: -x[2])
    return cands


def get_seed_out_edges(adj, seeds, v_filter):
    cands = []
    seen = set()
    for s in seeds:
        if s not in v_filter:  # ADD THIS CHECK
            continue
        for (v, p) in adj.get(s, []):
            if v in v_filter and (s, v) not in seen:
                cands.append((s, v, p))
                seen.add((s, v))
    cands.sort(key=lambda x: -x[2])
    return cands


"""
DOMINATOR TREE  (Cooper et al. 2001, BFS-order iterative)
"""

def build_dominator_tree(g_succ, g_pred, root):
    """
    Returns (idom dict, bfs_order list).
    Only processes nodes reachable from root.
    """
    order = []
    visited = {root}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in g_succ.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)

    idx = {n: i for i, n in enumerate(order)}
    idom = {root: root}

    def intersect(a, b):
        lim = len(order) + 5
        s = 0
        while a != b and s < lim:
            s += 1
            while idx.get(a, -1) > idx.get(b, -1):
                a = idom.get(a, root)
            while idx.get(b, -1) > idx.get(a, -1):
                b = idom.get(b, root)
        return a

    changed = True
    while changed:
        changed = False
        for u in order[1:]:
            processed = [p for p in g_pred.get(u, []) if p in idom]
            if not processed:
                continue
            new_idom = processed[0]
            for p in processed[1:]:
                new_idom = intersect(new_idom, p)
            if idom.get(u) != new_idom:
                idom[u] = new_idom
                changed = True

    return idom, order


"""
DESCE — simultaneous gain estimation via dominator trees
Xie et al. (2024) Algorithm 2, edge-blocking version
"""

def desce(adj, seeds, blocked_set, theta, v_filter):
    """
    Returns delta dict: (u,v) - estimated spread reduction from blocking (u,v).

    For each of theta IC realizations:
      1. Sample edges probabilistically.
      2. Build edge-sampled graph g^E: insert virtual node w=(u,v) per edge.
      3. Build dominator tree from virtual root R.
      4. delta[(u,v)] += |subtree(w) intersects original nodes intersects v_filter| / theta.

    This estimates the marginal gain of blocking each edge SIMULTANEOUSLY
    in O(theta * m * alpha(m,n)) — no per-edge re-simulation needed.
    """
    delta = defaultdict(float)
    inv_t = 1.0 / theta
    ROOT = "R"   # virtual root — string never clashes with int node ids

    for _ in range(theta):
        gs = defaultdict(list)   # g^E successors
        gp = defaultdict(list)   # g^E predecessors

        # Virtual root - all seeds (probability 1)
        for s in seeds:
            gs[ROOT].append(s)
            gp[s].append(ROOT)

        sampled = []   # [((u,v), virtual_node)]

        for u in adj:
            for (v, p) in adj[u]:
                if (u, v) in blocked_set:
                    continue
                if random.random() >= p:
                    continue
                w = (u, v)          # virtual node id = edge tuple
                sampled.append(((u, v), w))
                gs[u].append(w)
                gp[w].append(u)
                gs[w].append(v)
                gp[v].append(w)

        # Dominator tree of g^E from ROOT
        idom, order = build_dominator_tree(gs, gp, ROOT)

        # Build children map
        ch = defaultdict(list)
        for node, par in idom.items():
            if node != ROOT:
                ch[par].append(node)

        # Bottom-up subtree count: original int nodes in v_filter only
        sub = {}
        for node in reversed(order):
            own = 1 if (isinstance(node, int)
                             and node >= 0
                             and node in v_filter) else 0
            sub[node] = own + sum(sub.get(c, 0) for c in ch.get(node, []))

        for ((u, v), w) in sampled:
            if w in sub:
                delta[(u, v)] += sub[w] * inv_t

    return delta




"""
GREEDY REPLACE — IMIN-EB adaptation
Phase 1: greedy over seed out-edges only
Phase 2: replace in reverse order from all candidates
"""

def run_small_graphs(adj, seeds, all_candidates, seed_out_edges,
                      k, theta, v_filter, output_path, deadline):
    """
    GreedyReplace for edge blocking (IMIN-EB).

    Phase 1: restrict CB to direct out-edges of seeds.
             Greedily pick best min(|CB|, k) edges via DESCE.
    Phase 2: for each chosen edge in reverse order:
             - temporarily remove it from B
             - find globally best edge from ALL candidates via DESCE
             - replace if better; early-terminate if same edge wins
    """
    blocked_set = set()
    B = []       # ordered list of blocked edges
    insertion_order = []    # track Phase 1 insertion order for Phase 2
    start = time.time()

    """ PHASE 1: Greedy over seed out-edges """
    phase1_budget = min(len(seed_out_edges), k)
    print(f"[GR] Phase 1: {phase1_budget} rounds over "
          f"{len(seed_out_edges)} seed out-edges", flush=True)

    for i in range(phase1_budget):
        if time.time() > deadline - 90:
            print(f"[GR] Phase 1 timeout at round {i}.", flush=True)
            break

        elapsed = time.time() - start
        # Adaptive theta
        if i > 0:
            tpr = elapsed / i
            remaining = deadline - time.time()
            total_rounds_est = phase1_budget + len(B)   # phase1 + phase2
            if tpr * (total_rounds_est - i) > remaining * 0.5:
                atheta = max(5, theta // 3)
            else:
                atheta = theta
        else:
            atheta = theta

        delta = desce(adj, seeds, blocked_set, atheta, v_filter)

        # Best edge from seed out-edges not yet blocked
        best_edge, best_gain = None, -1.0
        for (u, v, p) in seed_out_edges:
            if (u, v) not in blocked_set:
                g = delta.get((u, v), 0.0)
                if g > best_gain:
                    best_gain = g
                    best_edge = (u, v)

        if best_edge is None:
            break

        u, v = best_edge
        blocked_set.add((u, v))
        B.append((u, v))
        insertion_order.append((u, v))
        flush_edge(output_path, u, v)
        print(f"[GR P1 {i+1}/{phase1_budget}] ({u},{v}) "
              f"gain={best_gain:.4f} theta={atheta} "
              f"t={time.time()-start:.1f}s", flush=True)

    """PHASE 2: Replace in reverse insertion order """
    print(f"[GR] Phase 2: replacing {len(insertion_order)} edges "
          f"from {len(all_candidates)} total candidates", flush=True)

    replaced_any = False
    for e_u in reversed(insertion_order):
        if time.time() > deadline - 60:
            print("[GR] Phase 2 timeout.", flush=True)
            break

        # Temporarily remove e_u from B
        blocked_set.discard(e_u)

        elapsed = time.time() - start
        remaining = deadline - time.time()
        atheta = max(5, theta // 2) if remaining < 300 else theta

        delta = desce(adj, seeds, blocked_set, atheta, v_filter)

        # Best edge from ALL candidates not in current B
        best_edge, best_gain = None, -1.0
        for (u, v, p) in all_candidates:
            if (u, v) not in blocked_set:
                g = delta.get((u, v), 0.0)
                if g > best_gain:
                    best_gain = g
                    best_edge = (u, v)

        if best_edge is None:
            blocked_set.add(e_u)   # restore
            continue

        # Replace e_u with best_edge in B
        B = [(u, v) for (u, v) in B if (u, v) != e_u]
        B.append(best_edge)
        blocked_set.add(best_edge)

        if best_edge != e_u:
            replaced_any = True
            print(f"[GR P2] Replaced {e_u} → {best_edge} "
                  f"gain={best_gain:.4f} t={time.time()-start:.1f}s",
                  flush=True)
        else:
            # Early termination: the removed edge is still the best
            blocked_set.add(e_u)
            print(f"[GR P2] Early stop — {e_u} is still best.",
                  flush=True)
            continue #break

    """If Phase 1 produced fewer than k edges, fill greedily"""
    if len(B) < k:
        remaining_budget = k - len(B)
        print(f"[GR] Phase 3: greedy fill {remaining_budget} more edges",
              flush=True)

        for i in range(remaining_budget):
            if time.time() > deadline - 45:
                break

            atheta = max(5, theta // 2)
            delta = desce(adj, seeds, blocked_set, atheta, v_filter)

            best_edge, best_gain = None, -1.0
            for (u, v, p) in all_candidates:
                if (u, v) not in blocked_set:
                    g = delta.get((u, v), 0.0)
                    if g > best_gain:
                        best_gain = g
                        best_edge = (u, v)

            if best_edge is None:
                break

            u, v = best_edge
            blocked_set.add((u, v))
            B.append((u, v))
            flush_edge(output_path, u, v)
            print(f"[GR P3 {i+1}/{remaining_budget}] ({u},{v}) "
                  f"gain={best_gain:.4f}", flush=True)

    # Rewrite output file with final B (Phase 2 may have changed edges)
    with open(output_path, 'w') as f:
        for (u, v) in B:
            f.write(f"{u} {v}\n")
            f.flush()

    return B, blocked_set


"""
MC + CELF FALLBACK  (large graphs where DT is too slow)
"""

def run_large_graphs(adj, seeds, candidates, k, r, v_filter,
                   output_path, deadline):
    # Build node2idx ONCE
    nodes = list(v_filter)
    node2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    seed_indices = [node2idx[s] for s in seeds if s in node2idx]
    
    blocked_set = set()
    blocked = []
    
    def spread_with_blocked(bset):
        """Build CSR excluding blocked edges, then run MC spread."""
        indptr, indices, probs = build_csr(adj, node2idx, bset)
        return mc_spread_csr(indptr, indices, probs, seed_indices, N, r)
    
    baseline = spread_with_blocked(blocked_set)
    print(f"[CELF] baseline={baseline:.2f}", flush=True)

    gain_cache = {}
    for (u, v, p) in candidates:
        if time.time() > deadline - 120:
            break
        blocked_set.add((u, v))
        s2 = spread_with_blocked(blocked_set)
        gain_cache[(u, v)] = (baseline - s2, 0)
        blocked_set.remove((u, v))

    pq = [(-g, it, u, v) for (u,v),(g,it) in gain_cache.items()]
    heapq.heapify(pq)
    cur_base = baseline

    for i in range(k):
        if time.time() > deadline - 45:
            break

        best = None
        while pq:
            neg_g, it, u, v = heapq.heappop(pq)
            if (u, v) in blocked_set:
                continue
            g, comp = gain_cache.get((u, v), (0, -1))
            if comp == i:
                best = (u, v, g)
                break
            blocked_set.add((u, v))
            new_s = spread_with_blocked(blocked_set)
            blocked_set.remove((u, v))
            new_g = cur_base - new_s
            gain_cache[(u, v)] = (new_g, i)
            heapq.heappush(pq, (-new_g, i, u, v))

        if best is None:
            break

        u, v, g = best
        blocked_set.add((u, v))
        blocked.append((u, v))
        flush_edge(output_path, u, v)
        cur_base = spread_with_blocked(blocked_set)
        print(f"[CELF {i+1}/{k}] ({u},{v}) gain={g:.3f} "
              f"spread={cur_base:.2f}", flush=True)

    return blocked, blocked_set

"""
Run Function
"""

DOM_TREE_THRESHOLD = 80000   # nodes; above this use MC-CELF

def run(graph_path, seed_path, output_path, k, theta, h):
    TIME_LIMIT = 55 * 60
    start = time.time()
    deadline = start + TIME_LIMIT

    edges, adj, nodes = read_graph(graph_path)
    seeds = read_seeds(seed_path)
    n, m = len(nodes), len(edges)
    print(f"[INFO] n={n} m={m} k={k} theta={theta} h={h} "
          f"seeds={seeds}", flush=True)

    open(output_path, 'w').close()

    if not seeds:
        for (u, v, p) in edges[:k]:
            flush_edge(output_path, u, v)
        return

    # v_filter: nodes fire is allowed to reach
    v_filter = nodes if h == -1 else multisource_bfs(adj, seeds, h)
    print(f"[INFO] v_filter={len(v_filter)}", flush=True)

    # Candidate edge sets
    all_candidates = get_all_candidates(adj, seeds, v_filter)
    seed_out_edges = get_seed_out_edges(adj, seeds, v_filter)
    print(f"[INFO] all_candidates={len(all_candidates)} "
          f"seed_out_edges={len(seed_out_edges)}", flush=True)

    if not all_candidates:
        bset = set()
        for (u, v, p) in edges[:k]:
            if (u, v) not in bset:
                flush_edge(output_path, u, v)
                bset.add((u, v))
        return

    if n <= DOM_TREE_THRESHOLD:
        print(f"[INFO] Mode: GreedyReplace+DESCE theta={theta}", flush=True)
        blocked, blocked_set = run_small_graphs(
            adj, seeds, all_candidates, seed_out_edges,
            k, theta, v_filter, output_path, deadline)
    else:
        r = max(15, min(theta, 80))
        print(f"[INFO] Mode: MC-CELF r={r}", flush=True)
        blocked, blocked_set = run_large_graphs(
            adj, seeds, all_candidates, k, r,
            v_filter, output_path, deadline)

    # Pad to exactly k if stopped early
    if len(blocked) < k:
        bset = set(blocked)
        for (u, v, p) in all_candidates:
            if len(blocked) >= k:
                break
            if (u, v) not in bset:
                blocked.append((u, v))
                bset.add((u, v))
                flush_edge(output_path, u, v)
        for (u, v, p) in edges:
            if len(blocked) >= k:
                break
            if (u, v) not in bset:
                blocked.append((u, v))
                bset.add((u, v))
                flush_edge(output_path, u, v)

    print(f"[DONE] {len(blocked)}/{k} blocked in "
          f"{time.time()-start:.1f}s", flush=True)


"""
ENTRY POINT
"""

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: forest_fire.py <graph> <seeds> <o> "
              "<k> <n_random_instances> <hops>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
"""Orthonormal Haar basis on a TreeMesh's own parent-child hierarchy.

Rebuilt from the 2026 implementation note (the original script was lost, only
its log survived): reconstruct the hierarchy from cell sizes and centres, weight
each node by the number of leaf cells beneath it so that best-N-term truncation
is optimal in per-cell L2, unify sibling groups and lone-child pass-throughs
into one ordered op list, and replay it reversed for synthesis.

Why this basis and not a DCT: a cosine basis is global, so a sharp contact costs
an enormous number of coefficients (measured: 1.6M for 79,863 cells, i.e.
negative compression). Haar on the octree is local AND multiscale, so a sharp
body spends coefficients only near its boundary. It also needs no regular grid
and therefore no air box.

analysis(x) -> coefficients; synthesis(c) -> cells. Bijective and orthonormal:
len(c) == n_active, and ||c|| == ||sqrt(w) * x||.
"""
import numpy as np


class OctreeHaar:
    def __init__(self, mesh, active):
        act = np.asarray(active, dtype=bool)
        cc = mesh.cell_centers[act]
        h = mesh.h_gridded[act][:, 0]          # cubic cells: one size is enough
        self.n = cc.shape[0]
        origin = cc.min(0) - h.max() * 4
        idx = np.arange(self.n)
        w = np.ones(self.n)                     # leaf count under each node
        val_rows = idx.copy()                   # which coeff slot each node owns
        self.ops = []
        free = self.n - 1                       # coefficients fill from the end
        cur_c, cur_h, cur_w, cur_i = cc, h, w, idx
        while len(cur_i) > 1:
            # parent key: snap centre onto the grid one level coarser
            key = np.floor((cur_c - origin) / (2 * cur_h[:, None])).astype(np.int64)
            key = np.c_[key, np.round(np.log2(cur_h / cur_h.min())).astype(np.int64)]
            _, inv = np.unique(key, axis=0, return_inverse=True)
            order = np.argsort(inv, kind="stable")
            inv_s = inv[order]
            starts = np.r_[0, np.flatnonzero(np.diff(inv_s)) + 1, len(inv_s)]
            nxt_c, nxt_h, nxt_w, nxt_i = [], [], [], []
            for a, b in zip(starts[:-1], starts[1:]):
                g = order[a:b]
                gi, gw = cur_i[g], cur_w[g]
                if len(g) == 1:                       # lone child: pass through
                    nxt_i.append(gi[0]); nxt_w.append(gw[0])
                else:
                    s = np.sqrt(gw)
                    Q = np.linalg.qr(np.c_[s / np.linalg.norm(s),
                                           np.eye(len(g))], mode="reduced")[0]
                    Q = Q[:, :len(g)]
                    if Q[:, 0] @ (s / np.linalg.norm(s)) < 0:
                        Q[:, 0] *= -1
                    slots = np.array([free - k for k in range(len(g) - 1)])
                    free -= len(g) - 1
                    self.ops.append((gi.copy(), Q.copy(), slots))
                    nxt_i.append(gi[0]); nxt_w.append(gw.sum())
                nxt_c.append(cur_c[g[0]]); nxt_h.append(cur_h[g[0]] * 2)
            cur_i = np.array(nxt_i); cur_w = np.array(nxt_w)
            cur_c = np.array(nxt_c); cur_h = np.array(nxt_h)
            if len(cur_i) == len(self.ops) and False:
                break
            if len(nxt_i) == len(order):              # nothing merged: stop
                break
        self.root, self.root_w = cur_i, cur_w
        self.free = free
        self.w_leaf = w

    def analysis(self, x):
        """cells -> coefficients (orthonormal, bijective)."""
        buf = np.asarray(x, dtype=float).copy()
        c = np.zeros(self.n)
        for gi, Q, slots in self.ops:
            y = Q.T @ buf[gi]
            buf[gi[0]] = y[0]
            c[slots] = y[1:]
        for k, i in enumerate(self.root):
            c[k] = buf[i]
        return c

    def synthesis(self, c):
        """coefficients -> cells."""
        buf = np.zeros(self.n)
        for k, i in enumerate(self.root):
            buf[i] = c[k]
        for gi, Q, slots in reversed(self.ops):
            y = np.r_[buf[gi[0]], c[slots]]
            buf[gi] = Q @ y
        return buf

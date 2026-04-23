"""Near-duplicate detection with MinHashLSH over word shingles.

Two records describing the same event (often syndicated across BEA / NTSB /
TSB / FAA) end up with very similar narratives. We shingle each normalized
narrative into k-word grams, feed the shingles into a MinHash, and bucket
via LSH. Within each connected component the longest narrative wins.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

from datasketch import MinHash, MinHashLSH


def _shingles(text: str, k: int = 5) -> List[bytes]:
    toks = text.lower().split()
    if len(toks) < k:
        return [" ".join(toks).encode("utf-8")] if toks else []
    return [" ".join(toks[i : i + k]).encode("utf-8") for i in range(len(toks) - k + 1)]


def _minhash(text: str, num_perm: int, k: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text, k=k):
        m.update(sh)
    return m


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def deduplicate_minhash(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_k: int = 5,
) -> List[List[int]]:
    """Return clusters of indices that are near-duplicates.

    Each cluster is a list of indices into ``texts``. Singletons are included
    as length-1 clusters so the caller can iterate clusters to build a unified
    corpus.
    """
    n = len(texts)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs: List[MinHash] = []
    for i, t in enumerate(texts):
        m = _minhash(t, num_perm=num_perm, k=shingle_k)
        mhs.append(m)
        lsh.insert(str(i), m)

    find, union = _union_find(n)
    for i, m in enumerate(mhs):
        for j_str in lsh.query(m):
            j = int(j_str)
            if j != i:
                union(i, j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

"""Offline, dependency-free embedder.

``HashEmbedder`` maps text into a fixed-dimension vector via feature hashing.
It needs no model download or network, so examples and tests run anywhere.

Features combine word tokens with character bigrams, so it works for both
space-delimited languages (English) and CJK text where a whole phrase is one
"word" (e.g. "用户对花生过敏" shares the bigram "花生" with the query "花生").
Feature overlap drives cosine similarity — enough for keyword-style retrieval.
Swap in a real embedder for production semantics.
"""

from __future__ import annotations

import hashlib
import math
import re

from morainet.memory.base import Embedder

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _features(text: str) -> list[str]:
    feats: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        feats.append(token)
        if len(token) == 1:
            continue
        feats.extend(token[i : i + 2] for i in range(len(token) - 1))
    return feats


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in _features(text):
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

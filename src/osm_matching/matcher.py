"""Fuzzy matching of an OCR text cue against OSM street names."""
from collections import defaultdict

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from . import config
from .normalize import compact_key, name_keys

MIN_QUERY_LEN = 4  # shorter cues carry too little information to match on

# Trailing words a sign may omit: 'Cubbon Road' ~ 'Cubbon'
STRIPPABLE_SUFFIXES = (
    "road", "street", "lane", "avenue", "boulevard", "highway",
    "circle", "marg", "veedhi", "path",
)
# A cue made only of these words says nothing about which street it is
GENERIC_WORDS = set(STRIPPABLE_SUFFIXES) | {"cross", "main"}


def _core(key):
    """Drop a trailing road-type word: 'cubbonroad' -> 'cubbon'. None if nothing to drop."""
    for suffix in STRIPPABLE_SUFFIXES:
        if key.endswith(suffix) and len(key) - len(suffix) >= MIN_QUERY_LEN:
            return key[: -len(suffix)]
    return None


class StreetIndex:
    """Index of OSM street names, built once from the roads table."""

    def __init__(self, roads):
        named = roads[roads["name"].notna()]
        self.name_stats = {
            name: {"segments": int(len(group)), "length_m": round(float(group["length_m"].sum()), 1)}
            for name, group in named.groupby("name")
        }
        # key -> {(street name, kind)}; kind is "full" or "initials"
        self.key_to_entries = defaultdict(set)
        for name in self.name_stats:
            full = compact_key(name)
            for key in name_keys(name):
                kind = "full" if key == full else "initials"
                self.key_to_entries[key].add((name, kind))
                core = _core(key)
                if core:
                    self.key_to_entries[core].add((name, kind))
        self.keys = list(self.key_to_entries)

    def match(self, text, limit=None, min_score=None):
        """Return up to `limit` candidate streets for a text cue, best first. [] if nothing fits."""
        limit = limit or config.MAX_CANDIDATES
        min_score = config.MIN_MATCH_SCORE if min_score is None else min_score

        query = compact_key(text)
        if len(query) < MIN_QUERY_LEN or query in GENERIC_WORDS:
            return []

        variants = {query}
        core = _core(query)
        if core:
            variants.add(core)

        # several keys (and several names) can lead to the same street: keep the best per name
        best = {}
        for variant in variants:
            hits = process.extract(
                variant, self.keys,
                scorer=Levenshtein.normalized_similarity,
                limit=limit * 4, score_cutoff=min_score,
            )
            for key, score, _ in hits:
                for name, kind in self.key_to_entries[key]:
                    if name not in best or score > best[name][0]:
                        best[name] = (score, kind)

        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:limit]
        return [
            {"name": name, "match_score": round(score, 3), "match_type": kind, **self.name_stats[name]}
            for name, (score, kind) in ranked
        ]


if __name__ == "__main__":
    import sys

    from .data_loader import load_roads

    index = StreetIndex(load_roads())
    queries = sys.argv[1:] or ["mgroad", "M.G. Rd", "MG R0AD", "cubbon rd", "1st cross", "palaceroad", "xyzxyz"]
    for q in queries:
        print(f"\ncue: {q!r}")
        results = index.match(q)
        if not results:
            print("   (no candidates)")
        for r in results:
            print(f"   {r['match_score']:.2f}  {r['match_type']:8}  {r['name']}  "
                  f"[{r['segments']} segments, {r['length_m']} m]")
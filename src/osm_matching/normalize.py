"""Text normalization for OCR cues and OSM street names.

normalize_text() cleans any text the same way, whether it comes from OSM or OCR.
compact_key() removes spaces, because the OCR model cannot output spaces.
name_keys() gives every key an OSM name should be findable by.
"""
import re
import unicodedata

# Road-type abbreviations that always mean the same thing.
ABBREVIATIONS = {
    "rd": "road",
    "ln": "lane",
    "ave": "avenue",
    "blvd": "boulevard",
    "hwy": "highway",
    "cir": "circle",
    "blk": "block",
    "cr": "cross",
    "crs": "cross",
    "mn": "main",
}

# Ambiguous abbreviations: (meaning when last word, meaning elsewhere).
POSITIONAL = {
    "st": ("street", "saint"),
    "dr": ("drive", "doctor"),
}

# Words that mark the end of a road name (used to build the initials variant).
ROAD_TYPES = {
    "road", "street", "lane", "avenue", "boulevard", "highway", "circle",
    "cross", "main", "marg", "veedhi", "path",
}


def _ascii_fold(text):
    """Drop accents and non-Latin characters (e.g. Kannada script becomes empty)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _merge_single_letters(tokens):
    """['m', 'g', 'road'] -> ['mg', 'road']  (initials split by spaces or dots)."""
    out, buf = [], ""
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            buf += tok
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(tok)
    if buf:
        out.append(buf)
    return out


def _expand(tokens):
    out = []
    last = len(tokens) - 1
    for i, tok in enumerate(tokens):
        if tok in ABBREVIATIONS:
            out.append(ABBREVIATIONS[tok])
        elif tok in POSITIONAL:
            out.append(POSITIONAL[tok][0 if i == last else 1])
        else:
            out.append(tok)
    return out


def normalize_text(text):
    """Lowercase, strip punctuation, merge initials, join ordinals, expand abbreviations."""
    if not text:
        return ""
    text = _ascii_fold(str(text)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)                     # punctuation -> space
    text = re.sub(r"\b(\d+)\s+(st|nd|rd|th)\b", r"\1\2", text)  # "1 st" -> "1st"
    tokens = _merge_single_letters(text.split())
    return " ".join(_expand(tokens))


def compact_key(text):
    """Normalized text with all spaces removed: 'MG Road' -> 'mgroad'."""
    return normalize_text(text).replace(" ", "")


def _tokens_unmerged(text):
    """Like normalize_text but keeps single letters separate: 'B.V.K. Road' -> ['b','v','k','road']."""
    if not text:
        return []
    text = _ascii_fold(str(text)).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(\d+)\s+(st|nd|rd|th)\b", r"\1\2", text)
    return _expand(text.split())


def _initials_variant(tokens):
    """['mahatma','gandhi','road'] -> 'mgroad'. Only for <words...> <road type>."""
    if len(tokens) < 3 or tokens[-1] not in ROAD_TYPES:
        return None
    lead = tokens[:-1]
    if not all(t.isalpha() and t not in ROAD_TYPES for t in lead):
        return None
    return "".join(t[0] for t in lead) + tokens[-1]


def name_keys(name):
    """Compact keys an OSM name should be findable by, including meaningful word spans."""
    normalized = normalize_text(name)
    if not normalized:
        return set()
    keys = {normalized.replace(" ", "")}
    tokens = _tokens_unmerged(name)
    # Index contiguous subphrases too: OCR may see only the distinctive part of
    # a mapped name ("Library" on a building mapped as "New Library").
    generic = ROAD_TYPES | {"building", "department", "block", "hostel", "center", "centre"}
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            span = tokens[start:end]
            if any(token in generic for token in span):
                continue
            key = "".join(span)
            if len(key) >= 4:
                keys.add(key)
    variant = _initials_variant(tokens)
    if variant:
        keys.add(variant)
    return keys


if __name__ == "__main__":
    # Inspect how the normalizer behaves on the real OSM data.
    from .data_loader import load_roads

    names = sorted(load_roads()["name"].dropna().unique())
    empty = [n for n in names if not normalize_text(n)]
    with_variant = [n for n in names if len(name_keys(n)) > 1]

    print(f"Unique OSM names                         : {len(names)}")
    print(f"Names that normalize to empty (non-Latin): {len(empty)}")
    for n in empty[:10]:
        print("   ", n)

    print(f"\nNames that also get an initials key      : {len(with_variant)}")
    for n in with_variant[:15]:
        print(f"   {n!r:40} -> {sorted(name_keys(n))}")

    print("\nSample of normalized names:")
    for n in names[:15]:
        print(f"   {n!r:40} -> {normalize_text(n)!r}")

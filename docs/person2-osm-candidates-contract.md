# Person 2 → Person 4: OSM candidate contract

Function: `CandidateGenerator().match_cue(cue)` in `src/osm_matching/candidates.py`.
Example outputs (placeholder area, not the final corridor): `docs/examples/person2_candidates_sample.json`.

## Input (from Person 1)

```json
{"timestamp_s": 12.4, "cue_type": "street_sign", "text": "MG ROAD", "ocr_confidence": 0.91, "bbox_xywh": [120, 75, 240, 58]}
```

`src/osm_matching/mock_cues.py` also converts the JSON written by `openvino_webcam_test.py` into this format.

## Output

```json
{
  "timestamp_s": 8.0,
  "query_text": "CUBBON RD",
  "normalized_text": "cubbon road",
  "ocr_confidence": 0.88,
  "ambiguous": false,
  "n_candidates_total": 1,
  "truncated": false,
  "candidates": [
    {
      "candidate_id": "way/362900185",
      "name": "Cubbon Road",
      "osm_type": "way",
      "osm_id": 362900185,
      "match_score": 1.0,
      "match_type": "full",
      "representative_lat": 12.97831,
      "representative_lon": 77.60517,
      "same_name_pieces": 1,
      "length_m": 4144.0
    }
  ]
}
```

Each candidate also has `osm_way_ids`, `bbox` and `segments` (omitted above for space).

## Fields

| Field | Meaning |
|---|---|
| `match_score` | Text similarity between the cue and the OSM name, 0 to 1. **Not** the probability that this is the correct place. |
| `match_type` | `full`: the cue matches the whole street name. `initials`: it matches the initials form (e.g. "MG Road" for "Mahatma Gandhi Road"). |
| `candidate_id`, `osm_id` | OSM way id of the longest segment of this street piece. |
| `osm_way_ids` | All OSM way ids in this piece. |
| `representative_lat/lon` | A point that lies on the road, near the centre of the piece (WGS84 degrees). |
| `bbox` | `[min_lon, min_lat, max_lon, max_lat]` of the piece. |
| `segments` | The road geometry: a list of polylines, each a list of `[lat, lon]` pairs. Note the order. Omitted when `include_geometry=False`. |
| `length_m` | Total length of the mapped segments in the piece. |
| `same_name_pieces` | How many separate pieces share this street name. |
| `ambiguous` | True when more than one candidate scores within 0.05 of the best. |
| `truncated` | True when more than 30 candidates matched and the list was cut. `n_candidates_total` is the full count. |
| `ocr_confidence` | Passed through unchanged from Person 1. |

## Things to know

- A street name is often many separate streets (e.g. "1st Cross Road" gives 23 pieces). Each piece is its own candidate, and they all share one `match_score`. Use previous belief and movement to choose between them.
- Several pieces do not always mean several different roads. A junction or underpass can split one real road into two pieces.
- Treat the candidates of one cue as **alternative explanations of the same observation**, not as independent observations.
- No candidates (`candidates: []`) means no name scored at or above 0.75, or the text was too short or only generic words (`road`, `cross`, ...). It can also mean OSM lacks that feature, not that OCR failed.
- Matching uses only OSM data. No GPS is involved.
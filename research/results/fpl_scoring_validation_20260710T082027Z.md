# FPL Scoring-Rule Validation - 20260710T082027Z

Recomputed the points for **2085 real scored matches** across 60 players (top 15 per position with 900+ minutes), and compared with the points FPL actually awarded.

## Result

| | |
|---|---|
| Matches checked | 2085 |
| Reconstructed exactly | **2085** |
| **Exact-match rate** | **100.0000%** |
| Mismatches | 0 |

**Every single match reconstructs exactly.** The scoring rules in `prediction_engine/fpl/scoring.py` are correct, including the new 2025/26 defensive-contribution rule. Every projection built on them inherits that correctness.
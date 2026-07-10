# Market Edge Benchmark - 20260710T063543Z

League: EPL. 2660 matches with both Pinnacle OPENING and CLOSING odds.

Four questions: what is late information worth (T1); can we beat the opening line (T2); do we carry signal the closing line misses (T3); and does the line move toward us when we disagree (T4 — closing line value)?

> **Integrity note:** the T3 blend is a measurement instrument, not a shipped model. Odds remain a yardstick, never an input to the champion.

## Comparison

| model                    |   n_predictions |   log_loss |    rps |   brier_score |   ece_home |   ece_draw |   ece_away |   total_runtime_seconds |   mean_fit_seconds_per_fold |
|:-------------------------|----------------:|-----------:|-------:|--------------:|-----------:|-----------:|-----------:|------------------------:|----------------------------:|
| market_close             |       2660.0000 |     0.9464 | 0.1939 |        0.5591 |     0.0249 |     0.0144 |     0.0161 |                  0.0000 |                      0.0000 |
| blend_engine_plus_market |       2660.0000 |     0.9466 | 0.1940 |        0.5593 |     0.0265 |     0.0141 |     0.0164 |                  0.0000 |                      0.0000 |
| market_open              |       2660.0000 |     0.9503 | 0.1952 |        0.5619 |     0.0232 |     0.0130 |     0.0124 |                  0.0000 |                      0.0000 |
| ensemble                 |       2660.0000 |     0.9821 | 0.2059 |        0.5838 |     0.0228 |     0.0074 |     0.0223 |                  0.0000 |                      0.0000 |

## Significance (paired, per-match log loss)

| model_a      | model_b                  |   n_matches |   mean_log_loss_diff |   paired_t_pvalue |   wilcoxon_pvalue |
|:-------------|:-------------------------|------------:|---------------------:|------------------:|------------------:|
| ensemble     | market_open              |        2660 |               0.0318 |            0.0000 |            0.0000 |
| ensemble     | market_close             |        2660 |               0.0357 |            0.0000 |            0.0000 |
| ensemble     | blend_engine_plus_market |        2660 |               0.0354 |            0.0000 |            0.0000 |
| market_open  | market_close             |        2660 |               0.0039 |            0.0043 |            0.0000 |
| market_open  | blend_engine_plus_market |        2660 |               0.0036 |            0.0060 |            0.0002 |
| market_close | blend_engine_plus_market |        2660 |              -0.0002 |            0.4499 |            0.0000 |

## Findings

**T1 - value of late information.** Opening line 0.9503 -> closing line 0.9464. The market improves by 0.0039 between the price going up and kickoff. That is what team news + sharp money are worth, and it bounds what any team-news feature could buy us.

**T2 - can we beat the OPENING line?** engine 0.9821 vs opening 0.9503 -> no, the opening price is already sharper than us.

**T3 - do we add information the market missed?** Blending the engine into the closing line scores 0.9466 vs the market's 0.9464 (not significant: t p=0.4499, Wilcoxon p=0.0000). Average fitted weight on our engine: **4%** (per season: 2018-19=0%, 2019-20=0%, 2020-21=12%, 2021-22=8%, 2022-23=7%, 2023-24=1%, 2024-25=0%).
  -> The blend does not improve on the market. Our forecast appears to be a strict information subset of the closing line - no exploitable edge here.

**T4 - Closing Line Value.** Correlation between our disagreement with the open and the line's subsequent movement: **r=-0.007** (p=0.54). On the 5719 strong disagreements, the line moved toward us **48.5%** of the time.
  -> No meaningful anticipation of the market's move.

## Closing Line Value detail

```
{
  "n_matches": 2660,
  "correlation": -0.006882845253194951,
  "p_value": 0.5387123370315697,
  "share_line_moved_toward_us": 0.4853995453750656,
  "n_strong_disagreements": 5719,
  "mean_abs_movement": 0.016603211959634538
}
```
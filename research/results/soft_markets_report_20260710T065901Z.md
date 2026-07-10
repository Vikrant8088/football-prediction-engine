# Soft-Market Benchmark (Phase 4d) - 20260710T065901Z

Can the engine beat a closing line where the market is less efficient? Matches and Pinnacle closing odds come from the same football-data.co.uk file, so there is no cross-source join. Understat does not cover these divisions, so the engine is goal-only: **Elo** is the probe (it carries ~72% of the champion ensemble's weight and, on the EPL, scores within 0.004 of it). The Premier League is the sharp-market reference row, scored with the identical model.

`gap_to_market` = engine log loss - closing-line log loss. **Negative means the engine wins.** `edge_captured` = share of the baseline->market span the engine covers.

## Results (sorted: closest to beating the market first)

|                |   n_matches |   overround |   baseline |   engine (elo) |   closing line |   gap_to_market |   edge_captured |   elo_accuracy |   market_accuracy |   paired_t_pvalue |
|:---------------|------------:|------------:|-----------:|---------------:|---------------:|----------------:|----------------:|---------------:|------------------:|------------------:|
| Bundesliga 2   |   2448.0000 |      0.0289 |     1.0795 |         1.0723 |         1.0522 |          0.0202 |          0.2633 |         0.4379 |            0.4542 |            0.0000 |
| Segunda        |   3672.0000 |      0.0296 |     1.0684 |         1.0595 |         1.0337 |          0.0258 |          0.2564 |         0.4412 |            0.4695 |            0.0000 |
| Championship   |   4414.0000 |      0.0259 |     1.0769 |         1.0643 |         1.0338 |          0.0304 |          0.2929 |         0.4472 |            0.4730 |            0.0000 |
| League Two     |   4299.0000 |      0.0314 |     1.0795 |         1.0834 |         1.0484 |          0.0350 |         -0.1271 |         0.4110 |            0.4603 |            0.0000 |
| Scottish Prem  |   1774.0000 |      0.0288 |     1.0673 |         0.9694 |         0.9314 |          0.0380 |          0.7206 |         0.5248 |            0.5626 |            0.0000 |
| Premier League |   3040.0000 |      0.0238 |     1.0667 |         0.9847 |         0.9457 |          0.0390 |          0.6774 |         0.5286 |            0.5572 |            0.0000 |
| Serie B        |   3075.0000 |      0.0288 |     1.0860 |         1.0831 |         1.0439 |          0.0392 |          0.0685 |         0.4062 |            0.4530 |            0.0000 |
| Ligue 2        |   2865.0000 |      0.0312 |     1.0851 |         1.0775 |         1.0371 |          0.0403 |          0.1595 |         0.4241 |            0.4719 |            0.0000 |
| League One     |   4257.0000 |      0.0310 |     1.0762 |         1.0722 |         1.0198 |          0.0524 |          0.0724 |         0.4416 |            0.4954 |            0.0000 |

## Verdict

**The closing line beats the engine in every league tested.** Even where the bookmaker's own margin says it is unsure, its price is still sharper than a public-data model. There is no soft market here - at least not one reachable with goals-only Elo.

Correlation between the bookmaker's margin (overround) and our gap to its price: **r = 0.178**. There is no clear relationship between the bookmaker's stated uncertainty and our ability to close on its price.

## Caveats

- Elo only. The full goal-only ensemble (Elo + Poisson + Dixon-Coles) would score a little better; any league that looks close here deserves that deeper run.
- Beating a closing line on log loss is NOT the same as beating it after the bookmaker's margin. A profitable bet needs to beat the price, not the probability - the overround is the toll.
- Odds are a yardstick, never a model input.
# Gate B: recent-form minutes on the FPL pts/GW edge - 20260714T073138Z

Pre-registered primary: GBP100m squad + captain, 8 Understat-xG seasons (263 GW). The shipped crude-minutes model beats player_ppg by **+5.02 pts/GW** (the Phase 5e headline). Gate A already proved recent-form is a much better *minutes* predictor; the question here is whether that converts into a better *edge*.

## Each configuration's own edge over player_ppg

| config | gain/GW vs ppg | t p | Wilcoxon p |
|---|---|---|---|
| crude (shipped) | **+5.02** | 0.0000 | 0.0000 |
| recent-form HL=1 | **+6.74** | 0.0000 | 0.0000 |
| recent-form HL=2 | **+7.94** | 0.0000 | 0.0000 |

## The decisive test: challenger vs crude, head-to-head

| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | Holm survives? | non-DC gain/GW | ships? |
|---|---|---|---|---|---|---|---|
| recent-form HL=1 | **+1.749** | 132/123 | 0.0434 | 0.1527 | no | +1.930 | no |
| recent-form HL=2 | **+2.947** | 147/106 | 0.0002 | 0.0016 | yes | +3.035 | **YES** |

## Per-season replication (head-to-head vs crude)

| season | recent-form HL=1 | recent-form HL=2 |
|---|---|---|
| 2018-19 | -0.061 | +2.667 |
| 2019-20 | +5.152 | +4.576 |
| 2020-21 | +2.758 | +3.788 |
| 2021-22 | -1.970 | -1.879 |
| 2022-23 | +1.219 | +0.062 |
| 2023-24 | +2.939 | +8.545 |
| 2024-25 | +3.455 | +3.394 |
| 2025-26 | +0.485 | +2.333 |

## Verdict

**The recent-form minutes model clears the bar** (recent-form HL=2): it beats the shipped crude model by +2.947 pts/GW head-to-head, significant on both tests after Holm correction, and stays positive without the DC-rule season. A genuinely better minutes signal that also grows the edge - adopt it and re-run the Phase 5e headline.

**Robustness of the best arm (recent-form HL=2):** positive in 7/8 seasons; pooled +2.947/GW, dropping its best season (2023-24) leaves +2.143/GW.

## Caveats

- Head-to-head isolates ONLY the minutes model: identical team model, player rates, optimizer, budget, captain and player_ppg baseline.
- **Backtest handicap:** live, the projection also has FPL's injury/availability flag, the single biggest minutes signal - this backtest cannot (flags are not published historically). So this UNDERSTATES the live value of a minutes model; it measures only the recent-form part.
- Distinct squad-cache tags per frame; the crude arm reuses the Phase 5e prediction cache unchanged.
# Phase 6a: Training-window x time-decay screen - 20260713T132555Z

League: EPL (Understat). Walk-forward, evaluated on 2019-20 to 2025-26 (7 seasons, 2660 matches) so every window setting (up to 5 seasons) has full history and all cells are scored on the IDENTICAL matches.

**This is the cheap SCREEN (Stage 1), goal models only.** It isolates the scoreline-grid channel that feeds the FPL edge - clean-sheet probability - which the prior expanding-window, 1X2-only tuning pass never tested. Screen-passing cells (if any) go to the Stage 2 FPL pts/GW primary; nothing here changes the shipped model on its own.

Lower is better for every column. `cs_log_loss` is the headline (the FPL channel); `wdl_log_loss`/`wdl_rps` guard against buying clean-sheet accuracy by wrecking the 1X2 forecast. The **default** cell (expanding window, xi=0.0065) is marked - it is what ships today.

## dixon_coles_xg

Default cell: cs_log_loss **0.5559**, wdl_log_loss 1.0168, wdl_rps 0.2137.

| window | xi | cs_log_loss | cs_brier | wdl_log_loss | wdl_rps | cs gain vs default | t p | Wilcoxon p | screen |
|---|---|---|---|---|---|---|---|---|---|
| expanding **PASS** | 0.001 | 0.5520 | 0.1849 | 1.0137 | 0.2134 | +0.00393 | 0.0297 | 0.0516 | yes |
| 5 **PASS** | 0.001 | 0.5520 | 0.1849 | 1.0113 | 0.2125 | +0.00392 | 0.0128 | 0.2651 | yes |
| 5 **PASS** | 0.0005 | 0.5522 | 0.1849 | 1.0124 | 0.2130 | +0.00378 | 0.0377 | 0.0846 | yes |
| expanding **PASS** | 0.002 | 0.5522 | 0.1850 | 1.0128 | 0.2129 | +0.00370 | 0.0089 | 0.7417 | yes |
| 5 **PASS** | 0.002 | 0.5526 | 0.1851 | 1.0113 | 0.2124 | +0.00339 | 0.0030 | 0.8956 | yes |
| 4 **PASS** | 0.001 | 0.5527 | 0.1852 | 1.0125 | 0.2129 | +0.00328 | 0.0297 | 0.3661 | yes |
| 5 **PASS** | 0.0 | 0.5528 | 0.1851 | 1.0145 | 0.2137 | +0.00319 | 0.1238 | 0.0159 | yes |
| 4 **PASS** | 0.0005 | 0.5528 | 0.1852 | 1.0136 | 0.2134 | +0.00314 | 0.0687 | 0.1619 | yes |
| expanding | 0.0005 | 0.5530 | 0.1852 | 1.0173 | 0.2146 | +0.00297 | 0.1498 | 0.0012 |  |
| 4 **PASS** | 0.002 | 0.5530 | 0.1853 | 1.0120 | 0.2126 | +0.00296 | 0.0076 | 0.9332 | yes |
| 4 **PASS** | 0.0 | 0.5532 | 0.1853 | 1.0154 | 0.2140 | +0.00272 | 0.1626 | 0.0527 | yes |
| 3 | 0.001 | 0.5540 | 0.1856 | 1.0180 | 0.2146 | +0.00192 | 0.2318 | 0.2699 |  |
| 3 | 0.002 | 0.5541 | 0.1857 | 1.0173 | 0.2143 | +0.00183 | 0.1660 | 0.6108 |  |
| 3 | 0.0005 | 0.5542 | 0.1857 | 1.0189 | 0.2149 | +0.00177 | 0.3129 | 0.1554 |  |
| 3 | 0.0 | 0.5545 | 0.1858 | 1.0201 | 0.2154 | +0.00148 | 0.4395 | 0.0796 |  |
| expanding | 0.0 | 0.5553 | 0.1862 | 1.0244 | 0.2170 | +0.00063 | 0.7887 | 0.0000 |  |
| 4 | 0.0065 | 0.5559 | 0.1865 | 1.0169 | 0.2137 | +0.00003 | 0.2947 | 0.4028 |  |
| expanding (default) | 0.0065 | 0.5559 | 0.1865 | 1.0168 | 0.2137 | +0.00000 | nan | nan |  |
| 5 | 0.0065 | 0.5560 | 0.1865 | 1.0168 | 0.2137 | -0.00001 | 0.2871 | 0.0256 |  |
| 3 | 0.0065 | 0.5564 | 0.1866 | 1.0205 | 0.2149 | -0.00044 | 0.5529 | 0.3544 |  |
| 2 | 0.002 | 0.5596 | 0.1879 | 1.0310 | 0.2189 | -0.00361 | 0.0319 | 0.0105 |  |
| 2 | 0.001 | 0.5598 | 0.1880 | 1.0319 | 0.2192 | -0.00390 | 0.0329 | 0.0038 |  |
| 2 | 0.0005 | 0.5601 | 0.1881 | 1.0326 | 0.2195 | -0.00416 | 0.0297 | 0.0021 |  |
| 2 | 0.0065 | 0.5602 | 0.1881 | 1.0312 | 0.2187 | -0.00428 | 0.0016 | 0.0000 |  |
| 2 | 0.0 | 0.5604 | 0.1882 | 1.0335 | 0.2198 | -0.00449 | 0.0248 | 0.0011 |  |

Best screen-passing cell: **window=expanding, xi=0.001** - cs_log_loss 0.5520 vs default 0.5559 (gain +0.00393/obs, t p=0.0297, Wilcoxon p=0.0516). NOT significant on both tests -> suggestive only; carry the top 1-2 cells to Stage 2 but expect a likely null.

## poisson_xg

Default cell: cs_log_loss **0.5553**, wdl_log_loss 1.0238, wdl_rps 0.2170.

| window | xi | cs_log_loss | cs_brier | wdl_log_loss | wdl_rps | cs gain vs default | t p | Wilcoxon p | screen |
|---|---|---|---|---|---|---|---|---|---|
| 5 **PASS** | n/a | 0.5528 | 0.1851 | 1.0137 | 0.2137 | +0.00256 | 0.0072 | 0.0000 | yes |
| 4 **PASS** | n/a | 0.5532 | 0.1853 | 1.0146 | 0.2140 | +0.00209 | 0.0807 | 0.0000 | yes |
| 3 **PASS** | n/a | 0.5545 | 0.1858 | 1.0194 | 0.2154 | +0.00085 | 0.6088 | 0.0000 | yes |
| expanding (default) | n/a | 0.5553 | 0.1862 | 1.0238 | 0.2170 | +0.00000 | nan | nan |  |
| 2 | n/a | 0.5604 | 0.1882 | 1.0331 | 0.2197 | -0.00512 | 0.0176 | 0.0009 |  |

Best screen-passing cell: **window=5, xi=n/a** - cs_log_loss 0.5528 vs default 0.5553 (gain +0.00256/obs, t p=0.0072, Wilcoxon p=0.0000). Significant on BOTH tests -> carry to Stage 2 FPL primary.

## What happens next

Per the pre-registration, screen-passing cells are carried to the Stage 2 FPL primary endpoint (GBP100m squad + captain gain over player_ppg, Holm-corrected). The shipped model changes only if a cell wins THERE. If no cell passes this screen, the expanding-window / xi=0.0065 default stands and the window/decay lever is recorded as a null - the honest, expected outcome given the prior 1X2 tuning null and the low FPL predictability ceiling.
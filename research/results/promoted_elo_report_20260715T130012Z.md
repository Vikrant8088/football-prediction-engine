# Phase 6e Gate A: promoted-team Elo prior - 20260715T130012Z

Season-level walk-forward on EPL Understat, eval 2018-19-2025-26 (864 promoted-team fixtures). Elo only. Lower is better; `promoted_log_loss` is the headline (the fixtures the prior acts on), `all_log_loss` guards against hurting the rest.

| promoted_penalty | promoted_log_loss | promoted_rps | all_log_loss | promoted gain vs 0 | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| 0 (default) | 0.9625 | 0.1990 | 0.9956 | +0.0000 | - | - |
| 50 | 0.9588 | 0.1974 | 0.9930 | +0.0038 | 0.2740 | 0.0117 |
| 100 | 0.9583 | 0.1975 | 0.9917 | +0.0043 | 0.3954 | 0.1449 |
| 150 | 0.9595 | 0.1981 | 0.9909 | +0.0030 | 0.6473 | 0.7723 |
| 200 | 0.9620 | 0.1990 | 0.9908 | +0.0006 | 0.9434 | 0.6756 |
| 250 | 0.9651 | 0.2002 | 0.9911 | -0.0026 | 0.7759 | 0.2866 |

## Verdict

**The prior predicts promoted fixtures better** at penalty=100: promoted log loss 0.9583 vs default 0.9625 (+0.0043), NOT significant on both - suggestive only. Carry it to Gate B (the FPL edge).
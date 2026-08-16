# Revision Log — GMC manuscript (2026-08-14)

**Paper:** GMC: Graph Multi-view Completion for Drug Repositioning (`papers/gmc_manuscript.tex`)
**Round:** 1 · **Simulated decision:** Minor Revision (7.8/10)
**Prior state:** manuscript already incorporated the five experiment blocks (overall / ablation / parameter / robustness / case study).

All numbers in this revision were verified against `Results/summaries/` (param sweep, robustness, random baseline, `gmc_unified`, `SIGNIFICANCE_paired.csv`). BH-FDR was recomputed over the 11 published baseline configurations per dataset.

---

## Revision Tracking Table

| # | Issue | Reviewer | Type | Section | Resolution | Location | Status |
|---|-------|----------|------|---------|-----------|----------|--------|
| M1 | "from the *same* evidence" is imprecise: block view embeds the cold-start-filled matrix F, tensor view takes the bare masked matrix M with its own internal fill | R1 | Major | Intro bullet 2 / Methods "Combining the Two Geometries" | Clarified: same similarity data + same completion target, but the association evidence is presented differently (verified against `gmc/model.py`: block builds T from `filled`, tensor calls `fitrpca(..., masked)`) | Methods L169; intro bullet 2 | RESOLVED |
| M2 | "combining them is more robust than either alone" overstated — ablation shows tensor fusion is neutral on F/C (C: 0.7259→0.7252, −0.0007) | R1 | Major | Intro bullet 2 / Methods L174 | Rephrased to "at least as accurate within fold-level noise, strictly better where the geometries differ most"; intro now "adds robustness precisely where the two views differ most" | Intro L79; Methods L174 | RESOLVED |
| M3 | Structural controls (observed-mask, rank-cap) run at 0.2/0.8 tensor-heavy weights, not the reported 0.5/0.5, without rationale | R2 | Major | Results Ablation | Added rationale: fixed-weight diagnostic; block-side effects are attenuated at 0.2 weight, so the reported gains are conservative. Rank-cap reconciled with Table `tab:param` (0.3548 rc200 vs 0.3693 center on CTD at reported weights) | Ablation prose L381 | RESOLVED (no re-run; 0.2/0.8 data kept) |
| M4 | No multiple-testing correction; F/C vs DNMFDDA margins are n.s. (p=0.131) | R1 | Major | sec:eval / Main Results | Added Benjamini–Hochberg correction across 11 published baselines/dataset; report FDR result (CTD/Y 11/11, F/C 10/11, sole n.s. = DNMFDDA); reported exact p=0.131 for F/C | sec:eval L258; Main Results L265 | RESOLVED |
| m1 | Param table center labeled "(reported)" but is the fresh-fold value (0.6579 ≠ 0.6569 test) | R2 | Minor | tab:param | Renamed to "center (fresh folds)" | tab:param row 1 | RESOLVED |
| m2 | eq:gmc-e shows only mean + filter toggle; text describes α/β blends for C/Y | R1 | Minor | Methods GMC-E | Generalized to avg / filt / blend with the graph operator G(M); prose already mapped configs | eq:gmc-e | RESOLVED |
| m3 | Case study table shows top-3 but 65.5%/86.2% stats are over top-10 | R2 | Minor | tab:case | Added note: full top-ten lists (40 pairs) in supporting repository | tab:case caption | RESOLVED |
| S2 | p-values cluster at 0.002; reviewer may ask why | R1 | Suggestion | sec:eval | Added: for n=10 folds, p=0.002 is the smallest two-sided Wilcoxon p (all 10 folds same direction) | sec:eval L258 | RESOLVED |
| S3 | Fold-set provenance should be prominent | R2 | Suggestion | sec:robust | Clarified 10% row reuses reported test folds; 5/20/30 rows are new masks at the same seed | sec:robust L422 | RESOLVED |
| S1 | GMC-E occupies 3 subsections for an explicitly non-method | R1 | Suggestion | throughout | **Deliberate:** GMC-E kept at 3 subsections as an upper reference quantifying fusion headroom; framing already explicit ("not a proposed method") | — | DELIBERATE_LIMITATION (author decision) |
| m4 | Placeholder metadata ([Journal Name], [DOI], [GitHub URL], authors, funding) | DA | Minor | front matter | Left to author pre-submission (not a manuscript flaw) | — | DEFERRED_TO_AUTHOR |

**Summary:** 12 items — 9 RESOLVED, 1 DELIBERATE_LIMITATION, 1 DEFERRED_TO_AUTHOR (metadata), 1 Suggestion folded into S2/S3 (S1 = deliberate). No re-runs; all changes text-only.

---

## Response to Reviewers (point-by-point, adaptable for submission)

**M1 — "same evidence" (Methods).** We agree the phrasing was loose. The two geometries do estimate the same masked-completion target from the same 5+2 similarity data, but they present the association evidence differently: the matrix geometry embeds the cold-start-filled matrix **F** in the joint block, while the tensor geometry operates on the bare masked matrix **M** with the solver's own internal WKNN fill. The sentence now states this explicitly.

**M2 — "more robust than either alone."** The ablation (Table 5) shows the tensor fusion is neutral on Fdataset/Cdataset (±0.001, within fold noise) and decisive on CTDdataset2023/Ydataset. We corrected the claim in both the Introduction and Methods to "at least as accurate within fold-level noise, and strictly better where the two geometries differ most."

**M3 — structural-control weights.** The observed-mask and rank-cap controls were evaluated at fixed tensor-heavy weights (0.2/0.8) as a diagnostic. Because these choices act on the block view, which carries only 0.2 weight in that fusion, the measured effects are attenuated — i.e. conservative. The rank-cap choice is additionally re-checked at the reported equal weights in the parameter analysis (Table 6: CTD rc200 0.3548 vs center 0.3693). The text now explains this.

**M4 — multiple comparisons.** We now report exact p-values (F/C vs DNMFDDA: p=0.131) and apply a Benjamini–Hochberg correction across the eleven published baseline configurations per dataset. After correction GMC remains significantly ahead of every baseline on CTDdataset2023 and Ydataset (11/11) and of all but DNMFDDA on Fdataset and Cdataset (10/11).

**m1–m3, S2, S3** — all addressed as in the tracking table (labeling, equation, repository note, p-value floor, fold provenance).

---

## What changed in `gmc_manuscript.tex` (10 edits)

1. Intro principle-2 closing sentence — M2
2. Methods L169 "same evidence" → same similarity data + different association presentation — M1
3. Methods L174 "more robust than either readout alone" → noise-band qualification — M2
4. sec:eval L258 — two-sided Wilcoxon, p=0.002 floor (S2), BH-FDR method (M4)
5. Main Results L265 — exact p=0.131 and FDR 11/11 / 10/11 (M4)
6. tab:param row 1 — "center (reported)" → "center (fresh folds)" (m1)
7. eq:gmc-e — avg/filt/blend equation (m2)
8. tab:case caption — full top-ten lists note (m3)
9. Ablation prose — structural-control diagnostic rationale + rank-cap reconciliation (M3)
10. sec:robust — fold provenance clarification (S3)

## Not changed (deliberate)

- **GMC-E** remains at 3 subsections as an upper reference (author decision; not a proposed method).
- **Abstract** unchanged — the FDR result could be added to the Results sentence at submission if desired.
- **Placeholder metadata** (journal/DOI/GitHub/authors/funding) — author's pre-submission task.

---

## Round 2 — narrative reframing (2026-08-14, after user review)

**Request:** stop presenting "one configuration on all four datasets" as a selling point; present **GMC as a single new model**, not a module assembly. Author's words: "统一配置是为了好叙事，你却把它放在论文里面洋洋洒洒写出来"; "GMC是研究出来的模型，不是模块的拼接".

**What changed in `gmc_manuscript.tex` (~20 text-only edits; zero re-runs, zero data changes):**

Removed every "unified-configuration-as-claim" formulation:
- Abstract: dropped "a single configuration combining..." and "One configuration is applied identically to all four datasets." → GMC defined directly as "a single nuclear-norm completion solved in two geometries of the same similarity data".
- Intro: dropped "applied under a single configuration", "GMC is one model, not a cascade", "single model rather than per-dataset module stitching", "the same configuration attains the highest AUPR on all four benchmarks", "With a single configuration".
- Methods: fig1 caption dropped "One configuration is applied to all four datasets"; "A single symmetric block ... used on all four datasets" → "used"; fusion eq dropped "no per-dataset tuning"; "In the unified configuration" → "In the rank-fused output".
- Results: dropped "With a single configuration", "GMC (proposed method, one configuration)", "a single GMC config is enough to beat", "attains the unified-config AUPR" → "attains the full-model AUPR"; ablation captions "of the unified configuration" → "of GMC".
- Parameter section: titles/captions "Parameter sensitivity of the unified configuration" → "Parameter sensitivity"; prose reframed from "The unified configuration fixes one shared hyperparameter set across all four datasets" → "GMC uses one shared hyperparameter set", conclusion "reported configuration is a robust default rather than a tuned peak" → "reported results are not a tuned peak" (kept the stability claim, which is a property of the model, not the config).
- Robustness: "the identical unified configuration" → "the identical model"; captions "unified configuration is held fixed" → "same model is held fixed"; "The unified configuration is therefore robust" → "GMC is therefore robust".
- Discussion/Limitations/Conclusion: dropped "This is what lets one configuration cover all four datasets"; limitations "the unified configuration fixes..." → "the fusion weights ... are fixed across datasets"; conclusion dropped "One configuration is applied identically to all four datasets"; sec:eval "The single GMC configuration ... was selected" → "GMC's hyperparameters ... were selected" (de-leakage statement kept, wording neutralized).

**"GMC is a researched model" — Relationship subsection (Methods L201-203) rewritten to state this POSITIVELY, with no negative definitions:**
- Old framing: "GMC sits between two families, pairing ... Combining the two---at the level of a single completion prior, not as a cascade---".
- Intermediate framing (first pass) still used a negative definition ("It is not an assembly of ... rather than as a separate smoothing stage") — the author rejected this too: no "not-a-... " definitional negatives at all.
- Final framing (purely positive): "GMC is a single model designed from one low-rank completion prior. It solves one nuclear-norm minimization in two complementary geometries of the same similarity data, and the local similarity-graph structure enters that single prior through the cold-start-restricted fill and the block geometry. ... GMC addresses both within one estimator---the low-rank prior supplies the global recovery, and the fill and block geometry supply the local structure as part of the same completion objective."
- Also removed "not a portfolio of predictors" → "Both readouts serve a single estimator of the same masked completion."

**Kept deliberately (neutral, not claims):**
- "GMC is a single completion model built on that prior" (Methods L113) — describes the model.
- "combined into one estimator" / "The goal is a single estimator, not a portfolio of predictors" (Methods) — describes the mechanism.
- GMC-E "per-dataset composition selected on the test folds" — factual de-leakage caveat for the upper reference, must stay.
- Parameter/robustness "not a tuned peak" / "not tuned to the exact 10% protocol" — neutral stability conclusions.

**Verification:** brace balance 0; all environments balanced (equation 10, table 4, figure 2, table* 4, figure* 2, algorithm 1, enumerate 1); numbers/tables/experiments untouched.

---

## Round 3 — full re-review pass (2026-08-14, after user request "再次审稿和修稿")

**Scope:** fresh critical read of the whole manuscript as a reviewer would; 6 text-only edits, zero re-runs, zero data changes. All numbers re-verified against Table 1 / Tables 2–6 / the robustness and case-study paragraphs in the source.

**Findings and resolutions:**

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| R1 | `tab:main_results` | `\multirow{14}` but each dataset block has **13 rows** (11 baselines + GMC + GMC-E) — LaTeX row-span mismatch miscenters the group label | `\multirow{14}` → `\multirow{13}` (×4) |
| R2 | Abstract | "score-level fusion … (GMC-E) **further raises** AUPR to … 0.3714 …" — on CTDdataset2023 GMC-E **equals** GMC (0.3714); "further raises" is false there | Rephrased: raises AUPR to 0.6730/0.7394/0.7522 on F/C/Y, with the CTD equality stated explicitly |
| R3 | `sec:robust` | "still above **most** published baselines" + "beats the **weaker half**" — recounting the table: at 30% mask GMC beats **4 of 11** baselines on F and on C, and 2/11 on Y; "most" and "half" both overstate (also internally inconsistent) | "comparable to several … on CTDdataset2023 the 30%-mask GMC exceeds nine of eleven … on F and C it still beats the weaker baselines (four of eleven on each)" |
| R4 | `sec:case` | "Predicted scores range 0.34–0.48, **far above the uniform-random baseline (AUPR ≈ base rate 0.01)**" — compares a **score** to an **AUPR**, dimensionally incoherent | Split the two statements: report the score range; state the random-predictor AUPR as a separate reference |
| R5 | Discussion "Why Multi-View Fusion Helps" | "the block view with the cold-start fill **already beats the completion baselines on all four datasets**" — ablation numbers are fresh-folds, baseline numbers are reported-folds (**cross-fold-set comparison**, invalid), and even within folds block+fill (0.3244) does not exceed ITRPCA on CTD | Rewrote using only within-ablation facts: full-model AUPR on F/C, most of capacity on CTD/Y, largest per-rung loss 0.2721 on CTD; tensor lift stated as within-ablation deltas |
| R6 | Ablation prose | "load-bearing rather than decorative" (colloquial) + "drops the block-only AUPR … by +0.016" (sign confusion after "drops") | "load-bearing:" + "reduces the block-only AUPR … by 0.016/0.013/0.052/0.011" |

**Kept deliberately (re-verified this pass):** all Round-1 M1–M4/m1–m3/S1–S3 resolutions; Round-2 narrative reframing (grep confirms zero "unified" mentions; the only "not a …" occurrences are the deliberate GMC-E de-leakage statements, the fill-role clarifications, "not a tuned peak", "not a propagation problem"). Citation audit re-confirmed: 38 bib entries = 38 cited keys, bidirectional, no orphans.

**Verification:** brace balance 0; all environments balanced (equation 10, table 4, figure 2, table* 4, figure* 2, algorithm 1, enumerate 1); `\multirow{13}` ×4, each dataset block has exactly 13 data rows; numbers/tables/experiments untouched.

# Patent Assessment 10 — Sim-to-Reality Evaluation Pipeline

**Date:** 2026-09-14
**Analyst role:** patent / prior-art research
**Subject code in repo:** `regulars/judge/calibration.py`, `regulars/eval/splits.py`, `regulars/eval/correlation.py`, `regulars/eval/mixed.py`, `regulars/eval/gate.py`
**Design source:** `docs/technical-architecture.md` §5 "Calibration", §6 "The evaluation design"

**Verdict up front:** Not independently patentable. Every component is textbook applied statistics with decades of published prior art, and *Recentive Analytics v. Fox Corp.* (Fed. Cir., Apr. 18, 2025) squarely forecloses claims that apply established statistical/ML methods to a new data environment. The pipeline is a real competitive asset, but the asset is execution and credibility, not exclusion rights. One narrow sub-piece (the dyadic double-sided split with straddler exclusion) is the only thing worth even a defensive publication.

---

## Mechanism Summary

Regulars produces a compatibility score for a directed pair of people (`y_hat`) from an LLM simulation of a conversation between two synthetic personas. Nothing about that score is grounded in reality until a pipeline connects it to what actually happened at a dinner. That pipeline is four ordered stages plus a gate.

**Stage 0 — the connective tissue: a permanent randomized exploration arm.** Roughly 30% of six-person dinner groups are assembled at random (`assembly_method = 'random'`); the remaining 70% are model-assembled and serve users. Outcomes observed on model-assembled groups are contaminated by the model's own selection, so the random arm is the only unbiased evaluation set. This is documented as §6 Problem 1 and implemented as `exploration_arm_only()` in `regulars/eval/correlation.py:47`, which filters to `assembly_method == "random"`.

**Stage 1 — isotonic calibration with a hard sample floor.** `IsotonicCalibrator` (`regulars/judge/calibration.py`) fits a monotone non-decreasing map from raw `y_hat` to observed `P(would_see_again)` using PAVA, then reports Brier score and a ten-bin reliability table. `MIN_CALIBRATION_N = 30` is a *refusal*, not a warning: `fit()` raises below 30 labeled dyads. The stated rationale is specific — PAVA's step count grows roughly as n^(1/3), each step height is the pooled mean of the binary labels inside it, so below ~30 labels you get two or three steps averaging a handful of Bernoulli draws, standard errors near ±0.2, and a reliability table with three observations per bin. "That is not calibration, it is a lookup table of noise."

**Stage 2 — person-grouped (not pair-grouped) cross-validation.** `person_grouped_folds()` (`regulars/eval/splits.py`) assigns *users* to folds, then classifies each dyad: test if **both** rater and ratee are in the test fold, train if **neither** is, and **dropped** otherwise. Straddling dyads are discarded because either side of a straddler leaks one person's actor or partner effect into the other partition. The cost is quantified rather than hidden — `FoldSplit.dropped_share` reports the realized loss (about 2/k·(1−1/k), i.e. 32% at k=5) — and `assert_no_leakage()` fails loudly if any user appears on both sides.

**Stage 3 — cluster-robust inference on the headline number.** `spearman_clustered()` (`regulars/eval/correlation.py`) computes Spearman ρ of sim score against real rating and obtains the SE by resampling whole `group_id` clusters with replacement (2000 reps), deliberately *not* nesting a person-level bootstrap on top ("the group is the coarsest, dominant cluster"). It simultaneously computes the naive i.i.d. bootstrap SE and reports both, plus a derived `design_effect = (se/naive_se)²` and `n_eff ≈ n/deff`. The headline ρ is computed on the exploration-arm subset only — Stage 0 is what makes Stage 3 mean anything.

**Stage 4 — crossed random effects as an artifact check.** `fit_mixed()` (`regulars/eval/mixed.py`) fits `y ~ y_hat` in statsmodels `MixedLM` with a random intercept for `group_id` and variance components for rater and ratee (`vc_formula={"rater": "0 + C(rater)", "ratee": "0 + C(ratee)"}`), returning the slope, its SE and p-value, and the four variance components with a `share()` accessor. This is the Social Relations Model working form: if all the signal lives in the ratee (partner) component, the product is a charisma detector, not a matching engine.

**The gate.** `regulars/eval/gate.py` binds the stages to a go / fix / stop verdict, where the overall verdict is the *worst* metric. `SPEARMAN_GO = 0.35`, `SPEARMAN_STOP = 0.15`, evaluated on the exploration arm with clustered SEs, alongside round-2 attendance, show rate, `MIN_YHAT_STD = 0.05` (the sim must discriminate at all), SRM relationship-variance significance, and a +15pp model-vs-random retention lift. This is the layer that turns a statistics checklist into a system: the pipeline decides whether the company is permitted to state a correlation number at all.

---

## Prior Art Found

### A. Isotonic calibration — extensive, foundational, not ownable

| Reference | Relevance |
|---|---|
| Zadrozny & Elkan, "Transforming classifier scores into accurate multiclass probability estimates" (KDD 2002) | The origin of isotonic regression for score calibration. Pure anticipation of Stage 1's core. |
| Platt (1999), Niculescu-Mizil & Caruana (ICML 2005) | Established the calibration-method family and the standard finding that **isotonic needs more data than Platt scaling and overfits on small samples**. The n=30 floor is a direct, documented consequence of this known property, not a discovery. |
| scikit-learn `sklearn.isotonic.IsotonicRegression` / `CalibratedClassifierCV(method="isotonic")` | The repo literally imports this. Claiming the technique is claiming a library call. |
| Silva Filho et al., "Classifier calibration: a survey" (arXiv:2112.10327) | Confirms reliability diagrams, Brier decomposition, and binning as standard reporting practice — Stage 1's `reliability_table()` and `brier()`. |
| [US12032919B1](https://patents.google.com/patent/US12032919B1/en) — Snowflake, filed Aug 2023, granted | Post-hoc calibration of **LLM confidence scores**. Notably, Snowflake claims *polynomial regression* and expressly distinguishes isotonic regression as the pre-existing conventional alternative. Two lessons: (i) even a large company could not claim isotonic itself and had to invent around it; (ii) there is now a granted patent in the adjacent space of calibrating LLM-produced scores. |
| [US20160341649A1 / WO2015095667A1](https://patents.google.com/patent/US20160341649) — "Particle score calibration" | Isotonic/monotone score calibration claimed only in a tightly bounded physical instrumentation context. Illustrates the only way this technique reaches allowance: heavy domain-apparatus limitation. |

### B. Grouped / leakage-preventing cross-validation — standard, and the *dyadic* case is also published

| Reference | Relevance |
|---|---|
| scikit-learn `GroupKFold`, `GroupShuffleSplit`, `LeaveOneGroupOut` | Group-aware CV as a shipped, documented primitive. |
| **Park & Marcotte, "Flaws in evaluation schemes for pair-input computational predictions," Nature Methods 9:1134 (2012)** | **The closest prior art to Stage 2, and it is close.** For pair-input prediction (protein–protein interaction, drug–target), it formally partitions test pairs into C1 (both members seen in training), C2 (one seen), C3 (neither seen) and shows that reporting anything but C3 inflates performance. Stage 2's "test only if both in test fold, train only if neither, drop the rest" **is** the C3/C1 partition with C2 discarded. Published 14 years ago in a top-tier journal. |
| Kaufman, Rosset, Perlich & Stitelman, "Leakage in data mining" (TKDD 2012) | Canonical formulation of leakage and the discipline of splitting on the entity that generates dependence. |
| Cold-start evaluation protocols in recsys (user-cold / item-cold / both-cold splits) | The same double-sided split logic, standard in recommender evaluation for well over a decade. |
| Participant-aware / subject-wise CV literature in clinical ML and digital health (e.g. [JMIR AI 2026 comparative study](https://ai.jmir.org/2026/1/e87728)) | Person-level folds to prevent within-subject leakage are the accepted norm in repeated-measures ML. |
| [US8170841B2](https://patents.google.com/patent/US8170841) — "Predictive model validation" | Generic k-fold CV inside a claimed validation system. Shows CV mechanics were already inside patent claims by 2012. |

**Nothing in the searched patent corpus claims dyadic double-sided grouped splitting.** But absence of a patent is not novelty when Nature Methods published the method.

### C. Cluster-robust standard errors — pure statistics, essentially zero patent activity

| Reference | Relevance |
|---|---|
| Liang & Zeger (1986); White (1980); Moulton (1986) | Clustered/robust variance estimation. Foundational econometrics. |
| Cameron, Gelbach & Miller, "Bootstrap-based improvements for inference with clustered errors" (RESTAT 2008) | The cluster bootstrap exactly as Stage 3 implements it: resample whole clusters with replacement, take the bootstrap SD as the SE. Also the standard caution about few clusters, which is why `MIN_CLUSTERS = 2` is generous to the point of being only a guardrail. |
| Kish design effect (1965); `deff`, `n_eff` | `CorrResult.design_effect` and the `n_eff` note are a 60-year-old survey-statistics construct. |
| Cluster-robust bootstrap literature (arXiv:1407.7166, arXiv:2308.10138) | Ongoing methodological refinement; confirms this is active academic territory, i.e. public domain. |

Targeted patent searching for cluster-robust SEs returned **no relevant patents at all**. That is meaningful: this is the kind of method no one attempts to claim, because it is a mathematical formula in the *Benson/Flook/Alice* sense.

### D. Randomized exploration arm for unbiased evaluation — prior art in both literature and patents

| Reference | Relevance |
|---|---|
| Li, Chu, Langford & Wang, "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms" (WSDM 2011, arXiv:1003.5956) | The canonical result: logs generated by a **random** policy give unbiased offline estimates. Stage 0's entire justification, published 2011. |
| Bottou et al., "Counterfactual reasoning and learning systems" (JMLR 2013) | Randomized exploration traffic as permanent infrastructure for unbiased causal estimation in a production ad system. |
| Yahoo/Coat/Open Bandit "randomized test set" debiasing line of work; [arXiv:1604.03506](https://arxiv.org/pdf/1604.03506) (unbiased data collection via exploration for personalization) | Explicitly frames a persistent random arm as the unbiased evaluation set alongside a serving arm — the 30/70 design. |
| [US11755947B1](https://patents.google.com/patent/US11755947B1/en) — Amazon, filed Dec 2019, granted Sep 2023 | Offline evaluation of ML models using experimental/replay traffic with noise reduction. Adjacent, though its claims target trigger-based noise reduction rather than unbiasedness. |
| [US8769094B2](https://patents.google.com/patent/US8769094) — "Managing sample selection bias" | Selection-bias management claimed as a system. |
| Dynamic recommendation patent with "randomized exploration lanes" (USPTO 12513350) | Randomized exploration slots in production traffic, claimed. |

### E. Mixed-effects / crossed random effects / SRM — psychometrics, fully public

| Reference | Relevance |
|---|---|
| Kenny & La Voie (1984); Kenny, Kashy & Cook, *Dyadic Data Analysis* (2006); [davidakenny.net/srm](https://davidakenny.net/srm/soremo.htm) | The Social Relations Model: actor / partner / relationship variance decomposition on round-robin designs. Stage 4 is an implementation of it. |
| Schönbrodt, Back & Schmukle, "TripleR" (Behav. Res. Methods, 2012) | An R package that does this, named in `technical-architecture.md` §5 itself. |
| Kluger, Ackerman, Kenny, Malloy & Eastwick, *AMPPS* 2025 — SRM for asymmetric block designs | Current tutorial-level treatment; the method is being taught, not guarded. |
| **Joel, Eastwick & Finkel, "Is romantic desire predictable? Machine learning applied to initial romantic attraction," Psychological Science 28(10):1478 (2017)** | Two speed-dating studies, 100+ self-report measures, random forests: **actor and partner variance modestly predictable; relationship variance not predictable at all.** This is both the intellectual motivation for Stage 4 and prior art establishing the diagnostic. Anyone in this field knows to run it. |
| statsmodels `MixedLM` with `vc_formula`; `lme4::lmer(y ~ x + (1|rater) + (1|ratee) + (1|group))` | Crossed random intercepts are a one-line model specification in two mainstream libraries. |

### F. Pipelines/gates that validate a model before it is allowed to ship

| Reference | Relevance |
|---|---|
| [US20190147371A1 → US10990901B2](https://patents.google.com/patent/US20190147371A1/en) — Accenture, filed Aug 2018 | **The closest structural prior art to the gate concept.** A validation platform that separates training/scoring data, removes bias, evaluates models against metrics, and validates before production deployment — expressly including ensuring "the scoring data includes enough observations to achieve statistical significance for evaluation results." A sample-size precondition on reporting an evaluation result is already inside a granted claim. |
| US12086053 — "Model validation as a service" | Claimed service for evaluating models for bias, robustness, explainability, regulatory compliance. |
| [US11263188B2](https://patents.google.com/patent/US11263188B2/en) — IBM, AI model documentation across lifecycle | Policy-driven collection of model facts through build/test/deploy/monitor, i.e. governance gating as a claimed system. |
| [US11544411B2](https://patents.google.com/patent/US11544411B2/en) — ML model validation and authentication | Gating within model pipelines. |
| Healthcare/public-health small-numbers suppression rules (WA DOH, Oregon OHA, CMS MIPS minimum-denominator rules) | **Directly relevant to the n=30 floor.** Refusing to report a rate below a minimum denominator is a decades-old, institutionalized reporting convention across government statistics. "Refuse to publish below minimum n" is the opposite of novel. |

### G. The application domain

| Reference | Relevance |
|---|---|
| [US8782038B2](https://patents.google.com/patent/US8782038) (eHarmony) — online compatibility matching and ranking; eHarmony's broader portfolio | Compatibility scoring, satisfaction approximation, and match ranking are long claimed in this domain. |
| [US20110138305A1](https://patents.google.com/patent/US20110138305) — random matching with real-time compatibility assessment | Randomized pairing in a dating context. |
| "Conversational social network" (USPTO 11695813, 12028392) | Compatibility matrices built up over accumulated conversations. |
| LLM user-simulator literature: SimUSER (arXiv:2504.12722), generative user simulators for conversational recsys (arXiv:2403.09738), AlignUSER (arXiv:2601.00930), value-based LLM agent simulation of mutual trust and interpersonal closeness (PMC12644628) | A fast-growing public literature on simulating users with LLMs **and validating the simulator against real human behavior**. The general idea "simulate with LLM personas, then check the simulation against reality" is in the open literature and accelerating. |

---

## Novelty Analysis

### What is definitively not novel

Each of the four stages, considered alone, is anticipated outright:

1. **Isotonic calibration of scores to probabilities** — Zadrozny & Elkan 2002, and a scikit-learn import in our own code.
2. **Refusing to calibrate below a sample floor** — a documented consequence of isotonic's known small-sample overfitting (Niculescu-Mizil & Caruana 2005), and structurally identical to minimum-denominator suppression rules used by health departments and CMS. Even the specific idea of conditioning an evaluation result on sufficient observations for statistical significance is inside Accenture's granted claim.
3. **Person-grouped CV to prevent identity leakage** — `GroupKFold`; and the *dyadic* two-sided version is Park & Marcotte's C3 class, Nature Methods 2012.
4. **Cluster-robust / cluster-bootstrap SEs** — Liang & Zeger 1986, Cameron–Gelbach–Miller 2008.
5. **Crossed random intercepts for rater, ratee and group** — the Social Relations Model, Kenny 1984 onward; a one-line `lme4` formula.
6. **A permanent randomized exploration arm as the unbiased evaluation set** — Li et al. 2011, Bottou et al. 2013, plus granted patents on randomized exploration lanes.

There is no argument available that any single stage is new. Attempting to claim one would be frivolous.

### What is arguably distinctive (and why it still probably fails)

**(i) The dyadic double-sided split with explicit straddler accounting.** This is the most technically interesting piece in the repo. Standard `GroupKFold` assumes each record belongs to one group; a directed dyad belongs to *two* (rater and ratee), so no stock splitter applies. The implementation partitions on a two-sided predicate and — importantly — **quantifies and surfaces the discarded fraction** via `dropped_share`, with a closed-form expectation of 2/k·(1−1/k). Surfacing the cost is a genuine engineering decision that most implementations skip. But the split rule itself is Park & Marcotte C3; the closed-form drop rate is elementary probability; and "report the fraction you dropped" is not an inventive concept.

**(ii) The exploration-arm restriction as a hard precondition on the headline correlation.** The composition — *the correlation you are permitted to quote is computed only on the randomized arm, with cluster-robust SEs, and gated against a threshold* — is the pipeline's real intellectual content. The arm makes the estimate unbiased; the clustering correction makes its precision honest; the gate makes the number consequential. Each link is individually old. Under KSR, combining known elements according to their known functions to yield predictable results is obvious, and every link here is a textbook pairing (exploration → unbiasedness; clustering → deflated SE; gate → go/no-go).

**(iii) The ordering as mandatory rather than advisory.** Stage 1 before Stage 3 (calibrate before you correlate), Stage 0 filtering Stage 3's input, Stage 4 as an artifact check on Stage 3's result, gate verdict = worst metric. This is the "system, not a checklist" argument. It is the best framing available, and it is still weak: the ordering is dictated by the statistics themselves (you cannot check whether a correlation is a clustering artifact before you have the correlation), which makes it the natural, and therefore obvious, sequence.

**(iv) The specific constants.** `MIN_CALIBRATION_N = 30`, `MIN_YHAT_STD = 0.05`, `SPEARMAN_GO = 0.35` / `STOP = 0.15`, `+15pp` retention lift, 30% exploration share. Numerical thresholds with no criticality demonstration are classic routine-optimization territory (*In re Aller*). Worse, 30 is the most conventional number in applied statistics — the central-limit rule of thumb. A claim reciting "at least 30" invites an examiner to cite a first-year textbook.

### The §101 problem, which dominates everything

*Recentive Analytics, Inc. v. Fox Corp.*, No. 2023-2437 (Fed. Cir. Apr. 18, 2025) held that **"patents that do no more than claim the application of generic machine learning to new data environments, without disclosing improvements to the machine learning models to be applied, are patent ineligible,"** and that an abstract idea "does not become nonabstract by limiting the invention to a particular field of use or technological environment." Speed and efficiency gains from applying ML do not confer eligibility.

That holding maps onto this mechanism almost line for line. The pipeline applies *established* statistical techniques — every one of them off-the-shelf — to a *new data environment* (LLM-simulated dyadic compatibility scores vs. dinner outcomes). It discloses no improvement to isotonic regression, to cross-validation, to the bootstrap, or to mixed models. Under Alice step one the claims are directed to mathematical concepts and mental processes of statistical evaluation; under step two, PAVA, cluster resampling and REML on generic computers are the definition of well-understood, routine and conventional activity. The August 2025 USPTO memorandum (§101 rejections require affirmative evidence of ineligibility, more-likely-than-not standard) raises the examiner's burden at prosecution, but it does not change the Federal Circuit's law, and it does not help against an invalidity challenge from a defendant with money.

---

## Patentability Verdict

**Not patentable, and not worth filing.** Concretely:

| Axis | Assessment |
|---|---|
| §101 eligibility | **Fatal.** *Recentive* (Fed. Cir. 2025) directly on point. The claims are mathematical concepts applied to a new data environment with no improvement to the underlying methods. |
| §102 novelty (per component) | **Fatal.** Zadrozny & Elkan 2002; Park & Marcotte 2012; Liang & Zeger 1986; Kenny 1984; Li et al. 2011. Every stage is anticipated. |
| §103 obviousness (the combination) | **Fatal.** Known elements, known functions, predictable results (KSR). Every practitioner in causal inference or psychometrics who was handed this problem would assemble a variant of this pipeline. Reviewer 2 would *demand* it. |
| Thresholds (n=30, ρ≥0.35, 30% arm) | **Routine optimization** (*In re Aller*); n=30 is the CLT rule of thumb; minimum-denominator suppression is institutionalized public-health practice. |
| Detectability / enforceability | **Poor.** This runs entirely inside the company. A competitor's identical pipeline is invisible unless they publish their methodology — and if they publish, they've earned the credibility anyway. Unenforceable claims have negative expected value. |
| Cost/benefit | **Negative.** Roughly $15–30k through issuance for a claim that is unenforceable when granted and likely dies under §101 in litigation or PTAB review. |

The honest summary: **this is rigorous applied statistics, executed correctly. Doing textbook methodology correctly is admirable, rare, and commercially decisive — and it is categorically not an invention.** Patent law rewards technical novelty, not intellectual honesty. These are different things, and confusing them wastes counsel budget.

---

## Draft Claims (if any)

I do not recommend filing. The sketches below exist only so the decision to skip is an informed one, and so counsel can see how thin the strongest available claims are.

### Sketch A — dyadic double-sided fold assignment with straddler exclusion (the strongest available, still weak)

> A computer-implemented method for evaluating a dyadic compatibility model, comprising:
> (a) storing a plurality of directed dyad records, each record identifying a first participant identifier, a second participant identifier, a group identifier, an assembly-method label, a model-generated compatibility score, and an observed post-interaction outcome;
> (b) assigning each distinct participant identifier to exactly one of k folds by a seeded balanced permutation;
> (c) for each fold, assigning a dyad record to a test partition only when **both** its participant identifiers are assigned to that fold, to a training partition only when **neither** is, and to a discard partition otherwise;
> (d) computing and persisting a discard fraction for each fold and emitting it with the evaluation result;
> (e) asserting, and failing the evaluation when violated, that no participant identifier appears in both the training and test partitions of the same fold; and
> (f) withholding the evaluation result from a downstream reporting interface when the assertion of (e) fails.

**Why it is still weak:** step (c) is Park & Marcotte's C3/C1 partition (Nature Methods, 2012). Steps (d)–(f) are reporting and assertion logic. A §101 rejection under *Recentive* is near-certain, and Park & Marcotte supplies §102/§103 art for the only substantive limitation.

### Sketch B — sample-floor-gated calibration coupled to an exploration-arm-restricted reporting permission (the "system, not a checklist" framing)

> A system for controlling publication of a simulation-to-reality correlation statistic, comprising one or more processors configured to:
> (a) generate, for each of a plurality of participant pairs, a compatibility score by executing a language-model simulation over synthetic participant representations, without reference to any observed interaction between the pair;
> (b) assemble a first subset of participant groups by a model-driven assembly process and a second subset by a randomized assembly process, and label each resulting dyad record with the assembly process used;
> (c) refuse to fit a monotone calibration map from compatibility score to observed outcome probability unless the count of labeled dyads meets a minimum-sample threshold, and, when the threshold is met, fit the monotone map and store calibrated probabilities;
> (d) compute a rank correlation between compatibility score and observed outcome **restricted to dyad records labeled with the randomized assembly process**, together with a standard error obtained by resampling whole group identifiers with replacement and a design-effect ratio of that standard error to an i.i.d. resampling standard error;
> (e) fit a linear mixed model of the observed outcome on the compatibility score with crossed random intercepts for rater identifier, ratee identifier and group identifier, and derive a variance share for each; and
> (f) emit the rank correlation through a reporting interface only when the threshold of (c) was met, the correlation of (d) exceeds a configured go-threshold, and the mixed model of (e) attributes a non-zero significant share to the dyad-specific residual component;
> otherwise emitting a withheld-result indicator.

**Why it is still weak:** (a)–(e) are each anticipated; (f) is a conditional-display rule, which is the weakest species of limitation post-*Alice*. Accenture's US10990901B2 already claims a validation platform gating deployment on sufficient observations for statistical significance. Expect a §101 rejection, a §103 rejection over Li et al. 2011 + Zadrozny & Elkan 2002 + Kenny 1984, and no allowable subject matter without narrowing to something worthless.

**If, against this advice, a filing happens:** file a provisional only, keep it cheap, and use it exclusively as a dated public-facing marker ("patent pending") for fundraising optics. Do not spend on a non-provisional. Do not budget for prosecution past the first office action.

---

## Design-Around Notes

Written as if we held the claims above — this is how trivially a competitor escapes, which is itself the argument against filing.

**Around Sketch A:**
- Keep straddling dyads in training instead of discarding them (weaker but defensible), or assign folds by *group* rather than by person and accept the residual leakage.
- Use nested CV with leave-one-person-out on the outer loop — no "k folds by seeded permutation," no discard partition.
- Weight straddlers down continuously instead of discarding them, producing no discard fraction to report.
- Simply don't emit `dropped_share` (limitation (d) evaporates).

**Around Sketch B:**
- Use Platt scaling, beta calibration, or Venn-Abers instead of isotonic — no monotone-step calibrator, and no sample floor needed. Snowflake already demonstrated this exact design-around in US12032919B1 by choosing polynomial regression.
- Set the floor at 25 or 50, or make it adaptive (e.g. minimum bin occupancy), rather than a fixed "minimum-sample threshold."
- Replace the cluster bootstrap with Liang–Zeger sandwich SEs, a wild cluster bootstrap, or a jackknife — all standard, none "resampling whole group identifiers with replacement."
- Replace the exploration arm with inverse-propensity weighting, doubly-robust off-policy estimation, or a regression-discontinuity on the assembly score. These estimate the same quantity without a randomized subset at all, and the recsys debiasing literature is full of them.
- Report Kendall's τ or an AUC instead of Spearman ρ.
- Replace the crossed mixed model with a full Bayesian SRM in Stan, or with `TripleR` variance partitioning — same decomposition, different machinery.
- Publish the correlation unconditionally and put the caveats in a footnote. Limitation (f) — the conditional-emission rule — is a policy choice, and a competitor can decline to adopt it at zero technical cost.

**The load-bearing observation:** every one of these design-arounds is *equally good statistics*. There is no narrow corridor where our specific choices are uniquely correct. When the design-around space is this large and this cheap, the claim protects nothing, and that is the clearest signal that the value here is not in exclusion.

---

## Is This a Patent or a Moat?

**It is a moat. It is not a patent. And that is a genuinely good outcome, not a consolation prize.**

The mechanism's value is real and it is large, but it is realized through four channels that patents do not touch:

**1. It is a credibility moat, and credibility is the actual scarce good in this category.** The dating and social-matching category is littered with companies that published a correlation coefficient computed on model-selected pairs with i.i.d. standard errors. That number is fake in a specific, technically demonstrable way, and the exploration arm is what makes ours not fake. When a technically competent diligence partner asks "what's your selection bias correction?" and the answer is "30% of every week's groups have been randomly assembled since week one, here is the arm-restricted Spearman with cluster-bootstrap SEs and the design effect," the conversation ends differently than it does for every competitor. §6 of the architecture doc says this outright: "a good investor's technical diligence partner will catch it." A patent does not make that conversation go better. Having run the pipeline does.

**2. It is an execution moat, because the barrier is discipline over time, not knowledge.** Nothing here is secret. Every method is on Wikipedia and in scikit-learn. What is scarce is a founder who gives up 30% of assembly quality *permanently, from week one*, before there is any model worth protecting — and who then refuses to quote the flattering number computed on the other 70%. That is a cost paid continuously in product quality and growth for an epistemic benefit realized much later. Most teams will not pay it, and a competitor who reads this document still has to go back in time to start their exploration arm. The moat is the accumulated randomized data, not the statistics applied to it. **That is worth saying twice: the defensible asset is the exploration-arm dataset, which compounds and cannot be retroactively created. The pipeline is just how it gets read.**

**3. It is a product-truth moat.** Stage 4 answers a question most competitors never ask: are we predicting dyadic fit, or are we detecting charisma? Joel, Eastwick & Finkel (2017) found actor and partner variance modestly predictable and relationship variance not predictable at all — the hardest published result in this domain, and a direct threat to the premise. If our relationship-variance share is significantly positive, we have something genuinely new and a defensible claim against the strongest prior scientific finding in the field. If it isn't, we learn we've built a different product before we build the wrong company around it. Both outcomes are worth more than a patent.

**4. Publishing beats patenting here, on every axis.** A methods write-up — exploration arm design, dyadic split with quantified straddler loss, arm-restricted correlation with design effects, SRM decomposition of a sim-driven matcher — would be a strong applied-stats blog post or workshop paper. It would (a) establish defensive prior art at zero cost, foreclosing anyone else's attempt to claim this against us; (b) function as recruiting and credibility collateral precisely because so few companies in this space can write it; (c) cost roughly nothing versus $15–30k and months of attorney attention; and (d) be read by exactly the people whose opinion determines whether the ρ number is believed. A granted patent, by contrast, would be unenforceable (we cannot see inside a competitor's eval harness), narrow (see Design-Around Notes), and likely invalid under *Recentive*.

**Recommendation:**

- **Do not file.** Not a non-provisional, and a provisional only if a specific investor conversation makes "patent pending" worth a few thousand dollars of theater. Be explicit internally that it would be theater.
- **Do publish**, as a defensive-prior-art and credibility play, once the week-6 gate has run on real data and there are actual numbers to show.
- **Do treat the exploration-arm dataset as the protected asset.** It is the thing that compounds, the thing a competitor cannot backfill, and the thing that should be governed as confidential business information and covered by trade-secret hygiene — access controls, NDAs, no external sharing of raw dyad-level data.
- **Do keep the thresholds in code and in the architecture doc, as we already do.** `MIN_CALIBRATION_N = 30` raising rather than warning, `SPEARMAN_GO = 0.35` on the exploration arm only, verdict = worst metric. The commit history showing these constants were set *before* the data arrived is more persuasive to a serious diligence partner than any patent number would be, because it demonstrates the one thing nobody can fake retroactively: that we set the bar before we knew whether we would clear it.
- **Direct patent budget elsewhere.** If anything in this system is worth filing on, it is a specific technical mechanism in the simulation harness, persona construction, or group-assembly optimization — not the evaluation statistics. This assessment should close that question for the eval pipeline permanently.

---

## Sources

- [US12032919B1 — Post-calibration of large language model confidence scoring (Snowflake)](https://patents.google.com/patent/US12032919B1/en)
- [US20160341649A1 / WO2015095667A1 — Particle score calibration](https://patents.google.com/patent/US20160341649)
- [US11755947B1 — Offline evaluation of machine learning models with noise reduction (Amazon)](https://patents.google.com/patent/US11755947B1/en)
- [US8769094B2 — Managing sample selection bias](https://patents.google.com/patent/US8769094)
- [US20190147371A1 / US10990901B2 — Training, validating, and monitoring AI and ML models (Accenture)](https://patents.google.com/patent/US20190147371A1/en)
- [US11263188B2 — Generation and management of AI model documentation throughout its lifecycle (IBM)](https://patents.google.com/patent/US11263188B2/en)
- [US11544411B2 — Machine learning model validation and authentication](https://patents.google.com/patent/US11544411B2/en)
- [US8170841B2 — Predictive model validation](https://patents.google.com/patent/US8170841)
- [US8782038B2 — Systems and methods for online compatibility matching and ranking (eHarmony)](https://patents.google.com/patent/US8782038?oq=eharmony.com)
- [US20110138305A1 — Random matching and real-time compatibility assessment](https://patents.google.com/patent/US20110138305)
- [Park & Marcotte, Flaws in evaluation schemes for pair-input computational predictions, Nature Methods (2012)](https://www.nature.com/articles/nmeth.2259)
- [Joel, Eastwick & Finkel, Is Romantic Desire Predictable?, Psychological Science (2017)](https://journals.sagepub.com/doi/abs/10.1177/0956797617714580)
- [Li et al., Unbiased offline evaluation of contextual-bandit-based news recommendation (arXiv:1003.5956)](https://arxiv.org/pdf/1003.5956)
- [An Unbiased Data Collection and Content Exploitation/Exploration Strategy for Personalization (arXiv:1604.03506)](https://arxiv.org/pdf/1604.03506)
- [Silva Filho et al., Classifier calibration: a survey (arXiv:2112.10327)](https://arxiv.org/pdf/2112.10327)
- [Cluster-Robust Bootstrap Inference in Quantile Regression Models (arXiv:1407.7166)](https://arxiv.org/pdf/1407.7166)
- [Genuinely Robust Inference for Clustered Data (arXiv:2308.10138)](https://arxiv.org/pdf/2308.10138)
- [David A. Kenny — Social Relations Model reference](https://davidakenny.net/srm/soremo.htm)
- [TripleR: An R package for social relations analyses based on round-robin designs](https://link.springer.com/article/10.3758/s13428-011-0150-4)
- [Kluger et al., The Social-Relations Model for Asymmetric-Block-Design Data, AMPPS (2025)](https://journals.sagepub.com/doi/full/10.1177/25152459241279522)
- [SimUSER: Simulating User Behavior with LLMs for Recommender System Evaluation (arXiv:2504.12722)](https://arxiv.org/pdf/2504.12722)
- [Evaluating LLMs as Generative User Simulators for Conversational Recommendation (arXiv:2403.09738)](https://arxiv.org/pdf/2403.09738)
- [Sterne Kessler — Federal Circuit's First Alice Analysis for Machine Learning Patents (Recentive v. Fox)](https://www.sternekessler.com/news-insights/client-alerts/federal-circuits-first-alice-analysis-for-machine-learning-patents/)
- [Baker Botts — Federal Circuit Refines Section 101 Eligibility of Machine Learning Inventions](https://www.bakerbotts.com/thought-leadership/publications/2025/may/federal-circuit-refines-section-101-eligibility-of-machine-learning-inventions)
- [O'Melveny — Federal Circuit Rules on §101 Eligibility of AI Machine Learning Patents](https://www.omm.com/insights/alerts-publications/client-alert-federal-circuit-rules-on-101-eligibility-of-ai-machine-learning-patents/)
- [Participant-Aware Model Validation for Repeated-Measures Data, JMIR AI (2026)](https://ai.jmir.org/2026/1/e87728)
- [Washington State DOH — Standards for Reporting Data with Small Numbers](https://www.doh.wa.gov/portals/1/documents/1500/smallnumbers.pdf)
- [Oregon Health Authority — Small Numbers Reporting Guidelines](https://www.oregon.gov/oha/HPA/ANALYTICS/Documents/HASmallNumberReportingGuidelines_External.pdf)

*Prepared as internal prior-art analysis. Not legal advice; a registered patent attorney should confirm before any filing decision is acted on — though this assessment recommends no filing.*

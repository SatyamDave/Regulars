# 04 — SRM Judge Diagnostic: Patentability Assessment

**Subject:** Decomposing a trained compatibility model's own predictions with Kenny's Social
Relations Model and comparing component-wise against the decomposition of human ratings, as a
build gate and product-claims gate.

**Implementation under review:** `regulars/judge/srm.py` (`fit_srm`, `group_effects`,
`variance_explained_by`), `regulars/judge/diagnostics.py`, `docs/technical-architecture.md`
§5 "v2 (from ~1,000 labeled directed dyads): Social Relations Model", §6 "The evaluation design",
§11 "Build order" row 6.

**Date of search:** 2026-09-14. **Analyst:** prior-art review, not legal advice. Nothing here is a
freedom-to-operate opinion or a registrability opinion from counsel.

**Bottom line:** Weak. Do not file a standalone application on the diagnostic as described.

---

## 1. Mechanism Summary

### What the code actually does

A Regulars dinner is a complete round-robin: `n` people, every person rates every other person,
`n(n-1)` directed ratings. `srm.py` fits the Social Relations Model to that matrix:

```
X_ij = mu + a_i + b_j + g_ij
```

- `a_i` — **actor** effect: how positively `i` rates others
- `b_j` — **partner** effect: how positively `j` is rated by others
- `g_ij` — **relationship** effect: the dyad-specific residual, the thing a matching engine
  would have to predict to be a matching engine

The estimators are the closed-form least-squares effects of Warner, Kenny & Stoto (1979) plus the
unbiased variance-component formulas of Kenny (1994) for a single unreplicated round-robin, pooled
across groups with weight equal to group size. `srm.py`'s own module docstring states that these are
"the same formulas the R package `TripleR` (Schoenbrodt, Back & Schmukle) implements in `RR()`."
Groups require `n >= 4`; with `n = 3` the symmetric relationship term has zero degrees of freedom.

### The step that is claimed to be novel

`variance_explained_by(y_hat_by_dyad, ratings)` does something the textbook use does not. The judge
emits a directed score `y_hat` for *every* ordered pair in the group, so `y_hat` is itself a complete
round-robin matrix. The function:

1. Decomposes the **human ratings** into `(a_i, b_j, g_ij)` per group.
2. Decomposes the **model's own predictions** with the identical estimator into
   `(â_i, b̂_j, ĝ_ij)`.
3. Computes r² between like components — actor vs. actor, partner vs. partner, relationship vs.
   relationship — pooled across all groups.

The read-out, quoted from the docstring:

- high `partner` r² only → the judge finds likeable people (a charisma detector)
- high `actor` r² only → the judge finds generous raters (rater bias)
- `relationship` r² > 0 → the judge predicts dyad-specific fit (a matching engine)

### What the result gates

Per §11 of the architecture doc, this is build-order step 6, and per §6 "The success bar," the
target is "SRM relationship-variance explained > 0 and significant." It is asserted as a gate on two
things simultaneously:

- **Engineering go/no-go** — whether the learned judge is promoted over judge v1.
- **Product claims** — whether the company may describe the system as a matching engine at all,
  as opposed to a popularity/charisma ranker, "before you pitch it."

Supporting constraints that make the diagnostic trustworthy and that are part of any credible claim
set: labels are drawn from a permanent ~30% randomized-assembly exploration arm (§6 Problem 1) so
the model does not grade its own pair selection; splits are by person, not by pair (§6 Problem 2),
precisely because a pair-split leaks the partner effect; significance is reported with clustered
standard errors (§6 Problem 3).

---

## 2. Prior Art Found

Organized by how much damage each tier does.

### Tier A — The statistical baseline. Public domain; must be designed around, cannot be claimed.

| Reference | What it establishes |
|---|---|
| Warner, R. M., Kenny, D. A., & Stoto, M. (1979). A new round-robin analysis of variance for social interaction data. *JPSP* 37, 1742–1757. | The exact closed-form least-squares actor/partner/relationship estimators implemented in `_decompose()`. |
| Kenny, D. A. (1994). *Interpersonal Perception: A Social Relations Analysis*, ch. 9. | The unbiased variance-component formulas including the MS_sym / MS_asym correction terms, verbatim the ones in `srm.py`. |
| Kenny, D. A., Kashy, D. A., & Cook, W. L. (2006). *Dyadic Data Analysis*, ch. 8. | Canonical textbook treatment; also the APIM. |
| Bond & Lashley; Nestler (ML/MCMC SRM estimation); Nestler, "Round-robin analysis of social interaction: exact and estimated standard errors," *Psychometrika*. | Standard-error machinery for exactly these estimates. |
| Schönbrodt, F. D., Back, M. D., & Schmukle, S. C. (2012). TripleR: An R package for social relations analyses based on round-robin designs. *Behavior Research Methods*. CRAN v1.5.5, GPL-2 \| GPL-3. | A published, freely distributed implementation. Public use and printed publication. |
| Kenny's SOREMO software; Lüdtke et al., social relations growth model (*Psychometrika* 2016); dynamic SRM for longitudinal dyadic data (*JRSS-A* 2023). | Continuous, active development of the method for decades. |

Anything claimed at the level of "decompose round-robin ratings into actor, partner and relationship
variance" is anticipated under §102(a)(1) by printed publication and by public use of TripleR. This
is not arguable.

### Tier B — The killer reference: SRM already combined with ML, on this exact question, in this exact field.

**Joel, S., Eastwick, P. W., & Finkel, E. J. (2017). "Is Romantic Desire Predictable? Machine
Learning Applied to Initial Romantic Attraction." *Psychological Science* 28(10), 1478–1489.**

Two speed-dating round-robins; >100 self-report measures; SRM decomposition of attraction ratings
into actor, partner and relationship variance; machine learning (random forests / ensemble methods)
used to predict each component; component-wise predictive accuracy reported. Finding: actor and
partner variance were predictable, relationship variance was **not** predictable above chance.

This reference is fatal-adjacent for four reasons:

1. It is in the **identical field of endeavor** — predicting interpersonal attraction from data.
2. It states the **identical motivation** — determine which SRM component a machine-learned model
   can actually predict.
3. It produces the **identical output structure** — a per-component predictive-accuracy number for
   actor, partner and relationship.
4. `docs/technical-architecture.md` §5 **cites it as the model for the design**: "This mirrors the
   Joel/Eastwick/Finkel decomposition exactly, where actor and partner were predictable and
   relationship was not." That sentence is an admission in the specification's own lineage.

The one real methodological difference: Joel et al. **decompose-then-predict** (SRM the ratings
first, then fit a separate learner per component), whereas `variance_explained_by` does
**predict-then-decompose** (take one model's directed output matrix, SRM it with the same
estimator, correlate component-wise). See §3 for whether that difference carries any weight. It is
the only thing standing between this mechanism and outright anticipation.

### Tier C — The same diagnostic, independently invented in other dyadic-ML domains (analogous art).

This is what makes a KSR combination argument easy for an examiner: the idea is not confined to
psychology.

- **Drug–target affinity (DTA) evaluation, Pahikkala and colleagues.** The DTA literature routinely
  decomposes an affinity value additively into *grand mean + drug main effect + target main effect +
  interaction effect*, and defines the interaction effect as the departure of the affinity from the
  sum of the main effects — structurally identical to `mu + a_i + b_j + g_ij`.
  - Pahikkala et al., "Toward more realistic drug–target interaction predictions," *Briefings in
    Bioinformatics* 16(2):325 (2015) — evaluation bias and leakage settings, including the
    now-standard warning that a DTA model can score well while having learned only main effects.
  - "Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods,"
    arXiv:2510.14419 (2025) — an evaluation metric constructed specifically so that a dyadic
    prediction method is scored on the interaction component rather than on main effects. This is
    the Regulars diagnostic's exact purpose, expressed as a metric, in bioinformatics.
- **Recommender systems.** The crossed random-effects form `y_ij = β0 + a_i + b_j + ε_ij` (user
  effect, item effect, residual interaction) is the standard baseline predictor and is the subject
  of, e.g., "Scalable recommender system based on factor analysis," arXiv:2408.05896. The partner
  effect in SRM is the item/popularity bias in recommenders under a different name.
- **Popularity-bias detection.** An entire subfield exists around diagnosing "the model is
  predicting item popularity, not personalization" — including granted patent **US 12,051,099**
  (system/method for handling popularity bias in item recommendations) and multi-level
  recommendation-concentration diagnostics in the e-commerce literature. Substitute "partner effect"
  for "popularity" and "relationship effect" for "personalization" and the Regulars framing is the
  received wisdom of that field.

### Tier D — Domain patents in matching/compatibility. None perform this diagnostic.

| Patent | Assignee / subject | Relevance |
|---|---|---|
| US 6,735,568 B1 | eHarmony — identifying people likely to have a successful relationship; PCA over 29 attributes | Compatibility scoring. No decomposition of model output; no actor/partner/relationship framing. |
| US 8,782,038 B2 | Online compatibility matching and ranking | Scoring and ranking only. |
| US 8,775,324 B2 | Facebook/Meta (Zhu, Lunt) — compatibility scoring of users in a social network | Scoring only. |
| US 8,595,257 B1 | Identifying romantically compatible subjects | Scoring only. |
| US 2011/0138305 A1 | Random matching + real-time compatibility assessment | Interesting for the randomized-arm limb, but it randomizes to create serendipity, not to produce unbiased evaluation labels. |
| US 10,089,074 / US 11,275,557 / WO 2018/009890 A1 | Relationship analysis utilizing biofeedback information | **Closest patent that uses "variance" in a compatibility context.** It compares *ratios of variance in sensor data between two users* and compares their linear regressions to produce a compatibility score. That is variance of a raw signal used as a feature for scoring a pair — not variance-component decomposition of a model's output used to diagnose the model. Clean distinction, and it should be cited to establish that the field's use of "variance" is unrelated. |
| US 2019/0026681 A1; US 11,000,218; US 10,902,384 B2; US 2022/0092514 A1 | HR-tech: data-driven talent identification, behavioral-insights team-composition dashboards with "compatibility analysis," candidate assessment | Team composition and pair compatibility from psychometrics. None decompose a model's predictions into variance components or gate on which component was learned. |

No patent found in this tier performs SRM, or any variance-component decomposition, on model output.

### Tier E — ML pipeline gating patents. These make the "mandatory checkpoint" limb conventional.

- **EP 4,577,958 A1** — automated ML pipeline deployment: retrieve model definition from registry,
  validate against test exemplars, instantiate inference pipeline.
- **US 12,367,418** — ML model verification for assessment pipeline deployment.
- **US 12,242,469** — training/testing data generation, reciting a TFX-style component list
  (ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, **Evaluator**,
  **InfraValidator**, Pusher). The Evaluator → Pusher hand-off *is* a metric-threshold deployment
  gate, granted and in the art.
- **US 12,423,615** — machine learning pipelines.
- **US 10,452,992 B2** — interactive interfaces for machine learning model evaluations.

Conditionally blocking a model's promotion on a computed metric crossing a threshold is
well-trodden. It will not supply the inventive concept at Alice step two, and it will not supply
non-obviousness at §103.

### Tier F — Controlling law on the §101 question.

**Recentive Analytics, Inc. v. Fox Corp., 134 F.4th 1205 (Fed. Cir. Apr. 18, 2025)** (Dyk, Prost,
Goldberg). Four ML patents held ineligible. Holdings that bear directly on this mechanism:

- Claims that "do no more than apply established methods of machine learning to a new data
  environment" are not patent eligible.
- The argument that the claims were eligible because they applied a known technique to a *new field
  of use* was expressly rejected.
- "An abstract idea does not become nonabstract by limiting the environment."
- Performing with a computer, faster, a task previously done by humans does not confer eligibility.

Our mechanism is, if anything, more exposed than Recentive's: it applies an established
*statistical* method (not even machine learning, which at least involves a trained artifact) to a
new data environment (model scores rather than human ratings), and outputs a number that a human
reads.

---

## 3. Novelty Analysis

### What genuinely was not found in any single reference

1. **Predict-then-decompose on an unmodifiable black-box scorer.** No reference applies SRM
   variance-component estimators to the *output matrix of an already-trained dyadic model* and
   correlates component-wise against the same decomposition of ground truth. Joel et al. decompose
   first and then fit one learner per component, which requires you to own and retrain the model
   three times. The Regulars diagnostic audits an arbitrary scorer post hoc — including a scorer
   whose `y_hat` is produced by LLM role-play simulation rather than by regression, where "train a
   separate model for the relationship component" is not even a well-defined operation. That is a
   real capability difference, not just a reordering, for the specific class of non-retrainable
   generative scorers.
2. **Requiring the model to emit a complete directed round-robin.** The diagnostic is only
   identified if `y_hat` exists for every ordered pair in the group, including pairs the assembly
   optimizer would never have proposed. That is a non-trivial constraint on how the scoring system
   must be operated, and it is what makes the SRM identifiable on the prediction side. No reference
   found imposes it.
3. **The composite pipeline.** Randomized exploration arm → complete round-robin ratings →
   person-level (not pair-level) splits → SRM on both `y` and `ŷ` → component-wise r² with clustered
   significance → promotion gate → permissible-marketing-claim gate. As a whole combination, not
   found.
4. **The product-claims gate.** Conditioning what the product may be *represented as* on which
   variance component the model demonstrably predicts. Not found anywhere. Also, almost certainly
   not technical subject matter.

### Why each of those is weaker than it looks

1. **Predict-then-decompose vs. decompose-then-predict is a rearrangement an examiner will reject.**
   Both the estimator and the comparison are linear and unbiased; on complete round-robin data the
   two orderings carry near-equivalent information about which component was learned. MPEP 2144.04
   (rearrangement of parts / changes in sequence) and KSR rationale D ("applying a known technique
   to a known device ready for improvement") both fit. The non-retrainable-scorer argument is the
   only version of this that survives, and it needs to be *in the claim* — the claim must recite
   that the scorer is treated as a black box and is not retrained — or it buys nothing.
2. **Two independent fields already teach it.** Joel et al. (psychology) and the Pahikkala DTA
   line (bioinformatics) independently arrived at "check whether the dyadic predictor captured the
   interaction term or only the marginals." Two independent teachings is the textbook fact pattern
   for obviousness. Add the recommender popularity-bias literature and it is three.
3. **New data type applied to old statistics is presumptively obvious.** MPEP 2144.07; and the
   motivation to combine is supplied explicitly by the references themselves — Joel et al. state the
   motivation in their abstract.
4. **The specification would contain admissions against interest.** `srm.py`'s docstring says the
   estimators are "the same formulas the R package `TripleR` … implements in `RR()`." Whatever goes
   into the spec, TripleR and Warner/Kenny/Stoto and Joel et al. must be disclosed in an IDS under
   37 C.F.R. §1.56. An examiner will quote that docstring sentence back at us.
5. **§101 is the harder wall than §103, and it comes first.**
   - *Alice step one.* The claim as it stands is directed to (i) a mathematical concept — a
     variance-component decomposition and a correlation coefficient — and (ii) a mental process —
     comparing a number to a threshold and drawing a conclusion about a model. Both are enumerated
     abstract-idea categories in the 2019 PEG / MPEP 2106.04(a)(2).
   - *Alice step two.* The remaining elements are "obtain ratings," "obtain model scores," "compute
     r²," "display/gate." Data gathering is insignificant extra-solution activity; threshold gating
     is conventional (Tier E). No inventive concept.
   - *Recentive* forecloses the most natural rebuttal — "but nobody had applied SRM to model
     outputs before" — because novelty of the *data environment* was the exact argument the Federal
     Circuit rejected.
   - The closest helpful analogy would be a *McRO*-style argument (specific rules producing a
     concrete technical improvement) or *Finjan*/*Enfish* (improvement to the functioning of the
     system itself). That argument only becomes available if the diagnostic's output **changes how a
     downstream technical system operates**, not merely what a human believes about the model. See
     Design-Around Notes.
   - EPO: under G 1/19 and T 0641/00, a statistical quality metric for a model is a non-technical
     aim. The only EPO-viable framing is a technical effect such as controlling compute allocation
     (per §8 of the architecture doc: whether to spend frontier-model simulation budget on a given
     candidate set).

---

## 4. Patentability Verdict

**Weak — do not file standalone. The diagnostic as described is very likely ineligible under §101
after *Recentive* and very likely obvious under §103 over Joel/Eastwick/Finkel 2017 combined with
the Warner-Kenny-Stoto/Kenny SRM estimators and the drug–target interaction-effect evaluation
literature.**

Component-by-component:

| Element | Assessment |
|---|---|
| SRM estimators themselves | Anticipated. Public domain since 1979. Zero chance. |
| SRM applied to human round-robin ratings in a social product | Anticipated. Textbook use. Zero chance. |
| SRM applied to a trained model's own predictions, compared component-wise | Novel in the strict §102 sense (no single reference), but obvious under §103 over Joel 2017 + Pahikkala, and abstract under §101. Low single-digit odds as claimed. |
| The diagnostic as a mandatory deployment gate | Conventional (Tier E). Adds nothing at step two. |
| The product-claims / marketing gate | Novel and unfound, but non-technical; likely rejected as an abstract business rule, and unenforceable in practice. |
| Closed loop: diagnostic output alters the assembly optimizer's objective or the simulation compute budget | **The only limb with a plausible path.** Converts "compute and report a statistic" into "compute a statistic and thereby change the technical operation of a downstream system." Still hard; meaningfully better than the rest. |

**Recommended action, in priority order:**

1. **Defensive publication.** Publish the `variance_explained_by` method and the component-wise
   gate — a technical blog post, an arXiv preprint, or an IDS-quality write-up with a date stamp.
   This costs nothing, prevents a competitor from later claiming it against us, and the diagnostic's
   commercial value to Regulars is as an internal truth-detector, which publication does not
   diminish. It also happens to be a credibility asset in front of a technical diligence partner,
   which is the audience §6 of the architecture doc is written for.
2. **Trade secret on the thresholds and the calibrated gate.** The method is publishable; the
   specific relationship-r² threshold, the clustered-SE significance procedure, the exploration-arm
   fraction, and the observed component shares on real Regulars data are not. Keep those out of any
   public artifact.
3. **If we file at all**, fold this in as dependent claims under a broader independent claim on the
   closed-loop assembly system (Draft Claim 4 below), never as the independent claim. A standalone
   application here spends filing and prosecution money to buy a rejection.

---

## 5. Draft Claims

Drafted to be as defensible as the subject matter allows. Claim 4 is the one to build around; claims
1–3 and 5 are written on the assumption they become dependents of it. Every claim is drafted to
recite a state change in a machine, not a number shown to a person.

---

**Claim 1 (method — the core diagnostic; weakest as an independent, include as fallback).**

A computer-implemented method for diagnosing whether a trained dyadic compatibility-prediction model
has learned dyad-specific effects or individual-level effects, comprising:

- (a) receiving, for each of a plurality of assembled groups each having at least four members, a
  set of directed interpersonal ratings forming a complete round-robin rating matrix in which each
  member has rated each other member;
- (b) obtaining, from a trained compatibility-prediction model, a predicted compatibility score for
  every ordered pair of members of each said group, including ordered pairs for which no group
  assembly decision was made, thereby forming a complete round-robin prediction matrix having the
  same index set as the rating matrix, wherein the trained model is treated as a black box and is
  not retrained as part of the method;
- (c) applying a round-robin variance-component estimator to the rating matrix to obtain, for each
  group, an actor effect per member, a partner effect per member, and a relationship effect per
  ordered pair;
- (d) applying the **same** round-robin variance-component estimator to the prediction matrix to
  obtain a predicted actor effect per member, a predicted partner effect per member, and a predicted
  relationship effect per ordered pair;
- (e) computing a component-wise agreement statistic between the rating-derived and
  prediction-derived effects separately for the actor component, the partner component, and the
  relationship component, pooled over the plurality of groups;
- (f) assigning the trained model to one of at least three diagnostic classes — a dyad-specific
  class, an individual-level-desirability class, and a rater-bias class — according to which
  component-wise agreement statistic exceeds a threshold; and
- (g) writing the assigned diagnostic class and the component-wise agreement statistics to a model
  registry record associated with a version identifier of the trained model.

**Claim 2 (dependent — the unbiased-label limb; this one is substantively load-bearing).**

The method of claim 1, wherein the plurality of assembled groups consists of groups assembled by a
randomized assembly procedure that is independent of the trained compatibility-prediction model,
such that the component-wise agreement statistics are not contaminated by the model's own pair
selection; and wherein the component-wise agreement statistics are accompanied by a significance
measure computed with standard errors clustered by rater, by ratee, and by group.

**Claim 3 (dependent — the unreplicated-round-robin correction; the most concretely technical limb).**

The method of claim 1, wherein applying the round-robin variance-component estimator comprises, for
each group of size n:

- computing symmetric and asymmetric mean squares of the relationship effects over the n(n−1)/2
  unordered pairs;
- estimating the relationship variance as one half the sum of said mean squares and a
  relationship-reciprocity covariance as one half their difference;
- estimating the actor variance and the partner variance as the mean squared effect less correction
  terms derived from said symmetric and asymmetric mean squares, such that the variance estimates
  are unbiased rather than merely consistent;
- rejecting any group of size less than four as under-identified; and
- pooling the per-group variance estimates across groups by a weighted mean with weight equal to
  group size;

and wherein the estimated relationship variance is recorded as a combined relationship-plus-error
quantity, the round-robin being unreplicated.

**Claim 4 (independent — closed loop; this is the claim to actually pursue).**

A group-assembly system, comprising one or more processors and a memory storing instructions that,
when executed, cause the system to:

- generate, for each ordered pair of candidate participants within a candidate set, a directed
  compatibility score by executing a language-model simulation of an interaction between persona
  representations of the two participants, thereby forming a complete directed score matrix over the
  candidate set;
- partition the candidate set into groups by optimizing an objective function over the directed
  score matrix;
- convene the groups and collect, for each convened group, a complete round-robin of directed
  post-interaction ratings;
- decompose both the collected ratings and the previously generated directed scores for the same
  convened groups into actor, partner, and relationship components using a common round-robin
  variance-component estimator, and compute a component-wise agreement statistic for each component;
  and
- **responsive to the relationship-component agreement statistic failing a threshold condition,
  automatically modify operation of the system by at least one of:** (i) substituting, in the
  objective function used for subsequent partitioning, a partner-component-derived term for a
  dyad-score term; (ii) increasing the fraction of subsequent groups assembled by a randomized
  assembly procedure; and (iii) reducing the per-pair simulation compute budget allocated to
  generating the directed compatibility scores, by reducing the number of simulation scenarios or
  seeds executed per ordered pair or by routing said pairs to a lower-cost language model;

and responsive to the relationship-component agreement statistic satisfying the threshold condition,
continuing to partition subsequent candidate sets using the dyad-score objective function.

**Claim 5 (dependent — the negative result as a control signal).**

The system of claim 4, wherein the instructions further cause the system to suppress a
dyad-specific-fit representation in a user-facing explanation surface and in an assembly audit
record for any group assembled while the relationship-component agreement statistic fails the
threshold condition, and to record in said audit record the identifier of the diagnostic evaluation
whose result caused the suppression.

---

**Claim-drafting notes.**

- Every independent claim must recite something that *changes* — a registry record, an objective
  function, a compute budget, a routing decision. A claim that ends in "and reporting the statistic"
  is a mathematical concept plus display and will not survive step one.
- Claim 1's limitation "treated as a black box and is not retrained" is doing specific work: it is
  the only textual hook that distinguishes the method from Joel et al.'s decompose-then-predict. Do
  not drop it in prosecution.
- Claim 4 limb (iii) is the EPO-facing limb. Compute allocation is a recognized technical effect;
  "the model is only a charisma detector" is not.
- Do **not** claim "the Social Relations Model" by name. Claim the estimator structurally, as in
  Claim 3, so the claim reads on the artifact rather than on a named academic method.
- Do not claim the marketing/product-representation gate as such. Claim 5 is the technical shadow of
  it — suppressing a representation in an explanation surface and an audit record — which is at
  least tied to system state.

---

## 6. Design-Around Notes

### What to cite (IDS obligations if we file)

Mandatory, all material under 37 C.F.R. §1.56:

- Warner, Kenny & Stoto (1979), *JPSP* 37:1742–1757.
- Kenny (1994), *Interpersonal Perception: A Social Relations Analysis*, ch. 9.
- Kenny, Kashy & Cook (2006), *Dyadic Data Analysis*, ch. 8.
- Schönbrodt, Back & Schmukle (2012), TripleR, *Behavior Research Methods*; plus the CRAN package
  itself as public use.
- **Joel, Eastwick & Finkel (2017), *Psychological Science* 28(10):1478–1489.** Non-negotiable; it is
  cited in our own architecture doc.
- Pahikkala et al. (2015), "Toward more realistic drug–target interaction predictions," *Briefings
  in Bioinformatics* 16(2):325.
- arXiv:2510.14419, "Interaction Concordance Index: Performance Evaluation for Interaction
  Prediction Methods."
- US 10,089,074 / US 11,275,557 (biofeedback relationship analysis, variance-ratio compatibility).
- US 12,051,099 (popularity bias in item recommendations).
- US 6,735,568 B1 (eHarmony); US 8,775,324 B2 (Meta).
- US 12,242,469 (ML pipeline with Evaluator/InfraValidator gating).

### What to design around, and how

| Obstacle | Design-around |
|---|---|
| Joel 2017 (decompose-then-predict) | Claim the black-box, no-retraining, complete-prediction-matrix path explicitly. Emphasize the LLM-simulation scorer, for which per-component retraining is undefined. This is a narrow lane but it is the real one. |
| Pahikkala / IC-index (interaction vs. main effects in dyadic ML evaluation) | Lean on asymmetry: DTA and recommender decompositions are on a *bipartite, undirected* affinity matrix with no actor/partner distinction and no reciprocity covariance. SRM's directed structure — the same person appears as both actor and partner, with an actor–partner covariance and a relationship-reciprocity covariance — has no counterpart there. Claim the reciprocity terms (Claim 3) as part of the estimator. |
| Tier E pipeline-gating patents | Never let the gate be the point of novelty. It is a dependent limitation at most. |
| *Recentive* §101 | Ensure the diagnostic's output drives a machine state change, not a human decision. Claim 4 is built for this. If the only thing the number does is tell a founder what to say to an investor, there is no eligible claim to be had. |
| Tier D compatibility patents | Nothing here blocks us. Their use of "variance" is variance of a raw signal used as a feature; ours is variance decomposition of a model's output used to audit the model. Cite US 10,089,074 affirmatively to draw the line. |

### Freedom to operate

Nothing found restricts Regulars from *practicing* this. The SRM mathematics is decades-old
published statistics and is not claimable by anyone. No granted patent found covers
variance-component decomposition of model output for model diagnosis. Practice freely.

### IP hygiene — one live issue

`regulars/judge/srm.py` is an independent NumPy implementation written from published formulas.
**Keep it that way.** TripleR is licensed **GPL-2 | GPL-3**. Mathematical formulas published in
*JPSP* and in Kenny (1994) carry no copyright; TripleR's *source code* does, under copyleft. Copying
or transliterating TripleR's `RR()` source into `srm.py` would raise a copyleft question over the
`regulars.judge` package and, transitively, the service that imports it. The current docstring
wording — "the same formulas the R package `TripleR` … implements" — correctly describes a formula
correspondence, not a code derivation, and that distinction should be preserved in any future edit
and in any diligence answer. Do not vendor TripleR. Do not port its code. Cite the papers, not the
package, as the source of the math.

### If we do not file

Two things are worth more than the patent would be:

1. **Publish it.** The diagnostic is a strong credibility artifact precisely because it is the test
   most companies in this category do not run. Publishing it establishes prior art against
   competitors, dates our authorship, and is the kind of thing that survives a technical diligence
   conversation better than a pending application would.
2. **Keep the results secret.** The measured component shares on real Regulars data — how much
   relationship variance the judge actually recovers — are the commercially sensitive quantity, and
   no patent would have protected them anyway.

---

## Sources

- [Warner, Kenny & Stoto (1979) / round-robin ANOVA lineage — Psychometrika](https://link.springer.com/article/10.1007/BF02294341)
- [TripleR: An R package for social relations analyses based on round-robin designs](https://link.springer.com/article/10.3758/s13428-011-0150-4)
- [TripleR on CRAN (v1.5.5, GPL-2 | GPL-3)](https://cran.r-project.org/web/packages/TripleR/index.html)
- [Kenny — SRM information and SOREMO](https://davidakenny.net/srm/soremo.htm)
- [Joel, Eastwick & Finkel (2017), Is Romantic Desire Predictable? — Psychological Science](https://journals.sagepub.com/doi/abs/10.1177/0956797617714580)
- [Joel et al. (2017) full text PDF](https://gwern.net/doc/psychology/2017-joel.pdf)
- [Association for Psychological Science — Romantic Matches Are Hard to Predict Before People Meet](https://www.psychologicalscience.org/news/releases/romantic-matches-are-hard-to-predict.html)
- [A dynamic social relations model for clustered longitudinal dyadic data — JRSS-A](https://doi.org/10.1093/jrsssa/qnad115)
- [Tackling Longitudinal Round-Robin Data: A Social Relations Growth Model — Psychometrika](https://link.springer.com/article/10.1007/s11336-016-9546-5)
- [Actor–partner interdependence model](https://en.wikipedia.org/wiki/Actor%E2%80%93partner_interdependence_model)
- [Pahikkala et al., Toward more realistic drug–target interaction predictions — Briefings in Bioinformatics](https://academic.oup.com/bib/article/16/2/325/246479)
- [Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods (arXiv:2510.14419)](https://arxiv.org/pdf/2510.14419)
- [Scalable recommender system based on factor analysis (arXiv:2408.05896)](https://arxiv.org/html/2408.05896)
- [US 12,051,099 — handling popularity bias in item recommendations](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12051099)
- [US 6,735,568 B1 — eHarmony, identifying people likely to have a successful relationship](https://patents.google.com/patent/US6735568B1/en)
- [US 8,775,324 B2 — Compatibility scoring of users in a social network](https://patents.google.com/patent/US8775324B2)
- [US 8,782,038 B2 — Systems and methods for online compatibility matching and ranking](https://patents.google.com/patent/US8782038)
- [US 8,595,257 B1 — System and method for identifying romantically compatible subjects](https://patents.google.com/patent/US8595257)
- [US 2011/0138305 A1 — Random matching and real-time compatibility assessment](https://patents.google.com/patent/US20110138305)
- [WO 2018/009890 A1 / US 10,089,074 / US 11,275,557 — Relationship analysis utilizing biofeedback information](https://patents.google.com/patent/WO2018009890A1/en)
- [US 2019/0026681 A1 — Data-driven identification of talent](https://patents.google.com/patent/US20190026681A1/en)
- [US 11,000,218 — Dynamically providing behavioral insights for individuals and groups](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11000218)
- [US 10,902,384 B2 — Assessing employment candidates](https://patents.google.com/patent/US10902384B2/en)
- [EP 4,577,958 A1 — Automated machine learning pipeline deployment](https://patents.google.com/patent/EP4577958A1/en)
- [US 12,367,418 — ML model verification for assessment pipeline deployment](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12367418)
- [US 12,242,469 — Automatic generation of training and testing data for ML models](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12242469)
- [US 10,452,992 B2 — Interactive interfaces for machine learning model evaluations](https://patents.google.com/patent/US10452992B2/en)
- [Recentive Analytics, Inc. v. Fox Corp., 134 F.4th 1205 (Fed. Cir. 2025) — slip opinion](https://www.cafc.uscourts.gov/opinions-orders/23-2437.OPINION.4-18-2025_2500790.pdf)
- [Mintz — Recentive Analytics v. Fox: patent eligibility of machine learning claims](https://www.mintz.com/insights-center/viewpoints/2231/2025-05-02-recentive-analytics-v-fox-federal-circuit-provides)
- [Sterne Kessler — 2025 Federal Circuit IP Appeals: Recentive v. Fox](https://www.sternekessler.com/news-insights/insights/2025-federal-circuit-ip-appeals-recentive-analytics-inc-v-fox-corp-134-f-4th-1205-fed-cir-2025-dyk-prost-goldberg/)

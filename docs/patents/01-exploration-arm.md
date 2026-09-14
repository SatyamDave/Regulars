# Patent Analysis 01 — The Permanent Randomized Exploration Arm

**Status:** research memo, not legal advice. Prepared for internal IP strategy.
**Date:** 2026-09-14
**Subject code:** `regulars/assembly/exploration.py`, `docs/technical-architecture.md` §1 rule 3, §5, §6 Problem 1 / Problem 2.

---

## 1. Mechanism Summary

Regulars assembles groups of six strangers for real dinners. Pairwise compatibility is predicted by an LLM-simulation judge; groups are formed by a max-min set-partitioning optimizer over those predicted pair scores.

The mechanism under analysis is the **exploration arm**: a permanent, non-terminating allocation of roughly 30% of live production group assemblies to randomized assembly rather than model-driven assembly. Concretely, as implemented in `regulars/assembly/exploration.py`:

1. **Arm assignment happens at the user level, before either optimizer runs.** `assemble_round` computes the number of table slots, draws a Bernoulli method per slot (`choose_method`, default `exploration_rate = 0.30`), then randomly carves off `group_size` users per random slot from a shuffled pool. Users therefore enter the random arm independently of their predicted scores.
2. **Both arms share the same hard-constraint prefilter.** The `plausible` pair set (zip, availability, stated exclusions) is enforced identically in both arms. Scores are masked to `plausible` before the model arm sees them, so an implausible pair is unscored (`-inf`) and cannot co-group in either arm. The arms differ **in scoring only**: the model arm maximizes the minimum within-group pair score; the random arm ignores scores entirely.
3. **The random arm samples uniformly over *feasible partitions*, not uniformly over all partitions.** `random_groups` performs rejection sampling — shuffle, chunk into groups of `group_size`, accept only if every within-group pair is in `plausible` — for up to `max_tries` (default 1,000) attempts.
4. **When rejection sampling fails, the relaxation is recorded, not hidden.** The fallback is a randomized greedy placement that relaxes *seating everyone* (leaving leftovers) and never relaxes plausibility. The resulting groups carry `relaxed=True`, explicitly flagging that the sample is no longer uniform — i.e., the record self-identifies as unfit for the unbiased estimator.
5. **Unseatable randomly-armed users fall through to the model arm** rather than being dropped, preserving service quality without contaminating the random arm.
6. **`assembly_method` is a first-class immutable column on `groups`**, present from day one (§1 rule 3), not a temporary experiment flag: `groups(id, zip, created_at, assembly_method, status)` with `assembly_method: 'model' | 'random'`.
7. **Every arm decision emits a human-readable audit record** (`decision_type='method_chosen'` / `'group_assembled'`) carrying `assembly_method`, `exploration_rate`, `min_score`, `relaxed`, and `tries`.

The arm is then consumed **twice**:

- **(a) As the unbiased label source.** Outcome labels are directed round-robin ratings (`ratings(plan_id, rater_id, ratee_id, would_see_again, felt_connected_1_7)`) collected after a real-world physical event (a dinner), six people, 30 directed dyads per dinner. Model-arm outcomes are observed only on pairs the model chose and are therefore contaminated; random-arm outcomes are the only clean evaluation set. These feed judge calibration (isotonic regression), the Social Relations Model variance decomposition (§5), and retraining. Cross-validation is **grouped on `user_id`, never on pair** (§6 Problem 2), so a person never straddles train and test.
- **(b) As a live counterfactual control arm reported as a product KPI.** "Round-2 retention, model arm vs. random arm, target +15pp" is an externally reported success-bar metric (§6 success bar table).

The novelty question is whether this combination — and specifically the group-partition mechanics in (2)-(5) — is patentable over a very crowded field.

---

## 2. Prior Art Found

### 2.1 Directly on-point patents (the core mechanism is anticipated)

**US 12,513,350 B1 — "Dynamic recommendation system using reinforcement learning for continual learning"** (PayPal Inc., filed 2024-08-07, published 2025-12-30).
https://patents.google.com/patent/US12513350B1/en
This is the closest patent found. It claims a recommendation system maintaining a **"random exploration lane"** as a standing data-traffic lane alongside streaming and batch lanes, where "randomized data can be obtained by the recommendation system using a random exploration lane," expressly to provide "a source of unbiased data, which improves model performance when deployed into production." The specification sizes the lane at "about 1% of the user actions performed by the plurality of users." This anticipates the core idea of a *persistent production randomization lane whose purpose is unbiased retraining data*. The differences from Regulars are rate (1% vs 30%), the absence of any group/partition structure, and the absence of a business-KPI control-arm reading.

**US 11,204,973 B2 — "Two-stage training with non-randomized and randomized data"** (Microsoft Technology Licensing, LLC / LinkedIn lineage, filed 2019-06-21).
https://patents.google.com/patent/US11204973B2/en
Trains a model in phase one on abundant biased production interaction data, then refines it in phase two on deliberately randomized (shuffled result ordering) data "to remove the bias from the machine-learned model," at roughly a 95/5 split. Directly anticipates *using a randomized production slice to de-bias a model trained on the system's own non-randomized decisions*.

**US 12,038,986 B2 — "Recommendation model training method, recommendation method, apparatus, and computer-readable medium"** (Huawei Technologies Co., Ltd., priority 2019-10-31).
https://patents.google.com/patent/US12038986B2/en
Defines a **"random launch policy"** under which a recommended item "is randomly displayed to a user" with equal probability, expressly to produce "training samples without bias," contrasted against ranked display producing biased samples. Notably it *declines* permanent large-scale randomization on revenue grounds, using a small randomized sample to fit an interpolation/correction model instead. This is the closest art to the "arm indicator distinguishes randomly-selected from model-selected records, and the distinction drives training" concept.

**US 2024/0403713 A1 — "Implementing and maintaining feedback loops in recommendation systems"** (Netflix, Inc., filed 2024-05-30).
https://patents.google.com/patent/US20240403713A1/en
Uses a "default-model-random" baseline trained on "random uniform exploration data that has no feedback loop effect," and randomly splits users into groups each running its own closed loop, to measure feedback-loop degradation. Relevant to the "random arm as a measurement baseline" half of Regulars' claim, though it frames random exploration as a comparison baseline rather than a permanent production allocation.

**US 9,733,811 B2 — "Matching process system and method"** (Match Group lineage).
https://patents.google.com/patent/US9733811B2/en
The Match Group patent family covers relevance-algorithm matching on user attributes and behavior. Reporting on this family (e.g. *Time*, "When Love and the Algorithm Don't Mix," https://time.com/6694129/dating-app-inequality-essay/) notes the specification contemplates that pairings "need not be completely random" and that certain pairings may be excluded or biased by pairing criteria — i.e., constrained randomization in matching is already disclosed in the dating-app patent literature, though not as a labeled evaluation arm.

**US 2011/0138305 A1 — "Method and system for random matching and real-time compatibility assessment to facilitate serendipitous dating."**
https://patents.google.com/patent/US20110138305
Selects an individual **randomly** from each virtual matching pool. Important as prior art for *random pairing in a matching service*, but the randomness serves a serendipity/UX goal; the patent explicitly does not train or evaluate a compatibility model on the random outcomes.

### 2.2 Adjacent patents in the social-group / in-person-event space (relevant to the "group + physical event" framing)

- **US 11,562,303 B2 — "System and method for real-time geo-physical social group matching and generation"** (Rockspoon, Inc., priority 2019-11-21). https://patents.google.com/patent/US11562303B2/en — Forms groups of patrons to "meet, dine, and socialize" at a venue, using ML over customer and restaurant profiles, with a "feedback and ratings portal" whose ratings "adjust machine learning algorithm training data, weights, biases." **This is the closest art to Regulars' overall product**: ML group assembly for in-person dining with rating-driven retraining. It contains **no randomization and no unbiased-evaluation arm** — which is precisely the gap Regulars fills.
- **US 11,983,225** — real-time confidential electronic matching and grouping of users physically present at a live event.
- **US 9,077,767 B2 — "Inception of live events"** — stores event data so matching algorithms can use it, and adapts match thresholds based on observed attendance.
- **US 2010/0185625 A1** — evaluating relationship compatibility among members of a social network and referring compatible members to each other.
- **US 2005/0060183 A1** — system and method for providing feedback for social compatibility.
- **US 2008/0098313 A1** — developing and managing group social networks, expressly including scheduling dinner-party events.
- **US 9,589,058** — methods and systems for social matching (matching strangers on interests/preferences).
- **US 2014/0047023 A1** (Meta/Facebook lineage) — generating guest suggestions for events in a social networking system.

### 2.3 Non-patent literature (this is the strongest anticipation)

The "log a uniformly random slice of production decisions so you can evaluate and retrain without selection bias" idea is foundational, published, and fifteen years old:

- **Li, Chu, Langford, Wang (WSDM 2011), "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms"** — https://arxiv.org/pdf/1003.5956 — proves replay evaluation is unbiased when the logging policy chose arms uniformly at random; validated on Yahoo! Front Page production traffic.
- **Li, Chu, Langford, Schapire (WWW 2010), "A Contextual-Bandit Approach to Personalized News Article Recommendation"** — https://arxiv.org/pdf/1003.0146 — the random bucket in production.
- **"An Unbiased Data Collection and Content Exploitation/Exploration Strategy for Personalization"** — https://arxiv.org/pdf/1604.03506 — explicitly an *unbiased data collection* production strategy.
- **Randomized-exposure production datasets**, each of which is literally a permanent or semi-permanent random arm in a live system: **Yahoo! R3** (MNAR train / MCAR random test), **Coat** (24 self-selected + 16 randomly-exposed items per user), **KuaiRand** (https://arxiv.org/pdf/2208.08696, randomly exposed videos in the Kuaishou production feed), and the **Open Bandit Dataset** (ZOZO; a uniform-random policy running in production alongside a Bernoulli Thompson Sampling policy, ~26M interactions).
- **Schnabel et al., propensity-weighted recommendation; "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback"** (https://arxiv.org/pdf/1909.03601); **"Bounding System-Induced Biases in Recommender Systems with A Randomized Dataset"** (https://arxiv.org/pdf/2303.11574) — the entire MNAR/debiasing literature presupposes a randomized holdout.
- **"Evaluating Online Bandit Exploration In Large-Scale Recommender System"** — https://arxiv.org/pdf/2304.02572.
- **"Reducing Symbiosis Bias Through Better A/B Tests of Recommendation Algorithms"** — https://arxiv.org/pdf/2309.07107 — directly on A/B testing recommenders in production, including how exploration data leaks between arms.
- **Spotify / Netflix practice**: epsilon-greedy exploration on the Home surface (https://research.atspotify.com/2025/9/calibrated-recommendations-with-contextual-bandits-on-spotify-homepage); Netflix artwork exploration with regret "amortized across our large member base."

On the **group/partition** side specifically:

- **"Exploration vs. Exploitation in Team Formation"** — https://arxiv.org/pdf/1809.06937 — explicitly models team formation as a combinatorial bandit where "each team can be modeled as a basic action and one chooses a subset that corresponds to a partition of the workers into teams." **This is the single most dangerous piece of prior art for any claim that leans on "groups instead of pairs."**
- **Bento & Ioannidis, "Group Recommendations Via Multi-armed Bandits" (WWW 2012)** — https://archives.iw3c2.org/www2012/proceedings/companion/p463.pdf.
- **"An Arm-Wise Randomization Approach to Combinatorial Linear Semi-Bandits"** — https://arxiv.org/pdf/1909.02251.
- **"Ranked Prioritization of Groups in Combinatorial Bandit Allocation"** (IJCAI 2022) — https://www.ijcai.org/proceedings/2022/0723.pdf.

On the **always-on control arm as a business KPI** side:

- The "always-on global holdout" — permanently withholding a small random share of the addressable population from all treatment and reading the gap on conversion, retention and revenue — is standard, widely documented CRM and incrementality practice, not a novel invention. See e.g. https://crmknowledgebase.com/measurement/holdouts-and-control-groups and https://arxiv.org/pdf/1705.00634 ("Counterfactual-based Incrementality Measurement in a Digital Ad-Buying Platform").

On **leakage-safe cross-validation**:

- Person-grouped (as opposed to record-level) cross-validation is textbook. `GroupKFold` has shipped in scikit-learn for over a decade; "assigning each identity group wholly to one split" is the standard statement of the technique. See https://arxiv.org/pdf/2311.04179 ("On Leakage in Machine Learning Pipelines"). **Claim 3 of any application should not rest on this.**

On the **Social Relations Model**:

- Kenny's SRM and the round-robin design date to 1979 and are thoroughly published (https://davidakenny.net/srm/soremo.htm; TripleR, https://link.springer.com/article/10.3758/s13428-011-0150-4). SRM itself is unpatentable prior art — a mathematical/statistical method, published, and per §101 a "mathematical concept."

---

## 3. Novelty Analysis

### What is NOT novel (be blunt about this)

| Element | Status |
|---|---|
| Exploration/exploitation, epsilon-greedy, multi-armed bandits | Textbook. Decades old. |
| A permanent fraction of live production traffic routed to uniform-random decisions | Anticipated by US 12,513,350 B1, and by KuaiRand / Open Bandit Dataset / Yahoo R3 in practice. |
| Using that random slice as the unbiased label set for retraining/calibrating the model that made the other decisions | Anticipated by US 11,204,973 B2 and US 12,038,986 B2, and by Li et al. 2011. |
| Recording per-record whether the decision was random or model-made, and branching training on that flag | Anticipated by US 12,038,986 B2 (biased vs unbiased sample sets from random-launch policy). |
| A permanent randomized holdout read as a business KPI (retention lift) | Standard always-on global holdout / incrementality practice. |
| Person-grouped cross-validation to prevent identity leakage | Textbook; `GroupKFold`. |
| Directed round-robin ratings decomposed into actor / partner / relationship variance | Kenny 1979. Published mathematics. |
| ML matching of strangers into groups for an in-person dinner with rating feedback driving retraining | US 11,562,303 B2 (Rockspoon). |

Every *individual* element of the mechanism is in the prior art. An examiner will find US 12,513,350 B1 and US 11,204,973 B2 within one search.

### What IS arguably novel

Three things survived the search. Only the first two are technical.

**(N1) Constrained uniform partition sampling as the exploration policy, with an explicit uniformity-validity flag.**
Every piece of prior art randomizes an *independent per-impression decision* (which article, which item, which rank order). Regulars' decision is a **partition of a shared pool under pairwise feasibility constraints** — a structurally different randomization problem. Naively randomizing within a constrained combinatorial decision does *not* yield unbiased labels, because the constraint satisfier reintroduces structure. `random_groups` addresses this by rejection-sampling uniformly over the set of *feasible* partitions (every within-group pair in `plausible`), and, when rejection fails, relaxing *seating coverage* rather than feasibility and setting `relaxed=True` so the non-uniform sample self-identifies and can be excluded from the unbiased estimator. I found **no prior art claiming rejection-sampled uniform-over-feasible-partitions exploration with a recorded uniformity-validity flag.** The team-formation bandit literature (arXiv 1809.06937) chooses partitions to maximize reward; it does not sample uniformly over the feasible partition set for label de-biasing.

**(N2) Arm assignment at the entity level prior to partitioning, with fall-through.**
In flat recommendation, randomizing *the decision* and randomizing *the user* are equivalent because impressions are independent. In partition assembly they are not: if you flipped the coin per table after scoring, arm membership would correlate with score structure (high-scoring clusters would be systematically over- or under-represented). `assemble_round` therefore assigns the arm to *users* before either optimizer runs, and routes unseatable random-arm users into the model arm rather than dropping them (dropping would condition the random arm on constraint density, reintroducing bias). This entity-level-before-combinatorial-decision ordering is a real, non-obvious technical detail with no found counterpart.

**(N3) Dual consumption of one arm — unbiased label source *and* externally reported retention control — with a single immutable per-record field.**
Honestly assessed: this is a *design* observation, not a technical one. The prior art has both halves (PayPal/Microsoft/Huawei for labels; always-on global holdout for the KPI). Claiming the conjunction is likely obvious under §103: a person of ordinary skill who already runs a permanent random lane would plainly also compare outcomes between lanes. **Do not build the independent claim on N3.** It is good positioning copy; it is weak IP.

The "social group, not dyad" and "physical real-world event outcome" framings do **not** by themselves create novelty — they are the field of use. Applying a known technique to a new field of use is the archetypal §103 obviousness rejection, and post-*Alice* it also invites a §101 rejection for "applying standard machine learning to new data" (see the 2024 USPTO AI SME Update guidance, https://www.federalregister.gov/documents/2024/07/17/2024-15377/2024-guidance-update-on-patent-subject-matter-eligibility-including-on-artificial-intelligence, and practitioner commentary at https://www.dykema.com/news-insights/ai-and-software-patents-in-2025-new-leadership-and-101-eligibility-guidance.html).

---

## 4. Patentability Verdict

### **WEAK — narrow filing possible, broad filing not recommended.**

Reasoning:

1. **The mechanism as described in the brief is anticipated.** "Permanently allocate a fixed fraction of live decisions to random, use it for unbiased retraining and as a control arm, record the split per-decision" is substantially US 12,513,350 B1 + US 11,204,973 B2 + US 12,038,986 B2 + standard holdout practice. A broad independent claim on that would not survive prosecution and would not survive an IPR if it somehow issued.
2. **The social-group / dinner framing is field-of-use, not invention.** §103 obviousness rejection is near-certain for any claim whose novelty rests on "but for dinners" or "but for groups of six instead of pairs." Rockspoon's US 11,562,303 B2 already occupies ML group assembly for in-person dining with feedback retraining.
3. **§101 risk is real.** Group assembly, rating collection, and statistical de-biasing read as a mathematical concept plus a method of organizing human activity. The 2024 AI SME guidance and the September 2025 *Ex parte Desjardins* ARP decision have moved the bar in applicants' favor, but the surviving path requires claiming *a specific technical mechanism*, not *a desired outcome*. "Unbiased evaluation of a matching model" is an outcome. "Rejection-sample uniformly over the feasible partition set, and set a validity flag when you fall back to a non-uniform greedy placement" is a mechanism.
4. **There is a narrow, genuinely defensible core** — N1 + N2 — but it is narrow enough to design around by, for example, sampling feasible partitions by a different method. Its commercial value is as a defensive/optics asset, not as an enforcement asset.
5. **The real moat is not the algorithm.** It is the accumulated corpus of (immutable persona version → prediction → directed post-dinner rating) triples, which cannot be replicated without running the dinners. That asset is protected by **trade secret and data ownership, not by patent** — and a patent would *publish* the exact evaluation design that makes the corpus trustworthy.

### Recommended posture

- **Primary: trade secret** for `exploration_rate`, the prefilter thresholds, scoring weights, the scenario/seed design, and the persona schema. None of these need publishing and all lose value when published.
- **Secondary: defensive publication** of N1/N2 (a dated engineering blog post or an arXiv preprint) if the goal is freedom-to-operate rather than exclusion. This blocks a competitor from patenting it against Regulars at a fraction of the cost.
- **Optional: one narrow provisional** on the claims in §5 below, if IP optics matter for a fundraise. Budget it as a marketing expense with a modest chance of a narrow issued claim — not as an enforceable exclusion right. File before any public disclosure of §6 of the architecture doc, since the US grace period is 12 months and most foreign jurisdictions have none.
- **Do not** file anything resting on SRM, grouped cross-validation, or "random arm compared to model arm on retention." Those will be rejected and the rejections become file-wrapper estoppel against the parts that matter.

---

## 5. Draft Claims

Informal sketches for the narrowest defensible combination (N1 + N2). Deliberately mechanism-heavy and outcome-light, for §101 survivability.

### Independent Claim 1 (method)

A computer-implemented method of generating selection-bias-free training labels for a group-assembly model, comprising:

1. maintaining, for a pool of candidate participants, a feasibility set of participant pairs satisfying one or more hard constraints, wherein the feasibility set is applied identically to both a model-assembly path and a randomized-assembly path;
2. determining a number of group slots from the pool size and a target group size, and, **prior to computing or applying any compatibility score to the pool**, assigning each group slot to either the randomized-assembly path with a fixed non-zero probability or to the model-assembly path with the complementary probability;
3. partitioning the pool into a randomized-arm subset and a model-arm subset by randomly selecting, independently of any compatibility score, a number of participants equal to the target group size times the number of randomized slots;
4. assembling groups from the randomized-arm subset by **rejection sampling**, comprising repeatedly generating a candidate partition of the randomized-arm subset into groups of the target group size and accepting the candidate partition only when every within-group participant pair is a member of the feasibility set, such that an accepted partition is drawn uniformly from the set of feasible partitions;
5. responsive to no candidate partition being accepted within a bounded number of attempts, generating a fallback assembly that (i) preserves the feasibility constraint of step (4), (ii) relaxes a requirement that every participant of the randomized-arm subset be seated, and (iii) **stores a uniformity-validity indicator marking the resulting groups as non-uniformly sampled**;
6. routing any participant of the randomized-arm subset not seated by step (4) or (5) to the model-assembly path rather than removing the participant from the round;
7. assembling groups from the model-arm subset by optimizing an objective over predicted pairwise compatibility scores masked to the feasibility set;
8. storing, for each assembled group, an immutable assembly-method field recording whether the group originated from the randomized-assembly path or the model-assembly path;
9. after occurrence of a scheduled in-person event for each assembled group, recording a set of directed participant-to-participant outcome ratings, each rating associating a rater identifier and a distinct ratee identifier; and
10. training or calibrating the predicted pairwise compatibility model using only those directed outcome ratings whose group record has an assembly-method field indicating the randomized-assembly path **and** whose uniformity-validity indicator is unset.

### Dependent Claim 2

The method of claim 1, wherein training or calibrating in step (10) uses a cross-validation partition in which every record associated with a given participant identifier is assigned wholly to one fold, such that no participant identifier appears in both a training fold and an evaluation fold.

### Dependent Claim 3

The method of claim 1, further comprising decomposing the directed outcome ratings of each group into a rater-specific component, a ratee-specific component, and a dyad-specific residual component, and reporting a measure of the compatibility model's predictive association with the dyad-specific residual component computed over groups having the randomized assembly-method field.

### Dependent Claim 4

The method of claim 1, wherein the objective optimized in step (7) is a maximum of the minimum predicted pairwise compatibility score within each assembled group, and wherein pairs absent from the feasibility set are assigned a score that renders them unselectable by the optimization rather than merely penalized.

### Dependent Claim 5

The method of claim 1, further comprising computing a retention statistic over participants of groups having the randomized assembly-method field and a corresponding retention statistic over participants of groups having the model assembly-method field, and emitting a difference between the two statistics as a measure of the compatibility model's incremental effect, wherein the same stored assembly-method field of step (8) serves both step (10) and this computation.

### Dependent Claim 6

The method of claim 1, further comprising, for each assembled group, generating and persisting a natural-language audit record identifying the assembly path, the fixed probability used in step (2), and the uniformity-validity indicator of step (5).

### Dependent Claim 7

The method of claim 1, wherein the predicted pairwise compatibility score is produced by executing a simulated dialogue between two language-model agents conditioned on immutable versioned participant persona specifications, and wherein the persona specification version used to produce each score is persisted with the score.

**Claim-drafting notes.** Claim 1 steps (4), (5) and (6) are the load-bearing novelty; strip them and claim 1 reads on US 12,513,350 B1. Step (2)'s "prior to computing or applying any compatibility score" is the N2 limitation and should not be surrendered in prosecution. Claims 2 and 3 are weak on their own (GroupKFold; Kenny 1979) and are included only as dependent fallback positions — expect them to be rejected if ever presented independently. Claim 5 is the dual-use claim; expect a §103 rejection over the always-on-holdout literature, and keep it dependent.

---

## 6. Design-Around Notes

Prior art that Regulars should explicitly cite, design around, or at minimum be able to distinguish on demand:

1. **US 12,513,350 B1 (PayPal, random exploration lane).** *Cite it.* It is the closest art and an undisclosed known reference is an inequitable-conduct risk. Distinguish on: the decision unit (an independent per-user recommendation vs. a constrained partition of a shared pool), and the absence of any feasibility-constrained uniform sampling or uniformity-validity flag.
2. **US 11,204,973 B2 (Microsoft, two-stage randomized de-biasing).** *Cite it.* Distinguish on: it re-trains in two phases on a de-biasing corpus; Regulars runs a permanent parallel arm and filters the label set by an immutable per-decision field rather than sequencing training phases.
3. **US 12,038,986 B2 (Huawei, random launch policy).** *Cite it.* This is the strongest art against the "record whether it was random or model-chosen, and train differently on each" concept. Distinguish on the partition structure and on the fact that Huawei's specification affirmatively teaches *away* from permanent large-scale randomization (revenue loss) — teaching-away is a useful argument for the 30% permanent rate.
4. **US 11,562,303 B2 (Rockspoon, geo-physical social group matching for dining).** *Cite it and watch it.* This is the closest art to the Regulars **product**, not just the mechanism, and it already claims ML group formation for in-person dining with rating-driven retraining. Before any product-level filing, get a freedom-to-operate read on this family and its continuations. The distinguishing feature is exactly the exploration arm, which Rockspoon lacks.
5. **Match Group patent family (US 9,733,811 B2 and relatives).** Large, well-funded, litigious, and already discloses constrained/partially-random pairing in a matching service. Do not file any claim whose randomization limitation could be read onto "pairings need not be completely random." Keep Regulars' randomization limitation tied to *uniform sampling over a feasible partition set for label generation*, which Match's disclosure does not reach.
6. **US 2011/0138305 A1 (serendipitous random dating matches).** Anticipates "random matching in a compatibility service." Any claim that merely recites "randomly matching a subset of users" is dead on arrival. Every claim must recite the *label-generation and uniformity* purpose and mechanism.
7. **arXiv 1809.06937, "Exploration vs. Exploitation in Team Formation."** The most dangerous NPL. It already frames team formation as choosing partitions in a combinatorial bandit. Never argue that "exploration applied to groups rather than pairs" is itself the invention — that argument loses to this paper. Argue uniform-over-feasible-partitions rejection sampling with recorded relaxation instead.
8. **Li et al. (WSDM 2011) and the randomized-exposure dataset family (Yahoo R3, Coat, KuaiRand, Open Bandit Dataset).** These establish that permanent production random buckets for unbiased evaluation are ordinary skill in the art. Any marketing or investor claim that Regulars "invented" the unbiased exploration arm is factually wrong and creates litigation and diligence risk. The correct external framing is: *the technique is well established in large-scale recommendation; nobody in the friend-matching category is disciplined enough to run it, and the resulting corpus is the asset.*
9. **Always-on global holdout practice.** Prevents any independent claim on "reporting the random-arm/model-arm retention difference as a metric." Keep that in a dependent claim only.
10. **Kenny's SRM and GroupKFold.** Unpatentable published mathematics and standard tooling. Present only as dependent claims, and never as the point of novelty.

### Operational guardrails

- **File before disclosure.** If §6 of `docs/technical-architecture.md`, the `exploration.py` source, or any talk describing the arm is made public, the 12-month US grace period starts and most non-US rights are lost immediately. Decide file-vs-publish before the next external deck.
- **Preserve the uniformity-validity flag in the data model.** `relaxed` is currently a field on `RandomGroups` and appears in the audit payload, but the `groups` table in §1 stores only `assembly_method`. **If a patent is pursued, the schema must persist the relaxation flag on the group record**, because claim 1 step (5) depends on it and an implementation that does not store it does not practice the claim. This is also a correctness issue independent of IP: without it, non-uniform fallback groups silently contaminate the "unbiased" label set.
- **Keep `exploration_rate` configurable and unpublished.** It is a tuning secret; disclosing 30% in a patent dedicates the range to the public.
- **Do not name SRM, isotonic calibration, or GroupKFold as inventions anywhere external.** They are correct choices and good engineering signals; claiming them as novel undermines credibility with technical diligence partners who know the literature.

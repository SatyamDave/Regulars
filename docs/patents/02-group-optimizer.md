# Patent / Prior-Art Analysis 02 — Group Optimizer

**Subject:** max-min group assembly with continuity bonus and feedback-driven prune/backfill
**Code under analysis:** `regulars/assembly/optimizer.py`, `regulars/assembly/rounds.py`
**Spec under analysis:** `docs/technical-architecture.md` §5 (the judge), §7 (group assembly)
**Date:** 2026-09-14
**Status:** research memo, not legal advice. A registered patent attorney must run a formal
freedom-to-operate and patentability search before any filing decision.

---

## Mechanism Summary

Regulars assembles dinner tables of six from a pool of candidate users. Four components matter
for this analysis.

**1. Bottleneck (max-min) group objective.** A group's score is the *minimum* pairwise
compatibility score over all 15 dyads in the group, never the mean or the sum
(`optimizer.group_score`). The product premise is that one bad dyad ruins a table of six, so the
worst pair is the only pair that counts. Partitioning n candidates into size-6 groups so as to
maximise this minimum is a max-min set-partitioning problem and is NP-hard.

**2. Greedy-seed plus swap local search.** The solver seeds each group from the highest-scoring
available pair, grows it one member at a time by whichever candidate leaves the resulting minimum
highest, then runs steepest-ascent local search over member swaps (group↔group and
group↔leftover) for up to 200 iterations. The accepted objective is the vector of per-group
minima sorted ascending, compared lexicographically — so the global minimum dominates, and ties
on it are broken by improving the next-worst group.

**3. Hard constraints as unscored pairs.** `PairScores.get` returns `-inf` for any pair with no
score. Hard-constraint rejections are expressed by simply not scoring the pair. Because the group
score is a minimum, any group containing such a pair evaluates to `-inf`, and seeding, growth and
swaps all refuse it. Constraint satisfaction and the bottleneck objective are therefore the same
arithmetic operation rather than two separate mechanisms, and no penalty weight needs tuning.

**4. Continuity bonus.** The single additive term in the module. `Continuity.adjusted` adds a
small constant (`DEFAULT_CONTINUITY_BONUS = 0.05`) to the score of every pair that shared a group
in the *previous round*, before the minimum is taken. Its stated purpose is to bias the optimizer
toward preserving converging social groups across repeated rounds rather than re-shuffling
membership weekly. Three properties are load-bearing: the bonus is small enough to re-order
near-ties without rescuing a genuinely bad pairing; `-inf + bonus == -inf`, so it can never lift a
hard-constraint-rejected or unscored pair; and reported `global_min` / `per_group_min` are
recomputed against the *raw* scores, so what is persisted is the judge's actual prediction rather
than the continuity-inflated optimization score.

**5. Feedback-driven prune and backfill.** After each dinner, `rounds.prune_and_backfill` closes
the loop:

- **Directed multi-rater consensus.** Every attendee rates every other attendee on
  `would_see_again` (§1, §5: ratings are directed; a→b is not b→a). A member is removed when at
  least `MIN_RATERS = 3` other *attending* members rated them and fewer than
  `APPROVAL_THRESHOLD = 50%` said they would see them again. Self-ratings, ratings from
  non-attendees and ratings from non-members are discarded. Under three raters, nobody is removed.
- **Commitment rule.** A member who paid a deposit and did not attend is removed regardless of
  ratings.
- **Viability floor.** A removal that would take the group below `min_keep` members is not
  applied; the spared member is recorded in notes.
- **Backfill re-optimizing the same objective.** Open seats are refilled from a priority queue,
  each time adding whichever waiting candidate leaves the group's *minimum* pair score highest,
  ties to the earlier queue position, and never adding a candidate unscored against any current
  member.
- **Explainability.** Every assemble, removal and addition writes an `AuditRecord` with a
  natural-language reason addressed to the affected user (§10: removals are explainable).

The closed loop is the point: real-world post-event peer ratings change live group membership,
and the surviving core is then carried into the next round's global re-optimization with its
pairwise scores boosted by the continuity bonus.

---

## Prior Art Found

### A. The max-min objective itself — well-trodden, treat as unpatentable baseline

This is the honest headline. Egalitarian / Rawlsian max-min objectives over partitions are
textbook material in fair division and combinatorial optimization, and the specific case of
partitioning users into groups on *pairwise compatibility scores* under a *worst-case* objective
has been published explicitly.

| Reference | Relevance |
|---|---|
| Rajkumar, Mukherjee & Tulabandhula, **"Learning to Partition using Score Based Compatibilities"**, AAMAS 2017 ([arXiv:1703.07807](https://arxiv.org/abs/1703.07807)) | **Closest academic art on the core objective.** Studies partitioning users into groups from learned pairwise compatibilities, defines four objectives optimizing for *average and worst-case* compatibilities, proves the unstructured problems NP-hard with inapproximability results. This is our §7 objective, named and analysed, nine years before our filing window. |
| **Maximally Diverse Grouping Problem (MDGP) literature** — Feo & Khellaf; Lai & Hao, "Maximally diverse grouping: an iterated tabu search approach"; Brimberg, Mladenović & Urošević, skewed general VNS ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025514010263)); Gallego et al.; [GRAFO MDGP page](https://grafo.etsii.urjc.es/maximally-diverse-grouping/); [arXiv:2410.08482](https://arxiv.org/html/2410.08482) | Equal-size partitioning of n elements into groups on pairwise distance matrices, NP-hard, solved by **greedy construction plus swap-based local search / tabu / VNS** — structurally identical to our solver. Min-max and min-sum variants are both standard in this line. The "swap between the two worst groups" move is published. |
| Munson & Hutchins, **"Maximizing Satisfaction in Group Formation"** / GroupForm ([PDF](https://www.smunson.com/portfolio/projects/groupform/paper.pdf)) | Maximizes count of "ideal" groups — groups in which *every* member wants to work with *every* other member. That is a thresholded form of the same weakest-link intuition: one bad dyad disqualifies the group. |
| **Egalitarian item allocation / Santa Claus problem** ([Wikipedia](https://en.wikipedia.org/wiki/Egalitarian_item_allocation)); Aziz, *Algorithmic Fair Allocation of Indivisible Items* ([SIGecom Exchanges](https://www.sigecom.org/exchanges/volume_20/1/AZIZ.pdf)) | The max-min / Rawlsian objective over partitions, with hardness and approximation results. Foundational, public domain. |
| **Welfare-Centric Clustering** ([arXiv:2508.10345](https://arxiv.org/pdf/2508.10345)); **Fair Clustering Under a Bounded Cost** ([arXiv:2106.07239](https://arxiv.org/pdf/2106.07239)) | Egalitarian objectives applied to clustering — minimize maximum group disutility. |
| Bhaskara et al. / team-formation with faultlines ([arXiv:1811.05015](https://arxiv.org/pdf/1811.05015)); *Team Formation amidst Conflicts* ([arXiv:2403.00859](https://arxiv.org/pdf/2403.00859)); *Computational Team Assembly with Fairness Constraints* ([arXiv:2306.07023](https://arxiv.org/pdf/2306.07023)) | NP-hardness and inapproximability of compatibility-driven team partitioning; conflict/incompatibility edges as hard exclusions. |

**Verdict on component 1:** no claim reciting "maximize the minimum pairwise compatibility over a
partition into fixed-size groups, solved by greedy seed plus swap local search" is defensible.
It is anticipated on the merits and, even where not strictly anticipated, obvious.

### B. US11983225 — the closest single patent reference (and the biggest obstacle)

**"Method involving real-time confidential electronic matching for forming anonymous connections
at live events"**, US 11,983,225 B1, individual inventor, priority 2018-02-15
([Google Patents](https://patents.google.com/patent/US11983225)).

This is the reference to design around. It discloses:

- Dividing event attendees into small groups using clustering over a compatibility graph.
- **Multiple rounds** within an event: groups are formed, users are notified, the activity runs,
  and "a new set of groups is then determined, and the process repeats until the event is
  finished."
- **Feedback between rounds** — users may update their interests after each activity, and the
  next round re-optimizes.
- **An explicit prior-round co-membership term in the scoring function**, which *subtracts* the
  sum of edge weights for each group edge that appeared in prior rounds, to avoid repeating
  groupings across rounds.

So the patented art already contains: repeated-round group re-formation, compatibility-graph
scoring, an inter-round feedback path, and a prior-round co-membership adjustment term. Our
continuity bonus is **the same term with the opposite sign**.

That is a serious §103 problem and must be confronted head-on rather than papered over. A sign
flip on a known term, by itself, is the kind of modification an examiner will call an obvious
design choice. The available non-obviousness arguments are: (i) the reference *teaches away* —
its entire purpose is maximizing novel encounters at a single time-boxed event, and reversing the
sign destroys that purpose; (ii) the reference's term is a *sum* adjustment over group edges,
whereas ours is a per-dyad additive term applied *before a minimum is taken*, which has
qualitatively different behaviour (it can only re-order near-ties on the bottleneck pair and
provably cannot rescue a `-inf` pair); and (iii) the reference has no persistent cross-*event*
identity, no post-event rating, and no membership churn. Argument (iii) is the strong one.
Arguments (i) and (ii) are supporting, not load-bearing.

### C. Patents on compatibility-based group/team formation

| Patent | Assignee / priority | What it covers | Gap vs. ours |
|---|---|---|---|
| **US 9,549,070 B2** — "System and method for compatibility-based team formation" ([Google Patents](https://patents.google.com/patent/US9549070B2/en)) | Avaya, 2015-02-12 | Contact-center team assembly from agents, using compatibility information (e.g. Belbin behavioural model) alongside skills; evaluates permutations and "considers each decision as a whole." | One-shot, request-triggered, real-time; **no bottleneck objective** (holistic/aggregate scoring), no rounds, no feedback loop, no membership churn. |
| **US 10,320,734 B1** and **US 10,868,789** — "Social matching" ([Google Patents](https://patents.google.com/patent/US10320734B1/en)) | eHarmony, priority 2011-10-25 | Social-graph-restricted matching with a **feedback loop**: friends thumbs-up/down suggested matches, feedback informs subsequent match generation. | **Dyadic, not group.** No n-person partition, no bottleneck objective, no group membership churn. Establishes "feedback loop improves future matches" as old. |
| **US 2010/0131502 A1** — "Cohort group generation and automatic updating" ([Google Patents](https://patents.google.com/patent/US20100131502A1/en)) | — | Cohort groups created, built and **modified** passively/actively over time by automated and semi-automated means. | Generic cohort membership updating; no pairwise compatibility optimization, no bottleneck objective, no peer-rating trigger. |
| **US 9,519,682 B1** — "User trustworthiness" ([Google Patents](https://patents.google.com/patent/US9519682B1/en)) | — | Trustworthiness scores **dynamically evolve a trusted group by regrouping accounts** as scores change. | Closest art on "scores change → group membership changes automatically," but the score is a per-user reputation scalar, not directed peer ratings, and there is no pairwise optimization. |
| **US 10,140,796 / US 10,388,096** — "Selective seating controller" | — | Recommends seating based on attendee characteristics and preferences via correlation/optimization functions. | Single seating event; no multi-round persistence, no peer feedback churn. |
| **US 11,816,746 B2** — "System and method for dynamic dining party group management" ([Google Patents](https://patents.google.com/patent/US11816746B2/en)) | Rockspoon, priority 2020-01-01 | Despite the promising title: detects late-arriving party members via beacon/biometrics and synchronizes food service timing. Parties are formed by user selection from friend lists. | **Not relevant** to algorithmic group composition. Worth recording so it is not re-found and mis-cited later. |
| US 7,677,970 / 9,375,641 / 10,398,985 — "Social matching of game players on-line" | — | Matchmaking for game sessions using social attributes with a feedback loop. | Session-scoped, no persistent friend-group thesis, no bottleneck partition objective. |

### D. Repeated-round grouping with an *anti*-repetition term (the recurring pattern)

Independently of US11983225, the published art on multi-session grouping overwhelmingly optimizes
*against* repeat co-membership, not for it:

- **"Catalyzing collaborations: Prescribed interactions at conferences determine team formation"**
  ([arXiv:2112.08468](https://arxiv.org/pdf/2112.08468)) — group assignments across sessions are
  optimized simultaneously to *minimize repeated assignments*, so no two participants share a
  small group more than once.
- Social-golfer / round-robin scheduling literature generally — the canonical objective is
  maximizing distinct encounters.
- US11983225's negative prior-round term, above.

This is genuinely useful for us: it establishes that the field's default is churn-maximizing, and
that a *positive* continuity term serves a different and non-obvious product goal. It is the best
non-obviousness evidence available. It is also the reason the sign flip is not purely arbitrary.

### E. Feedback-driven membership churn in group formation

- **"Inclusive Study Group Formation At Scale"** ([arXiv:2202.07439](https://arxiv.org/pdf/2202.07439))
  — long-term study groups in 1,000+ student classes, with students able to **request
  reassignment mid-semester**, iteratively re-running the matcher. The paper notes explicitly that
  the system **"does not use peer-evaluation."** So: persistent groups plus mid-course
  re-optimization is published, but the trigger is *self-reported dissatisfaction by the person
  leaving*, not *directed multi-rater consensus about the person being removed*. That distinction
  is where our claim has to live.
- **CATME / Peerceptiv** and the peer-assessment literature — directed multi-rater peer evaluation
  of teammates is entirely standard in education, including quantified "would work with again"
  scales. It feeds grading and instructor intervention, **not automated membership churn feeding
  a re-optimizer**.
- Group formation in collaborative learning considering prior collaboration groupings
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360131511002284) and the
  ELARS recommender work) — some systems accept previous groupings as an input parameter.
  Directionally relevant to continuity but not quantified as an additive pairwise bonus under a
  bottleneck objective.

### F. Commercial landscape (no blocking art found)

- **HR / team-health tools** — Atlassian (Team Health Monitor), Culture Amp, Officevibe/Workleap,
  Range. Searched assignee patent portfolios via Justia and PatentGuru. Atlassian has ~479 patents
  globally but the portfolio is developer-tooling (databases, UI, collaboration infrastructure);
  **no team-formation or team-composition-optimization patents surfaced.** Culture Amp, Officevibe
  and Range surfaced **no relevant patent filings at all.** These products measure team health;
  none of them *composes* teams algorithmically. This space is open.
- **Dinner / friend-group startups** — Timeleft, 222, Belong, Lunchclub. All publicly describe
  algorithmic group matching with post-event ratings (Timeleft explicitly: "guests rate the venue,
  food and their companions"). **No patent filings located for any of them.** Two consequences:
  no obvious blocking patent from a direct competitor, but also a real §102(a)(1)/§103 exposure
  from their **public disclosures and product behaviour**, which are prior art regardless of
  whether anyone patented them. Timeleft's public description of rating companions and being
  rematched is, at minimum, art against a broad "rate your table, get rematched" claim.
- **Dating majors** — Match Group and eHarmony hold large matching portfolios, but the relevant
  ones (see C) are dyadic. Match Group is litigious on matching patents (Match v. Bumble), so
  freedom-to-operate on any *dyadic* scoring claim deserves separate diligence. Our exposure there
  is limited because we claim group partitioning, not dyadic recommendation.

### G. Subject-matter eligibility — the dominant risk

This is the section that should change the filing decision, more than any single reference.

- **Trinity Info Media, LLC v. Covalent, Inc.**, Fed. Cir. (July 2023), precedential — US
  9,087,321 and 10,936,685 ("Poll-Based Networking System") held ineligible under §101. Claims
  directed to the abstract idea of **"matching users who gave corresponding answers to a
  question"** with no inventive concept; generic computer implementation did not save them.
  ([IPWatchdog](https://ipwatchdog.com/2023/07/16/federal-circuit-says-user-matching-patent-claims-abstract-precedential-eligibility-decision/))
- **British Telecommunications v. IAC/InterActiveCorp (Match)** — a claim to improving online
  dating profiles by analysing responses to proposed matches held ineligible.

User-matching claims are a disfavoured class at the Federal Circuit. A claim reciting "compute
compatibility scores, form groups maximizing the minimum score, collect ratings, replace members"
reads as mental-process-plus-generic-computer and will very likely draw an Alice rejection. The
eligibility-hardening moves available are: recite a **specific non-abstract data structure and
operation** (the `-inf` sentinel making hard-constraint filtering and bottleneck evaluation a
single comparison, avoiding penalty-weight tuning), recite the **bounded-iteration steepest-ascent
search with the lexicographic sorted-minima acceptance vector** as a concrete computational
improvement, and recite the **dual-score bookkeeping** (optimize on bonus-adjusted scores, persist
raw scores). These are real implementation specifics, not field-of-use limitations, which is what
Alice step two wants. They will not make the claim broad.

---

## Novelty Analysis

Decomposed honestly, component by component.

| Component | Novel? | Basis |
|---|---|---|
| Max-min / bottleneck pairwise objective over a group partition | **No** | Rajkumar et al. AAMAS 2017; MDGP min-max variants; egalitarian allocation. Textbook. |
| Greedy seed + swap local search for NP-hard grouping | **No** | Standard MDGP solver family (tabu, VNS, GRASP). |
| Fixed group size 6 | **No** | Arbitrary parameter. |
| Hard constraints encoded as unscored pairs collapsing the group to `-inf` under a min objective | **Marginal** | Not found stated as such, but it is a natural consequence of using a min objective with an absorbing element. Weak on its own; useful as an eligibility-hardening dependent limitation. |
| Compatibility scores derived from LLM-simulated dyadic interaction + a directed judge | **Separate invention** | Covered by a different memo (§3–§5 of the architecture); not claimed here, but its recitation is what gives the claim a concrete non-generic score source. |
| Repeated-round group re-formation with a prior-round co-membership term | **No, and adversely occupied** | US11983225 discloses exactly this with a negative-signed term. |
| Prior-round term with **positive** sign (continuity bonus) pursuing group *persistence* | **Weak-to-moderate** | Sign flip of a disclosed term. Saved, if at all, by the teach-away in §D and by what it is combined with. Not defensible standing alone. |
| Continuity bonus applied **to a bottleneck (min) objective** with the property that it re-orders ties but provably cannot rescue a hard-excluded pair (`-inf + bonus == -inf`) | **Moderate** | Specific interaction between an additive bias term and a min-aggregated objective. Not found in the art. |
| Optimizing on bonus-adjusted scores while **persisting raw scores** for audit/evaluation | **Moderate** | Not found. Concrete, and it matters: it is what keeps §6's evaluation arm uncontaminated by the optimizer's own bias term. |
| **Directed multi-rater consensus threshold** (≥3 raters among *attendees*, <50% would-see-again, self- and non-attendee ratings excluded) triggering deterministic removal | **Moderate** | Peer evaluation is old; *automated membership removal keyed to a directed-rating quorum-plus-threshold rule* is not found in the patent art. |
| **Backfill from a queue maximizing the same minimum pair score** as the original assembler | **Moderate** | The closed-loop element. Not found. The published churn analogues (arXiv:2202.07439) are self-request-driven and do not re-run a bottleneck objective on the residual group. |
| **The closed loop as a whole**: bottleneck assembly → real-world event → directed multi-rater feedback → quorum-gated prune → min-maximizing backfill → continuity-weighted re-assembly of survivors into the next round | **Yes — this is the novelty** | No single reference, and no obvious two-reference combination, discloses a system in which post-event directed peer ratings drive live membership churn that re-optimizes the *same* min-score objective while a positive continuity term carries the surviving core forward. |

**What is actually novel, stated in one sentence:** a group-formation system whose optimization
objective is the group's worst dyad, and in which real-world post-event directed multi-rater
feedback performs surgery on group *membership* — evicting low-consensus members under a rater
quorum and backfilling from a queue against that same worst-dyad objective — while a positive
per-dyad continuity term biases the next round's global re-optimization toward preserving the
surviving core, so the system converges group composition toward persistence rather than
re-shuffling for novelty.

**What is not novel and must not be claimed broadly:** max-min partitioning, greedy+swap
heuristics, compatibility scoring, post-event ratings, feedback improving future matches.

---

## Patentability Verdict

**Weak-to-moderate, narrow only, and probably not worth filing at this stage.**

The three independent reasons:

1. **The core mechanism is public-domain.** Max-min partitioning on pairwise compatibilities,
   solved by greedy seed plus swap local search, is published verbatim in the AAMAS 2017 and MDGP
   literature. Any claim whose novelty rests there is dead on arrival.

2. **The nearest patent already occupies the repeated-round-with-prior-co-membership-term space
   (US11983225), with the sign reversed.** Our best distinguishing arguments are the *persistent
   cross-event identity* and the *post-event churn loop*, not the continuity bonus itself. A
   continuity-bonus-only claim would very likely fall to §103 over US11983225 alone.

3. **§101 is the dominant risk, not §102/§103.** Trinity Info Media makes user-matching claims a
   disfavoured class. Even a claim that survives the art may not survive Alice. Hardening requires
   reciting implementation specifics narrow enough that a competitor routes around them by
   changing a threshold or an aggregation order.

Add the commercial reality: the claim that survives all three filters is narrow enough that
Timeleft, 222 or Bumble could ship a functionally equivalent product by using a mean-with-floor
objective instead of a pure min, or a decay-weighted continuity term instead of a constant bonus,
or a rater quorum of two. Narrow claims in a fast-moving consumer category deter approximately
nobody.

**Recommendation, in priority order:**

1. **Defensive publication now.** Publish §7 and the prune/backfill rules — this repo's
   architecture doc largely already does. It costs nothing, establishes a §102(a)(1) date against
   anyone filing later, and preserves freedom to operate. This is the highest-value action.
2. **Trade secret for the parameters, not the mechanism.** The mechanism is describable in a
   paragraph and will be inferred from product behaviour. The *tuned* values — the continuity
   bonus magnitude relative to the judge's score distribution, the rater quorum, the approval
   threshold, the judge's feature weights — are where the real advantage sits and are not
   externally observable. Keep them out of anything public.
3. **A provisional only if a fundraise needs it.** A $150–300 provisional on the closed loop below
   buys twelve months and the phrase "patent pending" with no eligibility examination. That is a
   defensible use of $300. Converting to a non-provisional (roughly $15–25k through prosecution,
   with a likely Alice rejection to argue through) is not defensible until there is revenue and a
   real competitor.
4. **Do not file on the optimizer.** If anything is ever filed, the continuity bonus and the
   feedback-churn loop are the only parts worth the paragraph. The max-min solver goes in the
   specification as background, never in a claim.

---

## Draft Claims

Sketches for attorney review, not filing text. Drafted deliberately narrow, because the broad
versions are unpatentable. Claim 1 is the closed loop; it deliberately does **not** claim the
bottleneck objective as the point of novelty, only as the environment the novel churn operates in.

**Claim 1 (independent — the closed loop).** A computer-implemented method of forming and
maintaining persistent social groups across a plurality of scheduled in-person rounds, comprising:

1. storing, for each ordered pair of users in a candidate pool, a directed compatibility score
   generated by a scoring engine from user persona data;
2. partitioning at least a subset of the candidate pool into one or more groups of a fixed size by
   executing a search that maximizes, for each group, the **minimum** pairwise compatibility score
   among all member pairs of that group, wherein a pair for which no compatibility score is stored
   is assigned a sentinel value that causes any group containing that pair to be rejected by the
   search;
3. scheduling an in-person event for a first group and recording, for each member, an attendance
   record and a commitment record;
4. receiving, after the event, a plurality of **directed** post-event ratings, each rating
   identifying a rater who attended the event, a ratee who is a member of the group distinct from
   the rater, and a binary continuation indication;
5. determining that a member is a removal candidate when (a) the number of distinct attending
   raters who rated that member is at least a quorum threshold, and (b) the proportion of those
   raters whose continuation indication is affirmative is below an approval threshold, **wherein
   ratings issued by the member, ratings from non-attending members, and ratings from non-members
   are excluded from both the numerator and the denominator**;
6. removing the removal candidate from the group, conditioned on the resulting group size
   remaining at or above a viability floor, and writing an audit record comprising the rater count
   and affirmative count that caused the removal;
7. selecting from an ordered queue of waiting users a replacement user that maximizes the minimum
   pairwise compatibility score of the group as reconstituted with that replacement user, and
   adding the replacement user to the group; and
8. for a subsequent round, re-executing the partitioning of step 2 over an updated candidate pool
   using **adjusted** compatibility scores in which a continuity value is added to the score of
   each pair of users that were co-members of a common group in the immediately preceding round,
   wherein the search of step 2 is evaluated against the adjusted scores and the minimum pairwise
   score recorded for each resulting group is computed from the **unadjusted** scores.

*Rationale:* the novelty load is carried by the conjunction of steps 4–8 — directed rating
quorum with explicit exclusion rules (5), quorum-gated removal with a viability floor (6),
backfill against the same bottleneck objective (7), and the dual-score continuity re-assembly (8).
Step 2's sentinel limitation is deliberately included as an eligibility anchor, giving Alice step
two something concrete: constraint enforcement and objective evaluation collapse into one
comparison rather than requiring a tuned penalty term.

**Claim 2 (dependent — bounded continuity).** The method of claim 1, wherein the continuity value
is bounded such that adding it to any stored compatibility score cannot raise that score above a
score that would satisfy the search absent the continuity value for a pair assigned the sentinel
value, whereby the continuity value re-orders substantially tied candidate groupings without
overriding a hard exclusion.

*Rationale:* claims the `-inf + bonus == -inf` property and the "small enough to break ties, not
to rescue a bad pair" design intent as a structural limitation rather than a parameter choice.

**Claim 3 (dependent — lexicographic acceptance).** The method of claim 1, wherein the search of
step 2 comprises: constructing an initial partition by selecting a highest-scoring available pair
as a seed and iteratively adding the candidate that leaves the resulting group minimum highest;
and thereafter, for a bounded number of iterations, evaluating candidate member exchanges between
two groups and between a group and an unassigned user, and applying the single exchange that
produces the greatest improvement in a vector of per-group minimum scores sorted in ascending
order and compared lexicographically, terminating when no exchange improves that vector.

*Rationale:* the lexicographic sorted-minima acceptance criterion is the one solver detail not
found verbatim in the MDGP art — it keeps the search productive on non-worst groups once the
global minimum plateaus. Weak as novelty, useful as an Alice step-two "improvement in the
functioning of the computer" hook.

**Claim 4 (dependent — commitment-conditioned removal).** The method of claim 1, further
comprising removing from the group, independently of the ratings of step 4, a member whose
commitment record indicates a paid deposit and whose attendance record indicates non-attendance,
and not removing on that basis a member whose commitment record indicates no deposit.

*Rationale:* the asymmetry (deposit-paid no-shows are removed, non-committed absentees are not)
is a concrete rule tying an economic commitment artifact to membership churn. Narrow, but real,
and it distinguishes from any pure ratings-based system.

**Claim 5 (dependent — uncontaminated evaluation arm).** The method of claim 1, further
comprising assigning a first proportion of groups by the search of step 2 and a second proportion
by random assignment, recording an assembly-method identifier for each group, and computing a
retention metric separately over groups bearing each assembly-method identifier, wherein the
minimum pairwise scores recorded in step 8 from unadjusted scores are used in said computation.

*Rationale:* ties the §6 exploration arm to the dual-score bookkeeping of claim 1 step 8, and is
genuinely non-obvious as a *system* limitation — the reason to persist raw rather than adjusted
scores is precisely to keep the continuity bias out of the causal estimate. No reference found
combines a randomized-assignment evaluation arm with a biased optimizer in this way.

**Deliberately not claimed:** the max-min objective as such; greedy-plus-swap local search as
such; LLM-based compatibility scoring (separate memo); group size six.

---

## Design-Around Notes

### How a competitor designs around us

Assume our narrowest claim issues. These are the cheap routes around it, listed because they also
measure how little the claim is worth.

1. **Change the aggregator.** Use mean-with-a-floor ("mean pair score, subject to every pair
   exceeding t") instead of a pure minimum. Behaviourally near-identical at our scale; outside a
   claim reciting "maximizes the minimum."
2. **Change the continuity encoding.** Instead of adding a bonus to prior co-member pairs, seed
   the next round's search *from* the surviving core as a fixed sub-group and grow around it. Same
   persistence outcome, no additive term, no "adjusted scores" step.
3. **Change the churn trigger.** Use a continuous reputation scalar per user updated from ratings,
   and remove when it crosses a threshold — no rater quorum, no per-group approval proportion.
   This also walks straight into US 9,519,682's territory rather than ours.
4. **Move the decision to a human or an LLM.** Surface flagged members to a human moderator or a
   model that decides removals. A claim reciting a deterministic quorum-and-threshold test does
   not read on a judgment call.
5. **Undirected ratings.** Collect a single group-level satisfaction score rather than directed
   per-member ratings. Loses signal, but sidesteps the "directed rating, self-ratings excluded"
   limitations entirely.

### How we design around others

1. **US 11,983,225 (anonymous connections at live events, multi-round, negative prior-round
   term).** Our freedom to operate rests on: persistent identity across separate scheduled events
   rather than rounds within one live event; no anonymity requirement; post-event directed ratings
   rather than interest updates; membership churn with backfill, which the reference has no
   analogue for. **Action: keep the product framing explicitly cross-event and identity-persistent
   in all public material.** Do not describe Regulars as "rounds at an event." Also keep the
   continuity term additive-per-dyad-under-a-min rather than a group-edge-sum adjustment, since
   the reference's term is a sum.
2. **US 9,549,070 (Avaya, compatibility-based team formation).** Contact-center-scoped and
   one-shot. Low risk. Our bottleneck objective and repeated rounds are outside it. No action.
3. **eHarmony US 10,320,734 / 10,868,789 (social matching with feedback loop).** Dyadic and
   social-graph-gated. Low risk as long as we do not ship a feature where a user's *friends* rate
   their prospective matches. **Action: avoid building friend-endorsement of match candidates.**
4. **US 9,519,682 (trustworthiness-driven regrouping).** Low risk while our removal rule is a
   per-group rating quorum. **Action: if we ever add a global per-user reputation scalar that
   drives regrouping across all groups, re-run FTO on this reference first.**
5. **Match Group's matching portfolio.** Not directly implicated because we claim group
   partitioning. **Action: keep any dyadic recommendation surface (e.g. "you should meet X
   one-on-one") out of scope, or diligence it separately.** Match Group litigates.
6. **Timeleft / 222 / Lunchclub public disclosures.** No patents found, so no infringement risk —
   but their public product behaviour is §102 art against us. **Action: anything we file must
   claim the churn loop and continuity bookkeeping specifically, never "match strangers into
   dinner groups and rate each other," which Timeleft has been publicly doing since 2023.**

### Practical guardrails for the codebase

- Keep the raw-vs-adjusted score separation in `optimizer.assemble_groups` intact. It is
  simultaneously the evaluation-integrity mechanism (§6), the audit-honesty mechanism (§10), and
  the most distinctive claim limitation. It is the single line most likely to be "simplified away"
  by a future refactor that does not know it is load-bearing.
- Keep the `-inf` sentinel encoding rather than migrating to an explicit constraint-checking pass.
  Two mechanisms where one suffices is worse engineering *and* loses the eligibility anchor.
- Keep `MIN_RATERS`, `APPROVAL_THRESHOLD` and `DEFAULT_CONTINUITY_BONUS` as named constants in one
  place. If they are ever tuned per-cohort, that tuning is the trade secret and should not be
  logged anywhere user-visible or committed to a public repo.
- The audit records already capture `continuity_bonus`, `min_score`, `assembly_method` and
  `swaps_applied` per group. That is contemporaneous evidence of conception and reduction to
  practice. Do not prune the audit table's schema.

---

## Sources

- [Learning to Partition using Score Based Compatibilities (AAMAS 2017, arXiv:1703.07807)](https://arxiv.org/abs/1703.07807)
- [Maximally Diverse Grouping Problem — GRAFO](https://grafo.etsii.urjc.es/maximally-diverse-grouping/)
- [Solving the maximally diverse grouping problem by skewed general VNS](https://www.sciencedirect.com/science/article/abs/pii/S0020025514010263)
- [An Integer Programming Formulation for the MDGP (arXiv:2410.08482)](https://arxiv.org/html/2410.08482)
- [Maximizing Satisfaction in Group Formation — Munson & Hutchins](https://www.smunson.com/portfolio/projects/groupform/paper.pdf)
- [Egalitarian item allocation](https://en.wikipedia.org/wiki/Egalitarian_item_allocation)
- [Algorithmic Fair Allocation of Indivisible Items: A Survey (SIGecom Exchanges)](https://www.sigecom.org/exchanges/volume_20/1/AZIZ.pdf)
- [Welfare-Centric Clustering (arXiv:2508.10345)](https://arxiv.org/pdf/2508.10345)
- [Fair Clustering Under a Bounded Cost (arXiv:2106.07239)](https://arxiv.org/pdf/2106.07239)
- [A Team-Formation Algorithm for Faultline Minimization (arXiv:1811.05015)](https://arxiv.org/pdf/1811.05015)
- [Team Formation amidst Conflicts (arXiv:2403.00859)](https://arxiv.org/pdf/2403.00859)
- [Computational Team Assembly with Fairness Constraints (arXiv:2306.07023)](https://arxiv.org/pdf/2306.07023)
- [US 11,983,225 B1 — Real-time confidential electronic matching at live events](https://patents.google.com/patent/US11983225)
- [US 9,549,070 B2 — Compatibility-based team formation (Avaya)](https://patents.google.com/patent/US9549070B2/en)
- [US 10,320,734 B1 — Social matching (eHarmony)](https://patents.google.com/patent/US10320734B1/en)
- [US 2010/0131502 A1 — Cohort group generation and automatic updating](https://patents.google.com/patent/US20100131502A1/en)
- [US 9,519,682 B1 — User trustworthiness](https://patents.google.com/patent/US9519682B1/en)
- [US 11,816,746 B2 — Dynamic dining party group management (Rockspoon)](https://patents.google.com/patent/US11816746B2/en)
- [US 7,677,970 B2 — Social matching of game players on-line](https://patents.google.com/patent/US7677970)
- [Catalyzing collaborations: Prescribed interactions at conferences (arXiv:2112.08468)](https://arxiv.org/pdf/2112.08468)
- [Inclusive Study Group Formation At Scale (arXiv:2202.07439)](https://arxiv.org/pdf/2202.07439)
- [Genetic algorithm for group formation in collaborative learning](https://www.sciencedirect.com/science/article/abs/pii/S0360131511002284)
- [Federal Circuit Says User-Matching Patent Claims are Abstract (Trinity Info Media v. Covalent) — IPWatchdog](https://ipwatchdog.com/2023/07/16/federal-circuit-says-user-matching-patent-claims-abstract-precedential-eligibility-decision/)
- [The Current State of 35 U.S.C. §101 and Software Patents — Volpe Koenig](https://www.vklaw.com/ImagineThatIPLawBlog/the-current-state-of-35-u-s-c-101-and-software-patents)
- [Patents Assigned to Atlassian — Justia](https://patents.justia.com/assignee/atlassian-pty-ltd)
- [Timeleft algorithm description (company blog)](https://timeleft.com/blog/2023/11/10/timeleft-algorithm-the-maestro-of-your-dinners/)
- [Group Peer Assessment with Team Member Evaluation — Peerceptiv](https://peerceptiv.zendesk.com/hc/en-us/articles/6854114171539-Group-Peer-Assessment-with-Team-Member-Evaluation)

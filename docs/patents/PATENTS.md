# Patent & IP Strategy — Living Document

**Read this file before any patent, prior-art, provisional-filing, or "is this patentable"
conversation about Regulars.** It is the single index over everything researched so far. It is
meant to be updated, not just read — see **Maintenance** at the bottom before you close out any
session that touches IP strategy.

Last updated: 2026-09-14. Ten mechanisms researched, zero filings made, zero counsel engaged.

---

## 0. The one-paragraph version

Ten candidate mechanisms in the Regulars architecture were researched for patentability, each by
an independent research pass doing real prior-art search (Google Patents, USPTO, arXiv, Google
Scholar). **None clears the bar for a strong standalone patent.** Every mechanism has either
direct prior art (a granted patent doing substantially the same thing) or is a predictable
combination of well-known techniques that fails §103 obviousness, and most face serious §101
eligibility exposure after *Recentive Analytics v. Fox Corp.* (Fed. Cir., April 2025), which held
that applying established methods to a new data environment without improving the underlying
technology is patent-ineligible. **The consistent recommendation across all ten reports is trade
secret plus defensive publication, not a patent program.** One mechanism (anti-sycophancy variance
gate, #5) has a narrow, conditionally-defensible claim if drafted precisely. The real moat was
never going to be a patent — it's the compounding corpus of (immutable persona version →
prediction → directed real-world outcome) triples, which cannot be bought, scraped, or backfilled
by a competitor who starts later. That's protected by running dinners, not by filing.

---

## 1. Method

Each of the 10 reports below was produced by an agent instructed to: (1) read the specific
mechanism's spec in `docs/technical-architecture.md` and its implementation in `regulars/`, (2)
run real web searches against Google Patents, USPTO full-text search, arXiv, and Google Scholar —
not recall from training data, (3) identify the closest prior art with citations, (4) draft
claim sketches only where genuinely defensible, and (5) give an honest verdict including "don't
file" when that was the finding. Full reports are linked per mechanism below; this file is the
index and the synthesis.

**None of this is legal advice.** These are engineering-literate research passes, not a
freedom-to-operate opinion from patent counsel. Several reports explicitly flag searches that
could not complete (e.g. Justia's Match Group assignee listing returned HTTP 403) — treat "no
prior art found" as "no prior art found by this search," not as a clearance.

---

## 2. Mechanism-by-mechanism summary

| # | Mechanism | Spec ref | Verdict | Full report |
|---|---|---|---|---|
| 1 | Permanent randomized exploration arm (de-bias production labels + retention A/B) | §6 Problem 1 | **Weak.** Anticipated by US 12,513,350 (PayPal), US 11,204,973 (Microsoft), US 12,038,986 (Huawei). Closest product-level art: Rockspoon US 11,562,303 (ML dinner-group assembly + ratings-driven retraining) — this is an FTO concern, not just a novelty one. | [01-exploration-arm.md](01-exploration-arm.md) |
| 2 | Min-pair-score group optimizer with continuity bonus + prune/backfill | §7 | **Weak-to-moderate, narrow only.** Max-min partitioning is public-domain (AAMAS 2017, MDGP literature). US 11,983,225 already claims multi-round group re-formation with a prior-co-membership term — ours is a sign flip of a disclosed term. | [02-group-optimizer.md](02-group-optimizer.md) |
| 3 | Shared feature-extraction pipeline for sim and real transcripts | §0, §4 | **Weak.** Schatzmann, Georgila & Young (SIGdial 2005) already do "same metrics on real and simulated dialogue corpora to localize divergence" — 21 years old. Two 2025 papers (arXiv:2512.11844, arXiv:2510.00414) are much closer on application than Park et al. | [03-shared-feature-pipeline.md](03-shared-feature-pipeline.md) |
| 4 | SRM variance decomposition applied to a judge's own predictions | §5 (v2) | **Weak — do not file standalone.** Joel/Eastwick/Finkel (2017) — cited in our own architecture doc — already SRM-decomposes speed-dating outcomes and predicts each component. Independently anticipated in drug–target interaction-effect literature. | [04-srm-judge-diagnostic.md](04-srm-judge-diagnostic.md) |
| 5 | Anti-sycophancy sim harness (temp + friction injection + banned-phrase gate + variance-degeneracy gate) | §3 | **Narrow, conditional yes — the strongest candidate of the ten.** Every individual element (temperature, anti-agreeableness prompting, difficulty grading, banned-phrase screening) is separately anticipated. The variance-degeneracy gate used as a precondition for downstream compute/deployment decisions has no found prior art. **Urgent: our own architecture doc already publishes this in patent-defeating detail — file a provisional before further public disclosure if this is pursued.** | [05-anti-sycophancy-harness.md](05-anti-sycophancy-harness.md) |
| 6 | Immutable append-only persona versioning + FK-pinned prediction reproducibility | §1 rule 1, §2 | **Do not file — weakest of the ten.** Three references (Bottomline, IBM, Capital One, 2019–2021) each anticipate half the combination; the rest is textbook event sourcing. Fails §103 before §101 is even reached. | [06-immutable-persona-versioning.md](06-immutable-persona-versioning.md) |
| 7 | Two-stage cost cascade (cheap prefilter → frontier deep sim) + persona-prefix caching | §8 | **Not independently patentable.** Viola-Jones (2001) coarse-to-fine cascade; candidate-gen-then-rerank is patented by Google/Meta/LinkedIn already; prompt caching is Anthropic's own feature; hash-keyed LLM caching is patented (US 12,259,913) and commoditized. | [07-two-stage-cost-cascade.md](07-two-stage-cost-cascade.md) |
| 8 | Plan-object-as-source-of-truth + payment-gated lock/removal/backfill/redistribution | §9 | **Narrowly patentable at best.** "Chat as a view over structured state" and "minimal tool surface" are each ~20 years anticipated. The specific closed loop — deposit-window expiry → automatic removal → backfill → forfeited-stake redistribution — sits in genuinely unoccupied space (closest analogues, WO2020112614 and EP2724298, are both **ceased**). Worth a provisional if this mechanic proves out in production. | [08-plan-object-protocol.md](08-plan-object-protocol.md) |
| 9 | Per-user, per-hour-of-day normalized response latency as a directed compatibility feature | §4 (real chat only) | **Weak-to-moderate.** Match Group's own subsidiary (Plentyoffish, US 10,540,607) already claims response-time-as-matching-input; per-user/per-hour baselining is separately anticipated (AT&T US 8,831,571, IBM/Snap US 8,862,519). **A live dating-app portfolio may already read on this — get a real FTO search before any filing.** | [09-normalized-latency-signal.md](09-normalized-latency-signal.md) |
| 10 | Sim-to-reality eval pipeline (isotonic calibration, person-grouped CV, cluster-robust SEs, SRM-aware exploration-arm-only reporting) | §5, §6 | **Not patentable, not worth filing.** Every stage is textbook (Zadrozny & Elkan 2002; Park & Marcotte 2012; Liang & Zeger 1986; Kenny 1984; Li et al. 2011). Rigor executed correctly is a real competitive advantage — it's just not an invention. | [10-sim-to-reality-eval-pipeline.md](10-sim-to-reality-eval-pipeline.md) |

---

## 3. Cross-cutting findings that matter more than any single verdict

- **`Recentive Analytics v. Fox Corp.` (Fed. Cir., April 2025) is the recurring kill shot.** It
  holds that applying established ML/statistical methods to a new data environment, without
  improving the underlying methods themselves, is patent-ineligible under §101. Six of the ten
  reports cite it as dispositive or near-dispositive. Any future mechanism proposed for patenting
  needs to clear this bar specifically: the claimed improvement has to be to *how the computer
  works*, not to *what data it's applied to*.
- **Two papers already anticipate the product's central framing at the application level:**
  "Love First, Know Later" (arXiv:2512.11844, Dec 2025) simulates LLM personas conversing and
  scores romantic compatibility, validated on speed-dating data. CogniPair (arXiv:2506.03543) does
  LLM digital-twin conversations for dating/hiring compatibility prediction. Neither does the
  sim↔real shared-feature-pipeline calibration loop or the exploration-arm labeling design — but
  "simulate two personas, score compatibility" as a category is no longer novel on its own, and any
  pitch or filing should not lean on that framing as the differentiator.
- **A live FTO risk was flagged, not resolved, for the dating-app latency patent (#9) and the
  dinner-group-assembly patent (Rockspoon, relevant to #1 and #2).** Neither report could complete
  a full assignee-portfolio search (Justia returned HTTP 403 for Match Group; no Bumble Inc.
  listing was reachable). **Before any public fundraising claim about the matching algorithm, or
  before filing anything, this needs a real FTO search from actual patent counsel** — this
  document is not that.
- **Every report converges on the same two-part recommendation:** (1) trade secret on the tuned
  constants, weights, and thresholds (these are what actually drive prediction quality and are
  never observable from the product), (2) defensive publication of the general methodology
  (blocks a competitor from patenting it against us later, and doubles as the credibility signal
  §6/§7 of the architecture doc wants in front of an investor's technical diligence partner).
- **The one item with real urgency:** mechanism #5's variance-degeneracy gate is the strongest
  candidate found, but the architecture doc already describes it in enough detail to count as a
  public disclosure. If a provisional filing is wanted, it needs to happen before any further
  public repo activity, blog post, or pitch deck reproduces that description (US grace period is
  12 months from first disclosure; most foreign jurisdictions have none).

---

## 4. Recommended posture (synthesized, not a substitute for counsel)

1. **No patent program at this stage.** Ten mechanisms researched, zero recommended for immediate
   non-provisional filing.
2. **If founder wants IP-optics for a fundraise:** one narrow provisional on mechanism #5
   (anti-sycophancy variance gate) and, if the payment/removal loop (#8) proves out in production,
   a second provisional on that closed loop specifically. Budget both as inexpensive
   marketing/optionality spend, not as enforceable exclusion rights.
3. **Defensive publication** (a dated engineering blog post, or arXiv preprint) of: the shared
   sim/real feature pipeline (#3), the SRM-on-model-predictions diagnostic (#4), the latency
   normalization formula (#9), and the eval pipeline design (#10). This costs nothing, blocks
   competitors from patenting these against Regulars, and is a stronger diligence signal than a
   patent would be for a matching-algorithm company.
4. **Trade secret, never published, never filed:** `exploration_rate`, prefilter thresholds, judge
   weights (`DEFAULT_WEIGHTS` and anything tuned from real data), scenario/seed design constants,
   the specific top-K cascade value and cache-hit economics, `MIN_HOUR_SAMPLES`/`MAX_GAP_MS` and
   other latency-normalization constants, and any measured component-variance shares from real
   dinners.
5. **Before any filing decision:** get a real FTO search from patent counsel on the Match Group
   (Plentyoffish) and Rockspoon patent families specifically — both were flagged as close enough
   to matter and neither search completed.
6. **The actual moat stays what the architecture doc always said it was:** the labeled-outcomes
   corpus. Nothing here changes that; if anything, ten independent research passes concluding "the
   algorithms are not patentable" is further evidence that the corpus, not the code, is the asset.

---

## 5. Maintenance — read this before you close out a session

This file must be kept current. Whenever you (a future Claude session, an agent, or a human)
touch anything related to IP strategy — a new mechanism is added to the architecture, new prior
art is found, a filing decision is made, a mechanism's implementation changes enough to affect its
novelty argument, or an FTO search actually completes — **update this file in the same session**,
following this pattern:

- **New mechanism researched:** add a row to the table in §2, write the full report as
  `docs/patents/NN-short-name.md` following the existing reports' structure (Mechanism Summary,
  Prior Art Found, Novelty Analysis, Patentability Verdict, Draft Claims, Design-Around Notes),
  and add anything cross-cutting to §3.
- **New prior art found for an existing mechanism** (including a competitor's later filing, a new
  paper, or a completed FTO search): update that mechanism's row in §2, append a dated note to the
  relevant `NN-*.md` report under a `## Updates` section (create one if it doesn't exist) rather
  than silently rewriting the original verdict — keep the history visible.
- **A filing decision is made** (provisional filed, defensive publication posted, decision to
  abandon a mechanism): record it in §4 and update the "Last updated" line at the top of this
  file with the date and a one-line description of what changed.
- **Do not delete or soften a "don't file" verdict without new evidence.** These were reached by
  real prior-art search; if a later session disagrees, it should say why and cite what changed,
  not just assert a rosier read.
- Keep §0 (the one-paragraph version) in sync with the table — it's what gets read first.

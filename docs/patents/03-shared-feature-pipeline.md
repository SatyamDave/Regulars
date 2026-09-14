# 03 — Shared Feature Pipeline (Sim/Real Reconciliation)

Patent and prior-art analysis. Prepared 2026-09-14. Analyst working from
`docs/technical-architecture.md` §0 and §4, `regulars/core/transcript.py`,
`regulars/core/features.py`, `regulars/features/extract.py`, `regulars/features/real.py`.

This is a research memo, not legal advice. Nothing here is a freedom-to-operate opinion.

---

## Mechanism Summary

Regulars predicts whether two people will get along at a dinner. It produces that prediction
twice, from two different kinds of evidence, and the architectural claim under examination is
that both paths run **the same extraction code over the same normalized data type and emit the
same named feature keys**.

The concrete shape in this repo:

**One transcript type.** `regulars/core/transcript.py` defines `Transcript` with a `source`
field typed `Literal["sim", "real"]` and a `Turn` carrying `text`, optional `sent_at_ms`,
optional `reply_to`, and optional `tapback_target`. An LLM persona-vs-persona simulation and an
exported iMessage group chat are both reduced to this one shape. Simulated turns carry
`sent_at_ms=None`; real turns carry wall-clock timestamps plus the iMessage-only reply and
reaction pointers. `Transcript.restrict(a, b)` (`transcript.py:60`) projects a six-person group
chat down to the dyadic sub-transcript between two of its members, re-indexing `reply_to` and
`tapback_target` and dropping reactions that point outside the dyad — so a group chat yields
dyad-level objects structurally identical to what the simulator emits for a pair.

**One shared extractor.** `regulars/features/extract.py::extract_pair_features` takes a
`Transcript`, two participant ids, and an `ExtractorConfig` naming which implementation answers
each of three judgment calls (question classification, topic segmentation, disclosure depth
scoring — heuristic or LLM-backed). It returns a `(a→b, b→a)` pair of `PairFeatures`. Eight
features are symmetric and carried identically in both directions; `follow_up_rate` and
`disclosure_reciprocity` are directed. The function branches on nothing about `source`. It runs
identically on a simulated dyad and on a real dyad.

**One feature namespace, enforced at runtime.** `regulars/core/features.py` declares
`SHARED_FEATURES` (ten keys: `follow_up_rate`, `lsm_mean`, `lsm_slope`, `topic_survival`,
`topic_turns_p90`, `disclosure_reciprocity`, `disclosure_asymmetry`, `repair_rate`,
`repair_latency_turns`, `question_gini`) and `REAL_ONLY_FEATURES` (four keys:
`latency_norm_median`, `latency_norm_fast_share`, `reply_weight`, `tapback_rate`), with
`FEATURE_NAMES = SHARED_FEATURES + REAL_ONLY_FEATURES`. The shared extractor asserts set
equality before returning:

```python
if set(feats) != set(SHARED_FEATURES):
    raise RuntimeError(f"extractor emitted wrong key set: {sorted(feats)}")
```

(`extract.py:124`). The real-only extractor asserts the dual, a subset relation:

```python
if not set(features) <= set(REAL_ONLY_FEATURES):
    raise RuntimeError(f"real extractor emitted non-real keys: {sorted(features)}")
```

(`real.py:238`). The two assertions together are the load-bearing design constraint: neither
path can silently drift into a private feature namespace, which is exactly the failure that
would make a per-feature sim-vs-real comparison meaningless without anyone noticing.

**Real-only features as an explicit strict superset, absent rather than zero.**
`extract_real_features` gates on `transcript.source != "real" or not transcript.has_timing()`
and returns `({}, {})` (`real.py:227`). The docstring in `core/features.py` states the rule
directly: real-only features are "omitted, never zero-filled, for sim transcripts." A zero-fill
would be a lie the judge could not distinguish from a genuine measurement of zero. The three
real-only signal families are the ones a text simulation structurally cannot produce:

- **Response latency**, normalized per §4 as `log(raw_ms) − median(log latency of this user,
  this hour-of-day)`, with gaps over four hours dropped as sleep or work rather than
  disengagement (`MAX_GAP_MS`, `real.py:33`), and per-user baselines fit over the user's whole
  message history rather than the conversation being scored (`fit_baseline`, `real.py:127`).
  Grounded in Templeton et al. (PNAS 2022).
- **Reply graph**, a directed weighted graph built from explicit `reply_to` where present and
  most-recent-other-speaker otherwise, with `reply_weight(a,b)` normalized as a share of a's
  total out-weight.
- **Tapback rate**, directed reactions by a on b's turns per spoken turn of b.

**The loop this enables.** Because both scores live in the same named coordinates, a divergence
between the sim score and the reality score can be localized to a feature — "the simulator
overproduces follow-up questions and underproduces repair after flat moments" — rather than
observed only as an aggregate prediction error. The real-only superset then supplies signal for
calibration and retraining that the simulator never saw. Downstream the repo carries the
consumers of that loop: `regulars/judge/calibration.py` (isotonic mapping of raw judge scores to
observed `P(would_see_again)`), `regulars/judge/srm.py` (Kenny's Social Relations Model,
decomposing round-robin dinner ratings into actor, partner and relationship variance),
`regulars/judge/diagnostics.py` (variance floor and known-pairs AUC gate against the
"sim collapses to agreeable" failure), and `regulars/eval/` (cluster-bootstrapped Spearman,
mixed models with rater/ratee/group random intercepts).

The claimed inventive concept, stated as narrowly as it can be: *a shared-namespace,
single-code-path dyadic feature extractor spanning LLM-simulated and real consumer-messaging
conversations, with an explicitly declared real-only feature superset that is omitted rather
than imputed, used to compute per-feature simulation-to-reality divergence and to drive
continuous recalibration of a two-party social-matching predictor.*

---

## Prior Art Found

### A. Generative-agent simulation and its validation

**Park et al., "Generative Agent Simulations of 1,000 People," arXiv:2411.10109 (Nov 2024).**
The reference point named in the brief, and correctly characterized. Two-hour qualitative
interviews with 1,052 participants are used to condition agent architectures; the agents
reproduce the participants' General Social Survey answers at 0.85 normalized accuracy, where
1.0 means "as well as the participant reproduces their own answers two weeks later," with
further validation on Big Five and economic games. What this establishes and what it does not:

- It establishes, as prior art, that validating an LLM persona simulation against the real
  behavior of the same individual is known, published, and highly visible.
- It does **not** disclose a shared feature extractor. The validation target is a survey
  instrument — a fixed questionnaire — not a conversation. There is no dialogue transcript on
  the real side to extract from, hence no shared extractor, no shared feature namespace, and no
  per-feature divergence diagnostic.
- It is a one-shot validation study, not an architecture for continuous calibration in a
  production system.
- It is single-agent. Nothing in it addresses a dyad, a two-party interaction, or matching.

It remains the strongest §103 obviousness backdrop for "simulate a person, then check the
simulation against that person's real behavior." It is not an anticipation of the shared
extractor.

**Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," UIST 2023 /
arXiv:2304.03442.** The memory-reflection-planning agent architecture. Evaluation is human
believability rating and controlled ablation. No real-world behavioral counterpart is measured
with the same instrument.

**"Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World
Engines," arXiv:2512.11844 (Dec 2025).** This is the closest *paper* in subject matter and it
deserves careful handling, because on a skim it looks like the same system. It proposes
simulating a persona-vs-persona interaction first and assessing compatibility from the
simulation, formalizes compatibility assessment as reward modeling over observed matching
outcomes, proves a convergence result (as the LLM policy better approximates human behavior the
induced matching converges to optimal stable matching), and validates on speed-dating data for
initial chemistry and divorce data for long-term stability.

Where it differs, verified against the paper text:

- The signals extracted from the simulation are three **LLM ratings** — two participant
  self-assessments and one observer rating over "conversational flow, mutual engagement, and
  value alignment" — combined by a simple linear model trained on observed matching decisions.
  There is no named, reproducible behavioral feature vector.
- The extraction is **never run on real human conversation**. The speed-dating dataset is static
  profiles plus binary match decisions; the divorce dataset is 54 Likert items. Neither contains
  dialogue. There is therefore no real-side extraction to reconcile with.
- No side-by-side feature comparison, no per-feature divergence, no real-only privileged signal
  set, no continuous recalibration loop.

It anticipates "simulate the interaction, then score it, and check the score against real
outcomes." It does not anticipate feature-level reconciliation.

**RELATE-Sim, arXiv:2510.00414 (Oct 2025).** Two persona-aligned LLM agents play out
turning-point scenes (exclusivity talks, conflict-and-repair, relocations) under a Scene Master
that infers interpretable state changes and an auditable commitment estimate per scene.
Validated on 71 couples with two-year follow-ups; simulation-aware prediction beats a
personas-only baseline, and the paper surfaces markers such as "repair attempts acknowledged."
Note the overlap in vocabulary: *repair* is a feature in both systems. But RELATE-Sim's markers
are produced only inside the simulation and validated against a longitudinal **outcome**, not
against the same markers extracted from the couples' real conversations. Outcome-level
validation again, not feature-level.

### B. The most damaging prior art: user-simulator evaluation

**Schatzmann, Georgila & Young, "Quantitative Evaluation of User Simulation Techniques for
Spoken Dialogue Systems," SIGdial 2005, pp. 45–54.** This is the single most dangerous
reference for the core idea and it should be treated as effectively anticipatory of the
*concept*. The paper proposes a systematic set of quantitative measures — grouped as high-level
dialogue features, dialogue style and cooperativeness, and task success and efficiency —
computed **identically over a real human-computer dialogue corpus (DARPA Communicator) and over
a corpus generated by the user simulator**, precisely in order to determine where the simulator
diverges from real users. The paper's own conclusion — that simple statistical metrics still
suffice to tell synthetic from real dialogue — is a direct statement of the diagnostic use case.

Restated in the language of the brief: "run the same feature extraction on simulated and real
dialogue corpora and compare feature by feature to find where the simulation diverges from
reality" was published in 2005. That is twenty-one years of prior art on the mechanism's central
idea. Any claim drafted at that level of abstraction is dead.

**Pietquin & Hastie, "A survey on metrics for the evaluation of user simulations," Knowledge
Engineering Review.** A survey of this exact family of metrics, which by its existence
establishes the approach as a recognized field rather than an insight.

**Modern LLM-era continuations of the same practice**, all of which compute matched statistics
over real and simulated dialogue: "A Survey on LLM-based Conversational User Simulation"
(arXiv:2604.24977); RecUserSim (arXiv:2507.22897), which evaluates simulator realism via user
action distributions and distributional diversity; "Real or Robotic? Assessing Whether LLMs
Accurately Simulate Qualities of Human Responses in Dialogue" (arXiv:2409.08330), which measures
similarity between original human and simulated responses across lexical, syntactic, semantic
and style domains with correlation of class frequencies for distribution metrics; clem:todd
(arXiv:2505.05445); DIAL (arXiv:2512.20773). KL divergence between real and simulated
conversation feature distributions is routine in this literature.

### C. Sim-to-real transfer, domain adaptation, digital twins

**Shared feature extractor across a synthetic source domain and a real target domain is the
textbook structure of deep domain adaptation.** The lineage — domain-adversarial training,
discrepancy-loss alignment (MMD, CORAL), domain-invariant representation learning — is
summarized in "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey"
(arXiv:2009.13303). simDP (Sensors, 2026, doi:10.3390/s26134079) explicitly reduces the
sim-to-real gap by *aligning the action and observation spaces shared between simulated and
physical robots* and encoding real observations into the latent representation learned in
simulation. That is the robotics-domain analogue of "reduce both sources to one shared
representation so the gap is measurable," and it is directly on point conceptually.

**Reality-gap measurement at the metric level is established practice in autonomous driving.**
"A Multi-Modality Evaluation of the Reality Gap in Autonomous Driving Systems"
(arXiv:2509.22379) assesses the gap along actuation, perception, and behavioral-fidelity
dimensions across Software-in-the-Loop, Vehicle-in-the-Loop, Mixed-Reality, and real-world
testing. "Closing Sim2Real Gaps" (Sensors, doi:10.3390/s26041338) defines reality-alignment via
normalized cross-correlation between matched signals (poses, steering, velocity, detections)
computed in both domains and iterates Digital Twin → Parallel Execution → Real-World with
parameter fine-tuning to align behavior. See also the radar sim-to-real gap measurement work
(arXiv:2106.08372) and neural reality-gap quality metrics (arXiv:2404.18577). Per-signal
divergence driving simulator parameter correction is, in the autonomy world, standard
engineering.

**Digital-twin calibration with an explicit model-discrepancy term** descends from Kennedy &
O'Hagan-style Bayesian calibration: sparse real-world observations estimate uncertain simulator
parameters while a separate discrepancy function absorbs structural model error. Recent
instances: Arcones et al., "Model Bias Identification for Bayesian Calibration of Stochastic
Digital Twins of Bridges" (Applied Stochastic Models in Business and Industry, 2025); continuous
model calibration for smart-building digital twins (Applied Energy, 2024,
doi:10.1016/j.apenergy.2024.124053 lineage); "Bridging the Reality Gap in Digital Twins with
Context-Aware, Physics-Guided Deep Learning" (arXiv:2505.11847). The pattern "simulate, measure
the same quantities in reality, estimate the discrepancy, feed it back" is a mature field with
its own textbooks.

### D. Asymmetric information between training and inference

**Vapnik & Vashist, "A new learning paradigm: Learning using privileged information," Neural
Networks 22(5-6), 2009; SVM+.** LUPI is defined by exactly the asymmetry this mechanism claims:
additional information available about training examples that is *not* available at test time.
The real-only feature superset — signals the real side produces and the simulated side
structurally cannot — is a LUPI-shaped arrangement, and the framing is nineteen years old. See
also "Learning Using Privileged Information: SVM+ and Weighted SVM" (arXiv:1306.3161), "Learning
with Privileged Information for Multi-Label Classification" (arXiv:1703.09911), and "Privileged
Information for Modeling Affect In The Wild" (arXiv:2107.10552). The robotics analogue —
privileged-teacher / student distillation where the teacher sees simulator state the student
cannot observe — is the same idea under a different name.

This matters for claim drafting: "features present on one side and absent on the other, used to
train or correct the model on the side that lacks them" is a named, well-known learning
paradigm, not a novel architectural insight.

### E. Patents

Closest by mechanism:

| Patent | Assignee / date | Relevance |
|---|---|---|
| **US20200320347A1** — *System and method for domain adaptation using synthetic data* | Synthesis AI, priority 2019-04-02 | Claim 1: identify a source domain with labeled synthetic data, a target domain with unlabeled real data, **"identifying at least one common feature between the source domain and the target domain,"** and applying a model trained on source to label target. This is the closest patent to the shared-extractor concept. Domain is images/computer vision; nothing about conversation, dialogue, or behavioral features. |
| **US11640519B2** — *Domain adaptation in neural networks using cross-domain batch normalization* | — | Shared-network alignment of source and target feature statistics. |
| **US11537901** — *Unsupervised domain adaptation with mixup training* | — | Same family. |
| **US20240371393A1** — *Determining conversation analysis indicators for a multiparty conversation* | BetterUp Inc, filed 2024-07-16 | Extracts engagement, active listening, interruption counts, question identification, sentiment, topics, pauses and pacing from **real** multiparty conversations (coach/mentee videoconference) using ML. Closest patent on the real-side conversational feature-extraction half. Verified: no simulated or synthetic conversation anywhere in the disclosure. |
| **US11790251B1** — *Semantically detecting synthetic driven conversations in electronic media messages* | — | Feature extraction plus semi-supervised ML to distinguish synthetic from real conversation patterns. Defensively important: it establishes that computing features over both synthetic and real conversations, and comparing them, is known art — for a different purpose (fraud detection) but with the same computational shape. |

Closest by application (dating / matching from messaging behavior):

| Patent | Relevance |
|---|---|
| **US10540607** — *Apparatus, method and article to effect electronic message reply rate matching in a network environment* | Response-predictive models generating compatibility scores from user profiles, messaging activity and messaging behavior — i.e. the real-only reply-behavior half of the mechanism, in the dating domain, already patented. |
| **US7251640** and **US7711667** — *Method and system for measuring interest levels of digital messages* | Predicting communication behavior, replies and feedback; matching people by mutual propensity to communicate. Reply-behavior-as-compatibility-signal, early priority. |
| **US10984488** — *Predictive compatibility matching platform* | General compatibility prediction from analyzed characteristics. |
| **US20120005204A1** — *System for determining and optimizing for relevance in match-making systems* | Classifies matches by profile inspection and message exchange; treats "sent but no reply" as non-relevant. |
| **US11260306** — *Matchmaking for online gaming with simulated players* (issued 2022-03-01) | Simulated players in a matchmaking context. **Not verified** — patents.justia.com returned HTTP 403 on fetch. Flag for manual review before any filing; the title alone is close enough to matter. |

Closest by loop structure (real logs → simulator → system update):

| Patent | Relevance |
|---|---|
| **US10713317B2** / **US20180218080A1** — *Conversational agent for search* | An RL agent trained against a **user model generated from real search-engine session logs**. Real behavior → simulator → trained system is disclosed. |
| **US10395646B2** — *Two-stage training of a spoken dialogue system* | Stage one on known fixed real dialogues, stage two on interaction with a user simulator. Explicit real/simulated hybrid training. |
| **US12282744** — *Statistical language models for simulating communication sessions* | Simulated agents generating communications with outcome-likelihood scoring calibrated to real customer propensities. |
| **US20240289863A1** — *Adaptive AI-driven conversational agents* | Continuous generation of new training data from anonymized real conversational history paired with outcomes. The "closing the loop with production data" element. |

### F. The behavioral features themselves

All six shared features come from published, unpatented (and in most cases decades-old) social
science, and none is independently claimable:

- **Follow-up question rate** — Huang, Yeomans, Brooks, Minson & Gino, JPSP 2017 (trained NLP
  follow-up detector applied to speed dating; independently audited 2025).
- **Linguistic style matching** — Ireland & Pennebaker (2010); Ireland et al., *Psychological
  Science* (2011) on speed dating and relationship stability; Niederhoffer & Pennebaker (2002)
  on the contrary finding that LSM indexes engagement rather than rapport. The nine function-word
  categories and the `1 − |pa−pb|/(pa+pb+ε)` formula are published verbatim. Also relevant:
  Danescu-Niculescu-Mizil et al., "Chameleons in imagined conversations" (arXiv:1106.3077).
- **Topic survival, disclosure reciprocity, repair rate, question balance** — standard
  constructs from conversation analysis, self-disclosure research (Altman & Taylor; Sprecher),
  and repair sequences in CA (Schegloff).
- **Response latency** — Templeton et al., PNAS 2022.
- **Social Relations Model** — Kenny; `TripleR`. The SRM decomposition of round-robin ratings
  into actor, partner and relationship variance is published statistical method.

A search for "linguistic style matching" patents surfaced only the academic literature. LSM does
not appear to be patented, which is good news for freedom to operate and bad news for novelty:
it is public domain method.

---

## Novelty Analysis

Splitting the mechanism into elements and grading each honestly.

**Element 1 — Same extraction code over simulated and real conversation. NOT NOVEL.**
Schatzmann 2005 does this explicitly for spoken dialogue. The entire user-simulator evaluation
literature does this. Domain adaptation does this as its defining move, and US20200320347A1
claims it for images. There is no daylight here.

**Element 2 — Per-feature divergence used diagnostically rather than outcome-level error only.
NOT NOVEL.** This is the stated purpose of Schatzmann's three metric groups, and it is
day-to-day practice in autonomous-driving reality-gap measurement (arXiv:2509.22379;
doi:10.3390/s26041338). The brief's framing — "diagnose WHERE a simulation diverges from
reality" — is a restatement of the 2005 paper's motivation.

**Element 3 — Real-only features as an explicit strict superset. NOT NOVEL AS A PARADIGM, but
the specific set is unclaimed.** LUPI names and formalizes the asymmetry. However: no prior art
found combines *these* signals — per-user, per-hour-of-day log-normalized response latency with
a four-hour gap cutoff; a directed reply graph derived from explicit reply pointers with a
most-recent-other-speaker fallback; and tapback rate as a directed affect measure — as a
declared privileged set for correcting a *conversation simulator* in a *social matching*
context. US10540607 covers reply-rate-as-compatibility-signal in dating but not the simulator
correction use, and not the normalization.

**Element 4 — Runtime enforcement of feature-namespace identity (set-equality assertion on the
shared path, subset assertion on the real-only path, omission rather than zero-fill).
NOVEL-ISH, and almost certainly not patentable.** No prior art found states this as a design
constraint, and it is genuinely the thing that makes the loop survive contact with a codebase
over time: two pipelines that are "supposed to" compute the same features will drift, and the
drift is silent. But: an assertion that two dictionaries have the same key set is, to a patent
examiner, an obvious programming practice. Choosing to omit a missing feature rather than impute
zero is standard missing-data hygiene. This element is real engineering value and weak patent
value — the classic shape of a trade secret rather than a claim.

**Element 5 — Reduction of a multi-party real group chat to dyadic sub-transcripts so that
group-chat evidence and pairwise simulation produce commensurable objects. MODESTLY NOVEL.**
`Transcript.restrict` re-indexes reply and reaction pointers and discards reactions crossing the
dyad boundary. Prior art on dyad extraction from group conversation exists in the social-network
and conversation-disentanglement literature, but coupling it to a pairwise simulator so the two
sides land in the same coordinate system is not something the search surfaced. This is the more
defensible structural element than Element 4.

**Element 6 — SRM decomposition as a gating diagnostic on what the simulator actually predicts.
THE MOST NOVEL ELEMENT FOUND, and it is not what the brief asked about.** The question
"is my model predicting the partner effect (some people are simply likeable) rather than the
relationship effect (this specific pairing)?" — answered by fitting Kenny's SRM to round-robin
dinner ratings and checking which variance component the predictor tracks — did not appear in
any simulation-validation paper, matching patent, or user-simulator evaluation found. The Joel,
Eastwick & Finkel machine-learning-on-speed-dating result (actor and partner effects predictable,
relationship variance not) is the well-known finding; using SRM as a *deployment gate on a
simulator* is not. If anything in this system is worth a filing, it is plausibly this, or the
combination of this with the shared extractor.

**The combination.** The honest assessment: each element is known, and the combination is the
kind a competent engineer would assemble when told "build a persona simulator for matching and
make sure it's actually right." That is the §103 problem in one sentence. The combination does
have one feature examiners sometimes credit — it is *specific*: this exact feature namespace,
this exact real-only superset from a consumer messaging client, this exact normalization, this
loop closing on this exact outcome label. Specificity buys narrow claims. Narrow claims are
cheap to design around.

---

## Patentability Verdict

**Weak. This is better protected as an architecture pattern held as trade secret plus a
defensive publication, than pursued as a patent — with one possible narrow exception around the
consumer-messaging privileged-signal loop and the SRM deployment gate.**

### The case against filing

1. **§103 obviousness, and it is severe.** Schatzmann, Georgila & Young (SIGdial 2005) discloses
   computing a matched set of quantitative measures over a real dialogue corpus and a simulated
   dialogue corpus in order to localize where the simulator diverges. Park et al. (2024)
   discloses validating an LLM persona simulation against the same person's real behavior.
   Vapnik & Vashist (2009) discloses the training/test information asymmetry. US20200320347A1
   claims a shared feature across a synthetic source and a real target domain. An examiner has
   a clean obviousness rejection available combining any two of these, and the motivation to
   combine is stated in the references themselves. The brief's own instinct — that "extract the
   same features from simulated and real data" looks like obvious application of standard feature
   engineering — is correct, and it is worse than that: it is obvious application of a *named
   prior-art methodology* with its own survey paper.

2. **§101 / *Alice* risk.** Claims that collect conversation data, compute scores from it, and
   compare the scores are squarely in the territory of *Electric Power Group v. Alstom* (Fed.
   Cir. 2016) — collecting information from disparate sources, analyzing it, and presenting
   results is an abstract idea, and generic computer implementation does not save it. The
   features are mental-process-adjacent (a human coder can count follow-up questions). Without a
   concrete technical-improvement hook, step two fails.

3. **Detection and enforcement are near-impossible.** The entire mechanism is server-side. A
   competitor running the identical architecture emits no observable artifact: their product
   surface is "we suggest people you'll get along with." There is no way to read the feature
   namespace of a competitor's extractor from the outside, and therefore no way to build an
   infringement case without discovery you will never get to. A patent you cannot detect
   infringement of is a document, not an asset.

4. **Cost and disclosure.** Filing publishes the architecture — including the real-only feature
   list and the latency normalization, which are the parts a competitor would most benefit from
   knowing. Trading a genuinely useful secret for an unenforceable claim of dubious validity is
   a bad trade at this stage.

### The case for filing

1. **The specific combination is narrow enough to survive if drafted tightly.** A claim reciting
   *a consumer-messaging-derived privileged feature set (per-user per-hour-of-day log-normalized
   response latency with a maximum-gap exclusion, a directed reply graph with explicit-pointer
   precedence, and a directed reaction rate), computed only for timestamped real transcripts,
   used to compute per-feature divergence against a simulated transcript's shared features and
   to modify the simulation's generation parameters* has no single anticipating reference found.
   Schatzmann has no privileged set and no generation-parameter update; the domain-adaptation
   patents have no conversation; the dating patents have no simulator.

2. **The generation-parameter-modification step is the §101 lifeline.** If the claim ends at
   "compare and display," it dies under *Electric Power Group*. If it ends at "and thereby modify
   the parameters governing subsequent simulated-transcript generation, such that subsequent
   simulations exhibit reduced divergence" — an actuation, a changed technical process, closer to
   *McRO v. Bandai Namco* (specific rules improving a computer-driven process) — it has a
   fighting chance. Draft toward the loop, never toward the measurement.

3. **Defensive value is real even if offensive value is not.** A well-known, well-funded
   competitor entering this category (Hinge, Match, Bumble all have substantial portfolios; see
   US10540607, US10984488) will file in this space. An issued narrow patent, or even a published
   application, is a cross-licensing chip and a bar to someone else claiming the same ground and
   pointing it at Regulars.

4. **Investor and acquirer optics.** A filed application is a line item in a data room. This is
   not a legal argument and should not drive claim drafting, but it is a real reason companies
   at this stage file.

### Recommended posture

- **Do not file broadly on "shared extractor for sim and real."** It will be rejected and the
  rejection will be on the record, citing Schatzmann, where it will sit permanently as an
  admission about the state of the art in this exact area.
- **Consider one narrow provisional** covering Claims 1, 3 and 5 below — the closed loop with
  the specific consumer-messaging privileged set, the group-chat-to-dyad reduction, and the SRM
  deployment gate. Cheap, buys twelve months, and forces the claim-scope conversation with
  counsel while the evaluation data is still being collected.
- **Publish defensively** on the shared-namespace discipline itself (Element 4). It is
  unpatentable, it is useful, and publishing it prevents anyone else from claiming it while
  costing nothing that a competitor could not reconstruct from first principles anyway.
- **Hold as trade secret**: the feature weights `w` in the v1 judge, the classifier prompts, the
  thresholds (`MAX_GAP_MS`, `MIN_HOUR_SAMPLES`, `VARIANCE_FLOOR`, `AUC_BROKEN`), the persona
  extraction schema, and the empirical divergence profile — which features the simulator actually
  gets wrong and by how much. That last item is the genuinely valuable thing the loop produces,
  it is not derivable from the architecture, and it compounds with every dinner. Protect it with
  access control and employment agreements rather than with a claim set.

---

## Draft Claims

Sketches for discussion with counsel, not filing-ready language. Ordered from the broadest that
is still worth writing down to the narrowest and most defensible.

### Claim 1 (independent, method) — the closed loop with a privileged signal set

> A computer-implemented method for calibrating a conversational simulation of a two-party
> interaction, comprising:
>
> (a) generating, by one or more language models conditioned on a first behavioral persona
> specification and a second behavioral persona specification, a simulated transcript of a
> conversation between a first simulated participant and a second simulated participant, the
> simulated transcript being represented in a normalized transcript data structure comprising an
> ordered plurality of turn records;
>
> (b) receiving a real transcript of a conversation between a first human participant and a
> second human participant from a messaging client, the real transcript being represented in the
> same normalized transcript data structure, wherein turn records of the real transcript further
> comprise a send timestamp and, where present, a reply pointer to a preceding turn record and a
> reaction pointer to a preceding turn record;
>
> (c) applying a single feature-extraction procedure to the simulated transcript and to the real
> transcript, the single feature-extraction procedure emitting, for each transcript and for each
> ordered pair of participants, a feature vector whose keys consist of a predetermined shared
> feature namespace, the procedure selecting no branch on whether the transcript is simulated or
> real;
>
> (d) computing, only for the real transcript, a supplemental feature vector whose keys are drawn
> from a predetermined real-only feature namespace disjoint from the shared feature namespace,
> the real-only feature namespace comprising at least a normalized response-latency feature, a
> directed reply-graph weight, and a directed reaction rate, and omitting the real-only feature
> namespace entirely from the feature vector of the simulated transcript without imputation;
>
> (e) computing a per-key divergence between the shared-namespace feature vector of the simulated
> transcript and the shared-namespace feature vector of the real transcript;
>
> (f) modifying, in dependence on the per-key divergence and on the supplemental feature vector,
> at least one generation parameter of the language-model conditioning used in step (a); and
>
> (g) generating at least one subsequent simulated transcript using the modified generation
> parameter, whereby the per-key divergence for the subsequent simulated transcript is reduced.

*Drafting notes.* Step (f) plus step (g) are the §101 survival kit — the claim must actuate, not
merely measure. Step (c)'s "selecting no branch" is an attempt to claim the architectural
constraint as a positive limitation; counsel should assess whether that is a structural
limitation or an unclaimable statement of programmer intent. Step (d)'s "without imputation" is
a negative limitation and will draw scrutiny; it is worth keeping because it is the element with
no analogue in the domain-adaptation art, where zero-filling and masking are both routine.

### Claim 2 (dependent on 1) — namespace-identity enforcement

> The method of claim 1, wherein the single feature-extraction procedure verifies, prior to
> returning each feature vector, that the set of keys of the feature vector is equal to the shared
> feature namespace and raises an error condition otherwise; and wherein the computing of the
> supplemental feature vector verifies that the set of its keys is a subset of the real-only
> feature namespace and raises an error condition otherwise; such that a divergence between the
> feature namespace of the simulated path and the feature namespace of the real path is detected
> at extraction time rather than at comparison time.

*Drafting notes.* This is Element 4 and it is the weakest claim in the set — an examiner will
characterize it as a data-validation step, which is routine. Included because the final clause
articulates *why* it matters (the failure is otherwise silent), and because if it survives it is
the hardest element for a competitor to design around while still having a working loop.

### Claim 3 (independent, method) — group chat to commensurable dyads

> A computer-implemented method comprising:
>
> (a) receiving a multi-party message log from a messaging client, the log comprising message
> records among three or more participants, each message record comprising a sender identifier, a
> send timestamp, and optionally a reply pointer or a reaction pointer to a preceding message
> record;
>
> (b) for a selected ordered pair of the three or more participants, forming a dyadic
> sub-transcript by retaining only message records whose sender identifier is one of the selected
> pair, re-indexing each retained reply pointer and reaction pointer to the position of its target
> within the dyadic sub-transcript, and discarding any reaction record whose target lies outside
> the dyadic sub-transcript;
>
> (c) computing, from the dyadic sub-transcript, a first feature vector over a shared feature
> namespace using a feature-extraction procedure that is also applied to simulated two-party
> transcripts generated by a language model;
>
> (d) computing, from the dyadic sub-transcript, a second feature vector over a real-only feature
> namespace, wherein computing a normalized response-latency feature comprises: identifying, for
> each message record of the first participant, an answered message record as the target of an
> explicit reply pointer where present and otherwise as the immediately preceding message record
> of a different sender; computing an elapsed interval; excluding intervals exceeding a
> predetermined maximum-gap threshold; taking the logarithm of each remaining interval; and
> subtracting a baseline comprising a median logarithm of the first participant's response
> intervals for the corresponding hour of day computed over a message history other than the
> dyadic sub-transcript being scored; and
>
> (e) producing a compatibility prediction for the selected ordered pair from the first feature
> vector and the second feature vector.

*Drafting notes.* This is the most concrete claim in the set and the most likely to issue,
because step (d) recites a specific, non-obvious normalization procedure rather than a result.
"Baseline computed over a message history other than the transcript being scored" is the
non-obvious bit — it is what makes the latency feature a within-person deviation rather than a
personality measurement, and it distinguishes the claim from US10540607's raw reply-rate
modeling. Note that step (b)'s re-indexing is machine-specific enough to read as data-structure
transformation rather than abstract analysis, which also helps under §101.

### Claim 4 (independent, system) — the diagnostic surface

> A system comprising one or more processors and memory storing instructions that, when executed,
> cause the system to: maintain a feature registry declaring a shared feature namespace and a
> real-only feature namespace; generate simulated dyadic transcripts from pairs of persona
> specifications; ingest real dyadic transcripts derived from consented message-log exports;
> compute, via a common extraction path, shared-namespace feature vectors for both; compute
> real-only-namespace feature vectors for the real transcripts only; store, for each of a
> plurality of participant pairs, a simulation-derived prediction and a reality-derived
> prediction; and generate a diagnostic output identifying, for each key of the shared feature
> namespace, a signed divergence between the simulation-derived and reality-derived values
> aggregated over the plurality of participant pairs, wherein the diagnostic output ranks keys by
> contribution to a difference between the simulation-derived and reality-derived predictions.

*Drafting notes.* Standard system-claim mirror, but note the last clause: ranking features by
*contribution to the prediction difference* rather than by raw divergence is a small step beyond
the Schatzmann-style corpus comparison, because it weights divergence by how much the judge
actually cares. Weak, but it is the one thing in the diagnostic half that the 2005 paper does
not do. Highest §101 risk in the set — it ends at "generate a diagnostic output," which is
exactly the *Electric Power Group* shape. File it only as a fallback behind Claim 1.

### Claim 5 (dependent on 1 or 4) — variance-component gating

> The method of claim 1, further comprising: collecting round-robin outcome ratings from a
> plurality of in-person gatherings, each gathering comprising three or more participants and
> each participant providing a directed rating of each other participant; fitting a social
> relations model decomposing variance in the directed ratings into an actor component, a partner
> component, and a relationship component; determining which of the components the
> simulation-derived prediction covaries with; and suppressing deployment of the
> simulation-derived prediction for group assembly responsive to determining that the
> simulation-derived prediction covaries with the partner component above a threshold and with
> the relationship component below a threshold.

*Drafting notes.* The most interesting and least-anticipated claim found, and the one no prior
art in this search touches. It is also the one that reads least like a software patent and most
like a statistical method, which cuts both ways under §101 — but the concrete actuation
("suppressing deployment ... for group assembly") gives it a technical effect to point at. If
only one thing is filed, there is a real argument it should be this plus Claim 1.

---

## Design-Around Notes

### What a competitor does to avoid each claim

- **Against Claim 1.** Do not modify generation parameters; instead modify only the *judge*
  weights in response to divergence, leaving the simulator untouched. This avoids limitations (f)
  and (g) entirely while capturing most of the practical benefit. This is a large hole and there
  may be no way to close it without a claim so broad it reads on Schatzmann. Alternatively: zero-
  fill the real-only features on the simulated side and add a binary "is-real" indicator feature,
  which is the standard missing-data treatment and avoids the "without imputation" limitation
  while being statistically near-equivalent.

- **Against Claim 2.** Use a single extraction function with a `source` parameter and compute
  real-only features inside it, guarded by a conditional — one code path, no two namespaces to
  reconcile, no assertion needed. Functionally similar, textually outside the claim. Note that
  this design is arguably *worse* engineering (the branch is the drift vector the assertion
  exists to catch) but it is trivially available.

- **Against Claim 3.** Normalize latency by a global rather than per-user-per-hour baseline; or
  use a 6-hour or 2-hour gap cutoff; or derive the answered-message target purely from explicit
  reply pointers with no fallback. Each is a parameter change. Numerical limitations are the
  easiest thing in a patent to step around, which argues for reciting the *structure* (a baseline
  computed over history external to the scored transcript) and letting the specific threshold
  live in the specification as an embodiment, not in the claim.

- **Against Claim 5.** Use a mixed model with rater and ratee random intercepts instead of a
  formal SRM fit — statistically overlapping, textually distinct — or gate on a different
  statistic entirely (e.g. residual variance after removing ratee means). The repo already
  contains `regulars/eval/mixed.py` doing exactly the mixed-model variant, which is worth noting:
  our own alternative implementation is evidence that the SRM limitation is substitutable.

### Prior art to cite defensively in any filing

Citing these in an IDS is mandatory once known, and citing them well is also the best available
argument that the applicant understood the field and claimed only past it:

1. **Schatzmann, Georgila & Young, SIGdial 2005** — cite and distinguish explicitly. The
   distinction to argue: their measures are computed over two *corpora* for offline comparison of
   simulator quality, with no privileged-signal set, no per-pair prediction, and no modification
   of the generator. If the claim cannot be distinguished from this reference in one clear
   sentence, the claim is not worth filing.
2. **Pietquin & Hastie survey on user-simulation metrics** — establishes the field; cite to show
   awareness.
3. **Park et al., arXiv:2411.10109** — cite as the highest-profile reference in the space.
   Distinguish on: survey instrument not conversation, single agent not dyad, one-shot validation
   not continuous loop, no shared extractor because there is no real-side transcript.
4. **arXiv:2512.11844 ("Love First, Know Later")** — the closest paper on application.
   Distinguish on: LLM rating scores rather than named behavioral features, and extraction never
   applied to real conversation. This distinction is factual and checkable in their §4; state it
   precisely.
5. **arXiv:2510.00414 (RELATE-Sim)** — distinguish on outcome-level validation with no real-side
   extraction. Note the shared "repair" vocabulary and address it head-on rather than hoping an
   examiner misses it.
6. **US20200320347A1 (Synthesis AI)** — the closest patent on shared features across synthetic
   and real domains. Distinguish on domain (images vs. dialogue) and on the absence of any
   privileged real-only set or generator-update loop.
7. **US20240371393A1 (BetterUp)** — closest on real-side conversational feature extraction.
   Distinguish on the total absence of simulation.
8. **US10540607, US7251640, US7711667, US10984488, US20120005204A1** — the dating/matching
   messaging-behavior family. Distinguish on the absence of simulation and of any shared
   namespace; but be aware these establish that reply behavior as a compatibility signal is long
   known, which narrows what the real-only set can claim.
9. **US11790251B1** — synthetic-vs-real conversation feature comparison. Cite it; it is the least
   obvious reference to find and the most damaging one to be seen to have missed.
10. **Vapnik & Vashist (2009), LUPI / SVM+** — cite for the training/test information asymmetry.
    Distinguish on: LUPI's privileged information is unavailable at *inference*, whereas here the
    asymmetry is between two *data sources* both used at inference, with the privileged side
    correcting the generator of the unprivileged side. That is a real structural difference and
    it is the best single novelty argument available for Element 3.
11. **US11260306 (matchmaking with simulated players)** — unverified; fetch and read the claims
    before any filing. If it claims simulated agents standing in for players in a matchmaking
    decision, it is closer than anything else in the patent set and must be distinguished
    carefully.

### Freedom-to-operate flags for the product, independent of filing

- **US10540607** (reply-rate compatibility scoring in a network messaging environment) is the FTO
  item to have counsel read against `regulars/features/real.py`. The `reply_weight` and latency
  features in a dating-adjacent product sit near its subject matter.
- **US7251640 / US7711667** (measuring interest levels of digital messages; matching by mutual
  propensity to communicate) have early priority and broad-sounding titles. Check expiry status —
  both are likely expired or near expiry given their numbering, which would remove the concern
  entirely and is a five-minute check worth doing before anything else in this list.
- The psycholinguistic features themselves (LSM, follow-up questions, latency) appear to be
  unencumbered published method. No LSM patent surfaced.

---

## Sources

- [Generative Agent Simulations of 1,000 People (Park et al., arXiv:2411.10109)](https://arxiv.org/pdf/2411.10109)
- [Generative Agents: Interactive Simulacra of Human Behavior (UIST 2023)](https://dl.acm.org/doi/10.1145/3586183.3606763)
- [Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World Engines (arXiv:2512.11844)](https://arxiv.org/abs/2512.11844)
- [RELATE-Sim: Leveraging Turning Point Theory and LLM Agents (arXiv:2510.00414)](https://arxiv.org/pdf/2510.00414)
- [Quantitative Evaluation of User Simulation Techniques for Spoken Dialogue Systems (Schatzmann et al., SIGdial 2005)](https://aclanthology.org/2005.sigdial-1.6/)
- [A survey on metrics for the evaluation of user simulations (Pietquin & Hastie)](https://homepages.inf.ed.ac.uk/hhastie2/pubs/pietquin_hastie_surveyusersimKER.pdf)
- [A Survey on LLM-based Conversational User Simulation (arXiv:2604.24977)](https://arxiv.org/html/2604.24977v1)
- [RecUserSim (arXiv:2507.22897)](https://arxiv.org/html/2507.22897v1)
- [Real or Robotic? Assessing Whether LLMs Accurately Simulate Qualities of Human Responses in Dialogue (arXiv:2409.08330)](https://arxiv.org/pdf/2409.08330)
- [Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey (arXiv:2009.13303)](https://arxiv.org/pdf/2009.13303/1000)
- [simDP: Sim-to-Real Transfer with Shared Action Spaces (Sensors)](https://doi.org/10.3390/s26134079)
- [A Multi-Modality Evaluation of the Reality Gap in Autonomous Driving Systems (arXiv:2509.22379)](https://arxiv.org/abs/2509.22379)
- [Closing Sim2Real Gaps: A Versatile Development and Validation Platform for Autonomous Driving Stacks (Sensors)](https://doi.org/10.3390/s26041338)
- [A Multi-Layered Approach for Measuring the Simulation-to-Reality Gap of Radar Perception (arXiv:2106.08372)](https://arxiv.org/pdf/2106.08372)
- [Bridging the Reality Gap in Digital Twins with Context-Aware, Physics-Guided Deep Learning (arXiv:2505.11847)](https://arxiv.org/html/2505.11847)
- [Model Bias Identification for Bayesian Calibration of Stochastic Digital Twins of Bridges (Wiley)](https://onlinelibrary.wiley.com/doi/full/10.1002/asmb.2897)
- [Learning Using Privileged Information: SVM+ and Weighted SVM (arXiv:1306.3161)](https://arxiv.org/abs/1306.3161)
- [A new learning paradigm: Learning using privileged information (Vapnik & Vashist, Neural Networks)](https://www.sciencedirect.com/science/article/abs/pii/S0893608009001130)
- [US20200320347A1 — System and method for domain adaptation using synthetic data (Synthesis AI)](https://patents.google.com/patent/US20200320347A1/en)
- [US11640519B2 — Domain adaptation using cross-domain batch normalization](https://patents.google.com/patent/US11640519B2/en)
- [US20240371393A1 — Determining conversation analysis indicators for a multiparty conversation (BetterUp)](https://patents.google.com/patent/US20240371393A1/en)
- [US11790251B1 — Semantically detecting synthetic driven conversations in electronic media messages](https://patents.google.com/patent/US11790251)
- [US10540607 — Electronic message reply rate matching](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10540607)
- [US7251640 / US7711667 — Measuring interest levels of digital messages](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7711667)
- [US10984488 — Predictive compatibility matching platform](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10984488)
- [US20120005204A1 — Determining and optimizing relevance in match-making systems](https://patents.google.com/patent/US20120005204)
- [US11260306 — Matchmaking for online gaming with simulated players (unverified)](https://patents.justia.com/patent/11260306)
- [US10713317B2 — Conversational agent for search](https://patents.google.com/patent/US10713317B2/en)
- [US10395646B2 — Two-stage training of a spoken dialogue system](https://patents.google.com/patent/US10395646B2/fr)
- [US12282744 — Statistical language models for simulating communication sessions](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12282744)
- [US20240289863A1 — Adaptive AI-driven conversational agents](https://patents.google.com/patent/US20240289863A1/en)
- [Language Style Matching Predicts Relationship Initiation and Stability (Ireland et al., Psych Science 2011)](https://sites.socsci.uci.edu/~lpearl/courses/readings/IrelandEtAl2011_RelationshipPrediction.pdf)
- [Linguistic Style Matching in Social Interaction (Niederhoffer & Pennebaker 2002)](https://journals.sagepub.com/doi/10.1177/026192702237953)
- [Chameleons in imagined conversations (Danescu-Niculescu-Mizil et al., arXiv:1106.3077)](https://arxiv.org/pdf/1106.3077)

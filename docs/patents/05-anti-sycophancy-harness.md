# 05 — Anti-Sycophancy Dyadic Simulation Harness

Prior-art and patentability assessment for the simulation harness described in
`docs/technical-architecture.md` §3 ("Simulation harness", *Anti-sycophancy measures*) and §6
(Problem 4, "the sim collapses to agreeable").

Research date: 2026-09-14. Searches run against Google Patents, USPTO full-text (patft
image-ppubs), arXiv, and general web. Patent-office searching here is keyword-based, not a
professional freedom-to-operate search; a registered practitioner should re-run CPC-class
searches (G06N 3/0475, G06N 5/04, G06F 40/35, G06Q 50/01, H04L 51/02) before any filing
decision.

---

## Mechanism Summary

The harness generates synthetic dyadic (two-person) conversation transcripts by having two LLM
instances, each conditioned on a structured behavioral `PersonaSpec` extracted from a real user,
alternate turns in a scripted social scene. The transcripts feed a feature extractor and a judge
that produce `y_hat`, a predicted compatibility score used to assemble real dinner groups.

The named technical problem is that **two RLHF-trained LLMs role-playing strangers are pleasant
to each other**, so every pair scores highly and the score has no discriminative power. §6 calls
this "the top technical risk". Five coupled countermeasures, as implemented:

1. **Elevated temperature as a variance instrument, not a creativity knob.**
   `SimConfig.temperature = 0.95`, annotated `# §3: 0.9-1.0, never 0.3`
   (`regulars/sim/harness.py`). The stated purpose is preserving between-pair behavioral spread,
   not prose quality.

2. **System prompts that license disengagement and withhold rapport.**
   `enactment_system_prompt` in `regulars/sim/prompts.py` instructs each agent: "You are not
   helpful or pleasant by default"; "When you are bored, disengage: answer short, change nothing,
   let the silence sit, check out"; "Do not ask a question every turn"; "Do not summarize,
   reflect back, or validate"; "Disagreement does not need resolving." Turns are capped at
   `MAX_SENTENCES_PER_TURN = 3` / `max_tokens = 160`, so a short, flat turn is a legal output
   rather than something the model must pad.

3. **Asymmetric friction injection.** For exactly one seed per pair
   (`config.seeds[config.friction_seed_index]`, default seed 0), persona B's *own stated*
   `social.friction_points` are injected into persona **A's** system prompt under a
   `TONIGHT, SPECIFICALLY` heading instructing A to perform those behaviors "without meaning to
   and without announcing it" — and symmetrically A's into B's. Neither agent is told that the
   other has been primed, and the injection is off for the remaining seeds. The transcript records
   `friction_injected` in metadata. This is an *information asymmetry* deliberately constructed
   from first-party user-stated dislikes.

4. **A categorical ban on pro-social framing, enforced as a test.**
   `BANNED_PHRASES` in `regulars/sim/prompts.py` is a 21-entry lower-cased substring list
   ("good conversation", "rapport", "be helpful", "find common ground", "hit it off",
   "connection", "positive", …). `tests/sim/test_prompts.py::test_prompt_has_no_banned_phrases`
   asserts absence across every scenario × injection combination, and adds five literals
   (`EXTRA_BANNED`) held outside the module specifically so the test cannot be satisfied by
   editing the module's own list. The ban is therefore a build-time invariant of the generator,
   not a prompt-authoring convention.

5. **Difficulty-graded scenario set + a variance gate that runs before the mean is read.**
   `regulars/sim/scenarios.py` defines three scenes of distinct social difficulty:
   `cold_open` (baseline), `dead_air` (carries four flat seed turns so the dying moment is *in*
   the transcript), and `mild_friction` (whose `{disagreement}` placeholder is filled per pair by
   `disagreement_for()`, drawing the first concrete item from either persona's `friction_points`
   or `escalates_on`, with a deliberately concrete fallback). `regulars/sim/variance.py`
   `sim_variance_report()` then computes between-pair std over per-pair mean scores and compares
   it both to a floor (`BETWEEN_PAIR_STD_FLOOR = 0.05`) and to the mean within-pair (seed-to-seed)
   std, flagging `degenerate` when between-pair spread fails either test. `regulars/eval/gate.py`
   (`MIN_YHAT_STD`, `_variance_metric`) and `regulars/judge/diagnostics.py`
   (`variance_check`, `broken` = degenerate ∨ AUC < floor) make that check a **precondition**:
   the CLI exits non-zero and the gate verdict degrades when variance is degenerate, so mean
   scores are not consumed downstream.

The five are coupled: (5) is the acceptance test for whether (1)–(4) worked, and (3)'s single
friction seed exists so that at least one transcript per pair is guaranteed to contain contested
material for (5) to find spread in.

---

## Prior Art Found

### A. Prior art on the *problem* (sycophancy) — dense, and all non-patent

| Ref | What it establishes |
|---|---|
| Perez et al., *Discovering Language Model Behaviors with Model-Written Evaluations*, [arXiv:2212.09251](https://arxiv.org/abs/2212.09251) (Anthropic, 2022) | Coined/measured LM "sycophancy" at scale; showed inverse scaling under RLHF — more RLHF, more agreement with the interlocutor's stated view. |
| Sharma et al., *Towards Understanding Sycophancy in Language Models*, [arXiv:2310.13548](https://arxiv.org/abs/2310.13548) (Anthropic, ICLR 2024) | Sycophancy is general across frontier assistants and is *caused* by human preference judgments preferring agreeable, convincingly-written responses. |
| *Sycophancy in Large Language Models: Causes and Mitigations*, [arXiv:2411.15287](https://arxiv.org/abs/2411.15287) (2024 survey) | Catalogues the mitigation families already public: training-data curation, fine-tuning, post-deployment control, **decoding strategies**, and prompt-level "sycophancy priors". |
| *Measuring Sycophancy of Language Models in Multi-turn Dialogues*, [arXiv:2505.23840](https://arxiv.org/abs/2505.23840); PARROT, [arXiv:2511.17220](https://arxiv.org/abs/2511.17220) | Multi-turn and robustness benchmarks for agreement bias. |
| *Not All Flips Are Conformity*, [arXiv:2606.00820](https://arxiv.org/pdf/2606.00820); *When Identity Skews Debate*, [arXiv:2510.07517](https://arxiv.org/pdf/2510.07517) | Agent-to-agent conformity in multi-agent LLM settings and structural interventions against it. |

**Consequence for claim drafting:** the *existence* of the problem and the *general idea* of
prompt-level and decoding-level mitigation are firmly in the public domain as of 2022–2024. No
claim may recite "mitigating sycophancy by prompting the model not to be agreeable" as its point
of novelty.

### B. Closest patent art — LLM persona simulation producing synthetic dialogue

**US12412128B1 — "Systems and methods for generating synthetic data using computer-simulated
personas and machine learning-based persona selection"** (Socialtrait Inc; priority 2024-03-11;
granted 2025). [Google Patents](https://patents.google.com/patent/US12412128B1/en)

This is the single closest reference. Claim 1 is a computer-implemented method for generating
automated discourse data using computer-generated synthetic personas: receive discourse input via
API, embed it, retrieve candidate personas from a vector DB, select by relevance threshold,
compute a **discourse activation value** per persona, instantiate an engagement subset, execute a
simulated discourse session, return synthetic data.

- **Overlaps:** multi-persona LLM simulation; synthetic discourse data as the product;
  per-persona engagement/willingness modeling (conceptually adjacent to "disengage when bored",
  though it is a *selection* gate, not an in-character behavioral instruction).
- **Does not teach:** sampling temperature selected for variance; injection of one persona's
  stated dislikes into a counterpart's context; any prohibited-phrase constraint on the
  enactment prompt; any variance/degeneracy diagnostic gating downstream use; dyadic
  compatibility prediction.

**US20210223854A1 — "Evaluation of digital twins using social automatons"** (AT&T; priority
2020-01-21). [Google Patents](https://patents.google.com/patent/US20210223854A1/en) — inserts
behavior-programmed virtual representations of individuals into a digital twin and reads off how
they interact, explicitly to infer how *real* individuals would interact. Teaches
simulation-of-people-to-predict-real-social-outcomes. Its diversity axis is **demographic**, not
behavioral-friction; no temperature, no injection, no variance gate.

**US20240095491A1** — personalized multimodal response generation through virtual agents
([link](https://patents.google.com/patent/US20240095491A1/en)) — persona-conditioned agent
response generation; background art only.

### C. Closest patent art — persona-parameterized test generation with graded variation

**Cyara Solutions, US11290400B2 / US11595332B2 / US11863507B2 — "System and method for testing of
automated contact center customer response systems"** (Fejes & Monegan; US11863507 filed
2022-11-27, granted 2024-01-02).
[USPTO full text](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11863507)

Retrieves a test case and a **persona** from databases, generates an ideal simulated customer
query, expands it into real-world variants with a **"conversation multiplier"**, then modifies
each variant to reflect persona traits, and analyzes the system-under-test's responses.

- **Overlaps:** deliberately manufacturing *variation* in simulated human utterances via personas
  because a single canonical query under-tests the system. That is structurally the same
  motivation as "one easy scenario produces uniformly pleasant transcripts."
- **Does not teach:** two simulated humans conversing with each other (it is simulated-human vs.
  system-under-test, not dyadic peer-to-peer); friction drawn from a *counterpart's* stated
  dislikes; banned-phrase enforcement; a variance statistic gating whether results are trusted.

**US10395646B2 — two-stage training of a spoken dialogue system**
([link](https://patents.google.com/patent/US10395646B2/en)) and the broad agenda-based /
data-driven **user-simulator** literature establish "simulate the human side of a dialogue to
generate training or evaluation data" as decades-old art.

### D. Closest patent art — prohibited-phrase enforcement over prompts

**US12118471B2 — "Mitigation for prompt injection in A.I. models capable of accepting text
input"** ([link](https://patents.google.com/patent/US12118471B2/en)) — compares tokens of an
incoming command against a **database of objectionable words and tokens**, flags and removes
matches before processing.

This is the most damaging reference for element (4) in isolation. Substring/blocklist screening
of text destined for an LLM is squarely claimed art. The distinctions available to Regulars are
directional and purposive: the blocklist here screens the **system prompt the operator itself
generates** (not untrusted user input), the banned terms are **benign pro-social phrases** rather
than objectionable content, and the purpose is **preserving output variance**, not safety. Those
are real distinctions but they are distinctions in *intent*, which is weak claim material unless
tied to a concrete downstream effect. See also the "Deriving input restrictions for artificial
intelligence agents" family (US12386924, US12547680, US12547681) on auto-deriving agent input
restrictions.

### E. Closest patent art — the downstream application (compatibility scoring)

**US7451161B2** and **US20100185580A1**, "Compatibility scoring of users in a social network"
([US7451161B2](https://patents.google.com/patent/US7451161B2/en)) — compatibility scores from
expressed interests. Establishes the application field long before LLMs; contains nothing about
simulation.

**Commercial / academic systems in the exact application:**
- **CogniPair**, [arXiv:2506.03543](https://arxiv.org/pdf/2506.03543) (2025) — GNWT-based
  multi-agent *digital twins* of users that hold **mock conversations with potential matches**
  and are analyzed for conversational flow and emotional resonance, explicitly for **dating and
  hiring pairing**. This is the closest non-patent art to the whole product concept and is a
  serious §102/§103 reference against any broad "simulate two user-twins to predict
  compatibility" claim.
- **Ditto** (commercial, press-reported) — builds a virtual persona per user and runs "a full
  date simulation … across a thousand hypothetical interactions" before recommending a pairing.
- **Jeter AI** — self-described *patent-pending* AI matchmaking/compatibility platform; contents
  of the application are not public. Flag for monitoring.

### F. Closest art — difficulty-graded and adversarial simulated interlocutors

**Agent-Testing Agent**, [arXiv:2508.17393](https://arxiv.org/pdf/2508.17393) (2025) — synthesizes
persona-driven dialogues **whose difficulty adaptively escalates** based on judge feedback.
Directly anticipates the *concept* of grading simulated social difficulty (and goes further, by
adapting it online). Regulars' three fixed scenes are a simpler, static instance.

Related: *Breakdowns in Conversational AI* ([arXiv:2604.02713](https://arxiv.org/pdf/2604.02713))
on multi-turn simulators that escalate/deflect; *Persona-Grounded Safety Evaluation of AI
Companions* ([arXiv:2605.00227](https://arxiv.org/pdf/2605.00227)).

### G. Closest art — temperature-for-diversity, and variance/diversity gating

- Temperature raising to increase output entropy and sample diversity is textbook, with an
  explicit quality/diversity tradeoff; see e.g. *Comparison of Diverse Decoding Methods*
  ([arXiv:1906.06362](https://arxiv.org/pdf/1906.06362)) and *Improving Diversity in Language
  Models: When Temperature Fails, Change the Loss*
  ([arXiv:2508.09654](https://arxiv.org/pdf/2508.09654)). **Element (1) is unpatentable alone and
  cannot carry a claim.**
- Diversity metrics computed over synthetic corpora and used to accept/reject data: *On the
  Diversity of Synthetic Data and its Impact on Training LLMs*
  ([arXiv:2410.15226](https://arxiv.org/abs/2410.15226), "LLM cluster-agent" metric);
  *Measuring Diversity in Synthetic Datasets*
  ([arXiv:2502.08512](https://arxiv.org/pdf/2502.08512)); a "DiversityGate" that rejects samples
  above a cosine-similarity threshold. These anticipate *diversity-thresholded acceptance of
  synthetic data* at the **sample** level.
- **US20200034665A1**, "Determining validity of machine learning algorithms for datasets"
  ([link](https://patents.google.com/patent/US20200034665A1/en)) — triggers an action when a
  predicted suitability fails a threshold. **US10990901B2**, "Training, validating, and
  monitoring artificial intelligence and machine learning models"
  ([link](https://patents.google.com/patent/US10990901B2/en)) — threshold-flagging of performance
  metrics. These are the generic "metric-below-threshold blocks use" art.

**What I did not find anywhere:** a reference that computes a **between-unit vs. within-unit
variance-decomposition** over *simulated interaction outcome scores*, and uses failure of that
comparison to declare the **simulator itself** non-discriminative and suppress its per-unit means
from a downstream matching decision. The sample-level diversity gates above filter *data items*;
US20200034665A1 validates a *model against a dataset*. Neither validates a *generative simulator's
ability to distinguish between the entities it is simulating*, which is what
`sim_variance_report()` does (between-pair std vs. mean within-pair std, the latter being pure
seed noise). That comparison — signal-between vs. noise-within, an F-ratio in spirit — is the most
defensible single novel element in the stack.

### H. Related but distinct

*Attention Mechanism for LLM-based Agents Dynamic Diffusion under Information Asymmetry*
([arXiv:2502.13160](https://arxiv.org/pdf/2502.13160)) and the hidden-preference negotiation
benchmarks (PrefBench [arXiv:2605.22855](https://arxiv.org/pdf/2605.22855); TERMS-Bench
[arXiv:2605.13909](https://arxiv.org/pdf/2605.13909)) establish **deliberate information asymmetry
between simulated agents** as known art — but as *withholding* a private valuation, never as
*injecting one party's stated aversions into the other party's behavior instructions*. Regulars'
injection is an unusual inversion: the asymmetry is manufactured out of first-party self-report
data to provoke an authentic reaction, not to model a bargaining game.

Park et al., *Generative Agents* ([arXiv:2304.03442](https://arxiv.org/pdf/2304.03442)) and
*Cohesive Conversations* ([arXiv:2407.09897](https://arxiv.org/pdf/2407.09897)) are the baseline
believable-agent art.

---

## Novelty Analysis

Element by element, against the art above:

| Element | Novel alone? | Closest art |
|---|---|---|
| (1) temperature 0.9–1.0 for behavioral variance | **No.** Textbook. | arXiv:1906.06362, arXiv:2508.09654 |
| (2) "disengage when bored / not helpful by default" system prompt | **No.** Ordinary prompt engineering; the survey art already names prompt-level sycophancy priors. | arXiv:2411.15287; general practice |
| (3) asymmetric injection of counterpart's stated friction points | **Arguably yes, narrowly.** Information asymmetry between simulated agents is known; sourcing the injected content from the *counterpart's own self-reported aversions* and instructing the agent to enact them unknowingly is not something I found. | arXiv:2502.13160; PrefBench |
| (4) banned-phrase list over the generated system prompt, test-enforced | **No as a mechanism; possibly yes as applied.** Blocklist screening of LLM-bound text is claimed art; applying it to *the operator's own prompt*, against *benign* phrases, as a build-time invariant, is unusual. | US12118471B2; US12386924 family |
| (5a) difficulty-graded scenario set | **No.** Anticipated, and exceeded, by adaptive-difficulty persona testing. | arXiv:2508.17393 |
| (5b) between-pair vs. within-pair variance gate on simulator trust | **Yes, most likely the strongest element.** I found no reference applying a between-unit/within-unit spread comparison to decide whether a *conversation simulator* discriminates, as a precondition on consuming its scores. | arXiv:2410.15226 (sample-level); US20200034665A1 (model-level) |

**The combination.** The defensible story is not "we prompt the model not to be nice." It is a
closed control loop over a generative simulator:

> deliberately widen the sampling distribution → deliberately remove the pro-social attractor
> from the prompt (and *enforce* its removal mechanically) → deliberately inject, asymmetrically
> and for a designated subset of runs, contested material derived from the counterpart's own
> stated aversions → run a graded scenario ladder → then **measure whether any of that produced
> between-subject spread exceeding within-subject noise, and refuse to use the means if it did
> not.**

Each of (1)–(4) is a knob known individually. What is not in the art is the ordered pairing of
those knobs with (5b) as their *acceptance criterion* — i.e. treating anti-sycophancy
interventions as a controlled experiment whose validity is verified by a variance decomposition
before any product decision (group assembly) consumes the output. §6's phrasing, "watch variance
before mean," is the inventive concept, and `regulars/eval/gate.py` and
`regulars/judge/diagnostics.py` show it is actually wired as a hard gate, not advice.

**Counter-argument a skilled examiner will make** (and it is a decent one): each element solves a
problem in the direction any person of ordinary skill would go — if outputs are too uniform, raise
temperature (known), remove the instruction causing uniformity (obvious once diagnosed), add
conflict (obvious), and check whether the fix worked with a standard dispersion statistic
(routine). KSR-style, the combination may be characterized as the predictable aggregation of known
techniques toward a known problem. The rebuttal is that the *specific ordering* — gate first,
means second, with an explicit between-vs-within comparison rather than a bare std floor, and with
the friction injection confined to one seed precisely so that the contrast between injected and
non-injected seeds is available — is a design choice with a non-obvious rationale, plus the
unexpected-results argument that a single easy scenario yields degenerate output at all.

**§101 eligibility is the bigger risk than §102/§103.** Under *Alice*, "simulate two people
talking, score how it went, check the scores have spread, then match people" reads as an abstract
idea (organizing human social activity + mathematical dispersion calculation) on generic
computers. Survival depends on reciting a concrete technical improvement to the operation of the
computing system. The best available hooks, all present in this codebase:

- the **two-stage compute router** (`regulars/sim/two_stage.py`): a cheap single-seed pass over
  all candidates, then the full scenario × seed grid on a frontier model for the top-k only, with
  `single_shot()` deliberately choosing the *friction* seed as the one cheap run. That is a
  concrete reduction in inference cost gated by simulation quality.
- the **prefix-stable prompt + `SimCacheKey`** design (`regulars/sim/cache.py`,
  `persona_hash()`, and `test_prompt_is_deterministic`): the enactment prompt is a pure function
  of (spec, scenario, counterpart, injection) precisely so it can be prompt-cached and
  content-addressed, which is a technical implementation detail, not an abstract idea.
- the variance gate as a mechanism that **prevents expenditure of downstream compute and
  prevents deployment** of a non-discriminative model.

Claims that recite the gate as controlling *compute allocation and model promotion* are materially
stronger under §101 than claims that recite it as controlling *a matchmaking recommendation*.

---

## Patentability Verdict

**Narrow, conditional yes — a defensible but tightly-scoped patent is available on the ordered
combination, with the variance-decomposition gate as the required point of novelty; any claim
whose weight rests on temperature, anti-agreeableness prompting, or difficulty-graded scenarios
will not survive prosecution.**

Assessment by scope:

| Scope | Verdict |
|---|---|
| "Mitigate LLM sycophancy in simulation by prompting and temperature" | **Unpatentable.** Anticipated and obvious; arXiv:2411.15287 alone. |
| "Simulate two persona agents to predict interpersonal compatibility" | **Unpatentable.** CogniPair (arXiv:2506.03543), Ditto, US20210223854A1, US12412128B1. |
| "Diversity-thresholded acceptance of synthetic dialogue data" | **Unpatentable.** arXiv:2410.15226, arXiv:2502.08512, DiversityGate, US20200034665A1. |
| Full ordered combination (1)+(3)+(4)+(5a)+(5b), gate as precondition, within-vs-between decomposition | **Plausibly allowable**, moderate confidence. Expect one or two §103 rejections combining US12412128B1 + the Cyara persona-variation family + a diversity-metric reference, plus a §101 rejection. |
| Same, further limited to gating **two-stage cheap/frontier compute allocation** and model promotion | **Best odds.** Strongest §101 posture; narrowest commercial coverage. |

Honest commercial read: the resulting claim would be narrow enough that a competitor could design
around it by changing the gate statistic (use AUC on a labeled dev set, or a permutation test, or
per-pair rank stability instead of between/within std) or by moving the friction into the scenario
text rather than the counterpart's prompt. **The primary value of filing is defensive and
narrative** — priority date, an asset to point at, and freedom from a later filer — not exclusion.
If budget forces a choice among the Regulars mechanisms, weigh this one against the persona
extraction and judge/SRM mechanisms before committing; this one's independent claim is the most
easily circumvented of the set. A provisional is cheap and preserves the option; file the
provisional now with the full implementation detail (including the specific thresholds, the
one-seed injection design, and the two-stage router), and decide on the non-provisional after the
§6 diagnostics have produced real data — an actual demonstration that the ungated harness goes
degenerate and the gated one does not is the single best piece of evidence for non-obviousness and
should be captured in the specification as an experimental example.

---

## Draft Claims

Drafting notes: `PersonaSpec` fields are recited functionally ("first-party self-reported
aversion data") to avoid being limited to this schema. The variance gate appears in every
independent claim because it is the point of novelty; do not draft an independent claim without
it.

### Claim 1 (independent — method, core combination)

A computer-implemented method for generating discriminative synthetic dyadic conversation data
for a compatibility-prediction system, the method comprising:

(a) storing, for each of a plurality of subjects, a structured behavioral persona record derived
from first-party input by that subject, the persona record comprising at least one
**self-reported aversion field** identifying interpersonal behaviors the subject has stated they
dislike;

(b) for each of a plurality of candidate subject pairs, and for each of a plurality of scenario
records having **mutually distinct assigned social-difficulty levels**, at least one scenario
record specifying an unresolved disagreement and at least one scenario record specifying a lapse
in conversation, executing a plurality of simulation runs indexed by a seed value, each
simulation run comprising:

  (i) constructing a first enactment prompt from the first subject's persona record and the
  scenario record, and a second enactment prompt from the second subject's persona record and
  the scenario record, each enactment prompt comprising an instruction authorizing the
  corresponding agent to **reduce its conversational engagement** in response to disinterest;

  (ii) **verifying, prior to submitting either enactment prompt to a language model, that neither
  enactment prompt contains any string from a stored prohibited-phrase set**, the prohibited-phrase
  set consisting of phrases directing an agent toward cooperative, rapport-building, or
  affirmative conversational behavior, and blocking the simulation run on a match;

  (iii) responsive to the seed value belonging to a designated proper subset of the seed values,
  **inserting into the first enactment prompt content derived from the second subject's
  self-reported aversion field** together with an instruction to enact that content during the
  simulated conversation, and inserting into the second enactment prompt content derived from the
  first subject's self-reported aversion field, **wherein neither enactment prompt discloses that
  the counterpart enactment prompt has been so modified**;

  (iv) generating a transcript of alternating turns by repeatedly sampling a language model
  under the first and second enactment prompts respectively, **at a sampling temperature of at
  least 0.9**, wherein each agent's context presents its own prior turns in a first role and the
  counterpart's prior turns in a second role;

(c) computing, from the transcripts, a per-simulation-run outcome score, and grouping the outcome
scores by candidate subject pair;

(d) computing a **degeneracy determination** by comparing (A) a measure of dispersion of a
per-pair central tendency of the outcome scores **across** the candidate subject pairs against (B)
a measure of dispersion of the outcome scores **within** candidate subject pairs across seed
values, and against a stored minimum-dispersion threshold; and

(e) **only in response to the degeneracy determination indicating that (A) exceeds both (B) and
the threshold**, releasing an aggregate of the per-pair outcome scores to a downstream
subject-grouping process; and otherwise withholding the aggregate and emitting a
simulator-degeneracy signal.

### Claim 2 (dependent — the compute-allocation hook; strongest §101 posture)

The method of claim 1, wherein the plurality of simulation runs comprises a first-stage
simulation executed for every candidate subject pair using a first language model at a first
inference cost and **restricted to a single seed value selected from the designated proper subset
of seed values of step (b)(iii)**, and a second-stage simulation executed using a second language
model of higher inference cost over all of the scenario records and seed values, the second-stage
simulation being executed only for candidate subject pairs ranked above a threshold by
first-stage outcome score; and wherein the degeneracy determination of step (d) further gates
execution of the second-stage simulation.

*Rationale: ties the gate to a concrete saving of computational resources, which is the standard
technical-effect argument, and captures `single_shot()` choosing the friction seed as the cheap
run.*

### Claim 3 (dependent — scenario derivation and the dead-air seed turns)

The method of claim 1, wherein constructing the scenario record specifying an unresolved
disagreement comprises **selecting a disagreement topic by deterministic traversal of the
self-reported aversion field and an escalation field of the first subject's persona record
followed by those of the second subject's persona record**, such that a given subject pair is
assigned the same disagreement topic on every execution; and wherein the scenario record
specifying a lapse in conversation comprises a stored sequence of seed turns that are prepended
to the transcript as already-spoken turns.

### Claim 4 (dependent — enforcement as a build-time invariant)

The method of claim 1, further comprising executing an automated verification procedure over the
Cartesian product of the scenario records and the presence and absence of the insertion of step
(b)(iii), the verification procedure asserting absence of each phrase of the prohibited-phrase set
and of **at least one additional prohibited phrase stored separately from the prohibited-phrase
set**, and failing a build of the simulation system on any assertion failure.

*Rationale: the separately-stored extra phrases (`EXTRA_BANNED`) are a genuine and unusual
implementation detail — the test is designed not to be satisfiable by editing the list under test.
Small, but it is the kind of specific limitation that gets an allowance.*

### Claim 5 (independent — system claim, alternate framing on the gate)

A system for validating a generative conversation simulator prior to use in an interpersonal
matching decision, comprising one or more processors and memory storing instructions that cause
the system to: execute a persona-conditioned dyadic simulator over a plurality of subject pairs to
produce, for each subject pair, a plurality of outcome scores differing by simulation seed and by
scenario social-difficulty level; **decompose the variance of the outcome scores into a
between-pair component and a within-pair component**; classify the simulator as non-discriminative
when the between-pair component does not exceed the within-pair component or falls below a stored
floor; and, responsive to the non-discriminative classification, **suppress transmission of the
per-pair outcome scores to a group-assembly optimizer and increase at least one of the sampling
temperature, the number of scenario social-difficulty levels, and the proportion of simulation
seeds for which counterpart-derived aversion content is injected into an enactment prompt**,
before re-executing the simulator.

*Rationale: makes the gate a closed feedback loop that reconfigures the simulator, not merely a
report. A gate that only reports is much closer to "display a number"; a gate that automatically
adjusts the generator is a control system.*

---

## Design-Around Notes

### Risks to our own position

1. **"Tell the model not to be nice" is well-known prompt engineering.** Element (2) will be
   treated as ordinary skill and is likely to draw an Official Notice rejection. **Never make it
   the point of novelty**; recite it only as the environment the other elements operate on. The
   survey art (arXiv:2411.15287) explicitly lists prompt-level sycophancy priors among known
   mitigations, and its "sycophancy priors" result is a direct 103 reference.
2. **Elevated temperature is unpatentable alone** and reciting "0.9 to 1.0" invites a narrowing
   that a competitor evades at 0.85. Prefer "at least 0.9" as in Claim 1, and consider a
   functional alternative: "a sampling temperature selected such that a measured between-pair
   dispersion exceeds a threshold," which is harder to design around but harder to allow.
3. **CogniPair (arXiv:2506.03543) is a live §102 threat to the application framing.** Any claim
   reciting "simulate digital twins of two users conversing to predict romantic/social
   compatibility" without the gate is anticipated. Keep the gate in every independent claim.
4. **US12118471B2 is a live §103 threat to the banned-phrase element.** Emphasize in prosecution
   that the screened text is the *system's own generated prompt* (not untrusted input), the
   screened phrases are *benign*, and the purpose is *statistical variance preservation* rather
   than safety. Support this in the specification with the actual list and the actual failure mode
   ("two LLMs told to have a good conversation are pleasant to each other and the sim has no
   variance"), which is already written into `regulars/sim/prompts.py`'s module docstring —
   contemporaneous evidence worth preserving verbatim in the spec.
5. **The Cyara family (US11290400 / US11595332 / US11863507) is the best §103 combination
   partner** an examiner has against us: persona-driven variation of simulated human utterances,
   explicitly because a single canonical query is insufficient. Distinguish on *dyadic
   peer-to-peer* simulation (both sides simulated, neither is the system under test) and on the
   fact that Cyara varies *surface form of a fixed intent*, whereas friction injection varies
   *what one party actually does to the other*.
6. **Agent-Testing Agent (arXiv:2508.17393) anticipates graded difficulty and does it better**
   (adaptively). Do not claim difficulty grading as an independent contribution; recite the
   specific three-level ladder including the *lapse-in-conversation* scene with prepended seed
   turns, which is the unusual one.
7. **Jeter AI is patent-pending in this exact space** and its disclosure is not public. Monitor
   for publication at 18 months; if it predates our priority it could be a problem for the
   application-level framing (not for the gate).
8. **Public disclosure hygiene.** `docs/technical-architecture.md` §3 and §6 already describe all
   five measures in publishable prose. If this repository or that document becomes public, the
   US one-year grace period starts and most foreign rights (EPO, absolute novelty) are lost
   immediately. **File at least a provisional before any public repo, blog post, demo day, or
   investor deck describes the anti-sycophancy measures or the variance gate.** Treat the
   `BANNED_PHRASES` list and the between/within thresholds as confidential until filing.

### How a competitor designs around the drafted claims

- **Swap the gate statistic.** Replace between-vs-within std with AUC on a labeled dev set (which
  §6 already names as the *second* diagnostic), a permutation test on pair labels, per-pair rank
  stability across seeds, or an ICC. Mitigation: draft a dependent-claim ladder covering
  "a statistic that compares variation attributable to pair identity against variation
  attributable to simulation seed," and include ICC/F-ratio/AUC as enumerated alternatives in
  the specification so they are available for later claim amendment.
- **Move the friction out of the prompt.** Put the counterpart's aversions into the *scenario
  text* both agents see, rather than asymmetrically into one agent's instructions. This evades
  the "neither prompt discloses" limitation. Mitigation: a separate dependent claim covering
  symmetric scenario-level injection, and a claim to injecting for *a proper subset of runs* (the
  subset limitation is more durable than the asymmetry limitation, because the subset is what
  makes injected-vs-non-injected contrast measurable).
- **Achieve the prompt constraint by fine-tuning instead of by blocklist.** Train the persona
  model to be non-sycophantic and skip the banned-phrase check entirely. This is a real and
  probably superior engineering path and our claims will not reach it. Consider a separate
  disclosure covering "training a persona-enactment model on transcripts selected for
  between-pair discriminativeness," which is the natural next mechanism and currently unclaimed
  by anyone I found.
- **Gate on something other than variance.** Use calibration error or Spearman against held-out
  real ratings as the precondition. Mitigation: claim the *precondition relationship* ("withhold
  aggregate scores from the grouping process until a simulator-discriminativeness criterion is
  satisfied") in at least one independent claim, with variance as the preferred embodiment.

### Prior art to cite affirmatively in an IDS

Citing the strong references ourselves is the right move — it forecloses inequitable-conduct
arguments and forces the examiner to allow *over* them: US12412128B1, US20210223854A1,
US11863507B2 (and its family), US12118471B2, US20200034665A1, US7451161B2; and non-patent
literature Perez et al. arXiv:2212.09251, Sharma et al. arXiv:2310.13548, arXiv:2411.15287,
Park et al. arXiv:2304.03442, CogniPair arXiv:2506.03543, Agent-Testing Agent arXiv:2508.17393,
arXiv:2410.15226.

---

## Sources

- [US12412128B1 — Synthetic data using computer-simulated personas](https://patents.google.com/patent/US12412128B1/en)
- [US20210223854A1 — Evaluation of digital twins using social automatons](https://patents.google.com/patent/US20210223854A1/en)
- [US11863507 — Testing of automated contact center customer response systems (Cyara)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11863507)
- [US11290400 — same family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11290400) · [US11595332 — same family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11595332)
- [US12118471B2 — Mitigation for prompt injection in A.I. models](https://patents.google.com/patent/US12118471B2/en)
- [Deriving input restrictions for artificial intelligence agents (US12386924)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12386924)
- [US10395646B2 — Two-stage training of a spoken dialogue system](https://patents.google.com/patent/US10395646B2/en)
- [US20240095491A1 — Personalized multimodal response generation through virtual agents](https://patents.google.com/patent/US20240095491A1/en)
- [US7451161B2 — Compatibility scoring of users in a social network](https://patents.google.com/patent/US7451161B2/en) · [US20100185580A1](https://patents.google.com/patent/US20100185580A1/en)
- [US20200034665A1 — Determining validity of machine learning algorithms for datasets](https://patents.google.com/patent/US20200034665A1/en) · [US10990901B2 — Training, validating, monitoring AI/ML models](https://patents.google.com/patent/US10990901B2/en)
- [Perez et al., Discovering Language Model Behaviors with Model-Written Evaluations (arXiv:2212.09251)](https://arxiv.org/abs/2212.09251)
- [Sharma et al., Towards Understanding Sycophancy in Language Models (arXiv:2310.13548)](https://arxiv.org/abs/2310.13548)
- [Sycophancy in Large Language Models: Causes and Mitigations (arXiv:2411.15287)](https://arxiv.org/abs/2411.15287)
- [Measuring Sycophancy of Language Models in Multi-turn Dialogues (arXiv:2505.23840)](https://arxiv.org/abs/2505.23840) · [PARROT (arXiv:2511.17220)](https://arxiv.org/pdf/2511.17220)
- [Park et al., Generative Agents: Interactive Simulacra of Human Behavior (arXiv:2304.03442)](https://arxiv.org/pdf/2304.03442) · [Cohesive Conversations (arXiv:2407.09897)](https://arxiv.org/pdf/2407.09897)
- [CogniPair: GNWT-Based Multi-Agent Digital Twins for Social Pairing (arXiv:2506.03543)](https://arxiv.org/pdf/2506.03543)
- [Agent-Testing Agent (arXiv:2508.17393)](https://arxiv.org/pdf/2508.17393) · [Breakdowns in Conversational AI (arXiv:2604.02713)](https://arxiv.org/pdf/2604.02713)
- [On the Diversity of Synthetic Data (arXiv:2410.15226)](https://arxiv.org/abs/2410.15226) · [Measuring Diversity in Synthetic Datasets (arXiv:2502.08512)](https://arxiv.org/pdf/2502.08512)
- [Comparison of Diverse Decoding Methods (arXiv:1906.06362)](https://arxiv.org/pdf/1906.06362) · [When Temperature Fails, Change the Loss (arXiv:2508.09654)](https://arxiv.org/pdf/2508.09654)
- [Attention Mechanism for LLM-based Agents under Information Asymmetry (arXiv:2502.13160)](https://arxiv.org/pdf/2502.13160) · [PrefBench (arXiv:2605.22855)](https://arxiv.org/pdf/2605.22855) · [TERMS-Bench (arXiv:2605.13909)](https://arxiv.org/pdf/2605.13909)
- [Not All Flips Are Conformity (arXiv:2606.00820)](https://arxiv.org/pdf/2606.00820) · [When Identity Skews Debate (arXiv:2510.07517)](https://arxiv.org/pdf/2510.07517)
- [Jeter AI — patent-pending AI matchmaking](https://jeter.ai/) · [Ditto date-simulation matchmaking (press)](https://medium.com/reflections-and-realities/ditto-built-an-ai-that-just-sends-you-on-dates-afb119bc5252)

*Not legal advice. Prepared as engineering input to a patent-counsel conversation.*

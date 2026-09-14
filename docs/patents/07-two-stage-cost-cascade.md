# 07 — Two-Stage Cost Cascade for LLM-Simulated Pairwise Compatibility

**Status:** Prior-art analysis complete. **Verdict: NOT independently patentable. Treat as trade secret / execution advantage.**
**Analyst date:** 2026-09-14
**Code under analysis:** `regulars/sim/two_stage.py`, `regulars/sim/cache.py`, `regulars/sim/harness.py`, `regulars/llm/anthropic_client.py`, `docs/technical-architecture.md` §8

---

## Mechanism Summary

Regulars must score pairwise compatibility by running an LLM-simulated dyadic conversation between two persona specifications. Naive all-pairs evaluation is O(n²) — §8 of the architecture doc puts this at ~50M simulations at 10,000 users. The implemented cost-control cascade has five layers:

1. **Prefilter (pre-LLM).** Geography, availability windows, and stated exclusions (`PersonaSpec.stated.exclusions`) cap the candidate set at roughly 200 per user before any model is called. §8 scores this as 50M → 1M.
2. **Cheap low-fidelity first pass.** `rank_candidates()` in `regulars/sim/two_stage.py` runs `single_shot(config)` — one scenario (`COLD_OPEN`) × one seed — for **every** prefiltered candidate on a Haiku-class `cheap_llm`, producing a rough ordering via an injected `score_fn`.
3. **Top-K frontier pass.** Only `first_pass_top_k` (default 40) candidates get the full grid — 3 scenarios × 5 seeds — on `frontier_llm`. Results are merged: top-K sorted by frontier score, the tail retained at cheap score with `frontier_score is None`.
4. **System-prompt caching.** `AnthropicLLM._system_param()` (`regulars/llm/anthropic_client.py`) emits the persona-prefix system prompt as a single text block carrying `cache_control: {"type": "ephemeral"}`. Because the persona spec is identical across every simulation that persona participates in, the prefix is billed at write cost once and at read cost thereafter. `UsageTotals` separately tracks `cache_read_tokens` and `cache_creation_tokens`.
5. **Result memoization keyed on persona identity.** `SimCacheKey(persona_hash_a, persona_hash_b, scenario_id, seed, model)` (`regulars/sim/cache.py`) with in-memory and append-only JSONL backends. `persona_hash()` in `harness.py` is the first 12 hex chars of SHA-256 over canonical sorted-key JSON of the `PersonaSpec`. Any spec change changes the hash and misses; identical inputs hit and skip the LLM entirely. Steady-state recomputation therefore happens only on a persona version bump or a genuinely new candidate — not on every batch run.

The claimed differentiator is that layer 5 is *cache-consistent with an append-only persona architecture*: persona versions are immutable (`regulars/persona/versioning.py`, `PersonaVersion.version` monotonic, `next_version()` never mutates), so invalidation is tied to a specific versioned behavioral-model update rather than a TTL or a similarity threshold.

---

## Prior Art Found

### A. Coarse-to-fine cascades — foundational, ~2001

Viola & Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features* (CVPR 2001), is the canonical reference: each candidate window passes through "a series of tests of increasing difficulty," where early stages cheaply reject the bulk of candidates and "subsequent stages perform more detailed analyses using larger and more discriminating sets of features." The entire architectural idea of *spend cheap compute on all candidates, expensive compute on survivors* is 25 years old and thoroughly published. Follow-on literature (Optimally Training a Cascade Classifier, arXiv:1008.3742; Training Effective Node Classifiers for Cascade Classification, arXiv:1301.2032) generalizes it well beyond vision.

**Effect on our mechanism:** layers 2–3 are a textbook two-stage cascade. No novelty.

### B. Candidate generation → reranking in search and recsys — universal

- Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016) established the "funnel": a candidate-generation model winnows millions to hundreds, a ranking model scores the hundreds. LinkedIn, Pinterest and YouTube all ship this shape.
- *Towards a Theoretical Understanding of Two-Stage Recommender Systems* (arXiv:2403.00802) treats candidate-retrieval-then-ranking as the industry default worth theorizing about — i.e. as settled practice.
- Two-stage retrieval + cross-encoder reranking is the default IR pipeline (first-stage retriever returns top-100, cross-encoder rescores).

**Patents in this family (all predate and read on layers 1–3):**

| Patent | Assignee | Priority | Relevance |
|---|---|---|---|
| **US20220335274A1** — Multi-stage computationally efficient neural network inference | Google LLC | 2021-04-14 | A cheap "student" network scores every input; a confidence test (margin between top categories, an explicit abstain class, hard-category membership) decides whether to escalate to an expensive "teacher" network. This is our layers 2–3 with a confidence gate instead of a top-K gate. **Closest general-purpose art.** |
| **US11568309B1** — Resource-efficient data collection for multi-stage ranking systems | Meta Platforms | 2019-07-02 | Trains early-stage rankers with soft labels expressing the probability an item would land in the top-K if the full set had been scored by the target model. Explicitly about avoiding exhaustive expensive scoring in a multi-stage ranker. |
| **US11194819B2 / US11704600B2 / US12387149** — Multistage feed ranking system (scoring-model optimization for scaling; scalable multi-objective model approximation) | LinkedIn / Microsoft | 2017–2019 | First-pass ranker deliberately omits expensive features; feature inclusion is decided on importance *versus computing-resource cost*; second pass applies the complex model to top-K only. |
| **US20110125732A1** — Internal ranking model representation schema | Microsoft | 2009 | Linear model first, neural re-rank of the survivors. |

### C. Prefiltering to defeat quadratic pairwise cost — standard "blocking"

Reducing |E|·(|E|−1)/2 comparisons by grouping records that share an attribute is the *blocking* step of record linkage / entity resolution, surveyed at length (Papadakis et al., *Blocking and Filtering Techniques for Entity Resolution: A Survey*, arXiv:1905.06167; VLDB vol. 9 p.684). Blocking "trades slightly lower effectiveness for significantly higher efficiency" by comparing only within blocks. Geographic bucketing is the textbook blocking key.

**Effect:** layer 1 is blocking. No novelty. Dating-specific geographic filtering is also explicit in dating patents (US7203674B2).

### D. LLM cost cascades and model routing — dense, recent, and heavily published

- **FrugalGPT** (Chen, Zaharia, Zou; arXiv:2305.05176, May 2023) — the canonical LLM cascade: cheap models first, escalate on low confidence; matches GPT-4 quality at up to 98% cost reduction. Names "LLM cascade" as one of three cost-reduction strategy families.
- **RouteLLM** (UC Berkeley, ICLR 2025) — >85% cost reduction, 14% of queries to the strong model.
- **Cascade-Aware Training of Language Models** (arXiv:2406.00060); **C3PO** (arXiv:2511.07396, NeurIPS 2025) — LLM cascades under probabilistic cost constraints; **EcoRank** (arXiv:2402.10866) — *budget-constrained text re-ranking using LLMs*, which is specifically cheap-LLM-then-expensive-LLM reranking; **SMART** (arXiv:2403.13835); **AutoRelAnnotator** (arXiv:2606.25871) — calibrated model cascades for cost-efficient relevance evaluation.

**Effect:** applying the cascade *to LLMs specifically*, which is the only thing that could have been fresh about layers 2–3, was published by 2023 and is now a commodity pattern with named products.

### E. Prompt / prefix caching — vendor feature, published, and not ours

- Gim et al., *Prompt Cache: Modular Attention Reuse for Low-Latency Inference* (arXiv:2311.04934); *RelayAttention for Efficient LLM Serving with Long System Prompts* (arXiv:2402.14808); NVIDIA NIM KV-cache-reuse documentation; vLLM automatic prefix caching.
- Anthropic's `cache_control: ephemeral` is **the provider's own feature**. `AnthropicLLM._system_param()` calls an API. Using a documented vendor API as documented is the definition of practising prior art — and any patent position here would belong to the provider, not to Regulars.

**Effect:** layer 4 is unpatentable by us. Full stop.

### F. LLM response caching keyed on an input hash

- **US12259913** — *Caching large language model (LLM) responses using hybrid retrieval and reciprocal rank fusion* (2024). Three-layer cache to cut GPT cost and latency.
- **MeanCache** (arXiv:2403.02694), GPTCache, LiteLLM caching, and the broader semantic-cache literature all describe hashing the prompt to a key and returning stored output on a hit. Exact-hash caching is explicitly described as the *baseline* that semantic caching improves on.
- Version- and content-hash-based invalidation is ancient art in its own right: **US6934720B1** (Automatic invalidation of cached data), **US7120747B2** (hashed-key cache invalidation), EP3053323B1 / US20150100664A1 (notification-based invalidation), plus every content-addressed build system (ccache, Bazel, Nix) keying an action cache on a hash of its inputs including the tool version.

**Effect:** layer 5's structure — hash the inputs (including the model identifier) and memoize — is squarely prior art. Keying on the *model* as part of the cache key is standard in build-system action caches ("the compiler version is part of the key").

### G. The application domain is also occupied

| Patent / paper | Owner | Relevance |
|---|---|---|
| **US9679259** — Training and employing a machine learning system in evaluating entity pairs | (dating/matching) | ML scoring of *entity pairs* for match potential — the pairwise-compatibility framing. |
| **WO2007118202A2** — Learning-based recommendation system incorporating collaborative filtering and feedback | eHarmony | Pairs date-seekers with candidates, produces a compatibility value predicting mutual satisfaction. |
| **US7203674B2**, **US20020040310A1**, **US20170228819A1**, **US11455827** | various | Bidirectional criteria scoring, behavior-derived preference weighting, per-user continuously-learning compatibility models. |
| **US11257496** — Persona-based agent interactions with online visitors | — | LLM/agent personas driving conversations. |
| **arXiv:2512.11844 — "Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World Engines"** (Shang, Yan & Liu, Dec 2025) | — | **Most on-point prior art for the domain.** Generates 300–500 word personas from structured profiles, simulates dyadic conversations with an LLM, and scores compatibility from the transcript (either agent self-rating or an external LLM judge on interaction quality / mutual engagement / conversational flow). Validated on speed-dating and divorce-prediction data. It does *not* address cost cascades, cheap-model prefiltering, top-K reranking, or caching. |

---

## Novelty Analysis

Taking the five layers individually, **none is novel**, and four of the five are not merely anticipated but are named, taught, and commercially deployed techniques:

| Layer | Novel? | Killed by |
|---|---|---|
| 1. Geographic/availability/exclusion prefilter | No | Blocking in entity resolution (arXiv:1905.06167); dating patents US7203674B2 |
| 2. Cheap model, 1 scenario × 1 seed, all candidates | No | Viola-Jones 2001; US20220335274A1 (Google); FrugalGPT |
| 3. Frontier model on top-K, 3 × 5 grid | No | YouTube funnel; US11568309B1 (Meta); US11194819B2 (LinkedIn); EcoRank |
| 4. System prompt with `cache_control: ephemeral` | No | It is Anthropic's own published API feature; arXiv:2311.04934 |
| 5. Memoize on hash(persona_a, persona_b, scenario, seed, model) | No | US12259913; US6934720B1; content-addressed build caches |

**The strongest remaining argument** — and it is genuinely the only one worth examining — is the *combination* claim: this specific cascade applied to **LLM-simulated dyadic conversation for compatibility scoring**, with a cache key whose invalidation semantics are bound to an **append-only, immutable persona version** rather than a TTL or an embedding-similarity threshold.

Three problems with it, in descending order of severity:

**1. Combination of known elements toward a predictable result.** Under *KSR v. Teleflex*, combining a cascade classifier, a blocking prefilter, a vendor caching directive, and an input-hash memo cache — each performing exactly its known function, each yielding exactly its expected cost saving, with no interaction between them that a person of ordinary skill would find surprising — is prima facie obvious. §8 of our own architecture doc presents the levers in a table of independent multipliers. That table is, rhetorically, an obviousness argument written by us: each row is a known lever with a known effect.

**2. The version-keyed-invalidation angle is weaker in the code than in the pitch.** `SimCacheKey` does not carry `PersonaVersion.version`. It carries `persona_hash()` — SHA-256 over canonical JSON of the spec. That is a **content hash**, which is the *most* generic and most heavily anticipated cache-key formulation in the entire prior-art set (every content-addressed build cache; US7120747B2). The append-only version store makes the content hash *reliable*, but the cache does not read the version number and would behave identically against a mutable store that happened not to mutate. To claim version-bound invalidation we would have to actually key on the version identifier — and even then, "cache key includes the version of the input artifact" is what an ETag is.

**3. §101 exposure.** In the US, a claim reciting "prefilter candidates, score them with a cheap model, rescore the best with an expensive model, and cache the results" is at serious risk of being read as an abstract idea (organizing/comparing information, plus a mental-process-style ranking) implemented on a generic computer, with the LLM as the "apply it" element. There is no asserted improvement to the functioning of a computer beyond "does fewer expensive calls," which is the essence of the abstract idea rather than a technical solution to a technical problem. EPO would additionally treat the cost saving as an administrative/economic aim and the cascade as an obvious implementation.

**What *is* genuinely distinctive** (and worth writing down, just not in a patent):
- The cheap first pass is not an approximation of a *different* function — it is the *same* simulation function at reduced fidelity (1 scenario × 1 seed vs. 3 × 5). The cascade's two stages differ in *sampling budget over a stochastic simulator*, not in model architecture or feature set. That framing is somewhat unusual relative to classifier cascades, where stages compute different features.
- The cache key spans a **symmetric pair** of behavioral specs plus a seed — i.e. it memoizes a *stochastic, order-dependent dyadic interaction*, not a single-input inference. Note the code does **not** currently canonicalize pair order, so `(a,b)` and `(b,a)` are distinct keys.
- The interaction between the append-only persona store and the memo cache means the system has no invalidation *event* at all: a new version is simply a new key, and the old transcript remains valid evidence of what the old persona would have done. That is an audit-friendly property.

None of those three rises to patentable novelty. The first two are engineering choices; the third is a consequence of using immutable data, which is itself the point of immutable data.

---

## Patentability Verdict

**Not independently patentable. Do not file on this mechanism.**

Confidence: high. The cascade is Viola-Jones; the candidate-generation-then-rerank funnel is the default architecture of every large recommender and is covered by granted patents at Google (US20220335274A1), Meta (US11568309B1) and LinkedIn/Microsoft (US11194819B2, US11704600B2); the prefilter is blocking; prompt caching is Anthropic's own feature; and hash-keyed LLM response caching is both patented (US12259913) and commoditized (GPTCache, LiteLLM). The only unoccupied ground — the combination applied to LLM-simulated dyadic conversation — is a predictable aggregation of known elements under *KSR*, sits in a domain where the simulate-then-score framing is itself now published prior art (arXiv:2512.11844), and would face substantial §101 abstract-idea exposure.

**Recommended treatment: trade secret and competitive execution advantage.** The value here is real but it is operational, not proprietary: the specific top-K value, the choice of `COLD_OPEN` as the first-pass scenario, the seed count at which frontier-score variance stops mattering, the measured cache hit rate in steady state, and the resulting cost-per-match curve. Those numbers are what make the difference between a $50 run and a five-figure bill, they are not discoverable from the product surface, and they are not disclosed by publishing nothing. Keep them out of public repos, public docs, blog posts and conference talks.

**Defensive note.** Because the pattern is so thoroughly anticipated, freedom-to-operate risk is correspondingly low but not zero — the Google, Meta and LinkedIn patents above are granted and broad. We are almost certainly safe under the sheer density of prior art plus the domain distance (none of them claims simulated conversation), but the tail risk is non-trivial enough that it should go to counsel if the company ever raises on this architecture. Consider a defensive publication (see Design-Around Notes) if the goal is only to prevent someone else from patenting it.

---

## Draft Claims

These are drafted **only** to preserve the narrowest defensible option if counsel decides a filing has strategic (not enforcement) value — e.g. to pad a portfolio or to establish priority in a combination filing alongside mechanisms 1–6. Standing alone, my recommendation is not to file any of them. Each is deliberately narrow enough to be nearly unenforceable, which is the honest cost of getting past the art above.

### Claim Sketch 1 — Fidelity-graded stochastic simulation cascade (narrowest, least bad)

> A computer-implemented method for evaluating pairwise compatibility among users of a matching system, comprising:
> (a) storing, for each user, an append-only sequence of immutable behavioral specification records, each record comprising a version identifier and a plurality of structured natural-language behavioral fields including at least one negative-space field enumerating friction points or interaction costs;
> (b) selecting, for a first user, a candidate set by applying non-model filters comprising geographic proximity, availability overlap, and user-stated exclusions, prior to invoking any language model;
> (c) for **every** candidate in the candidate set, executing a first simulation pass in which a first language model generates a dyadic conversation transcript between a first agent conditioned on the first user's current behavioral specification record and a second agent conditioned on the candidate's current record, the first pass being constrained to a single scenario and a single random seed;
> (d) ranking the candidate set by a score derived from the first-pass transcripts;
> (e) for a top-K subset only, executing a second simulation pass using a second language model of greater per-token cost, the second pass executing the **same simulation procedure** at higher fidelity, namely a plurality of distinct scenarios each replicated across a plurality of distinct random seeds; and
> (f) producing a final ordering in which the top-K subset is ordered by second-pass score and the remainder retains first-pass order.

*Distinguishing feature relied upon:* (c) and (e) invoke the identical simulation procedure differing only in sampling budget over a stochastic simulator, combined with (a)'s negative-space requirement. **Assessment: still likely obvious over US20220335274A1 + FrugalGPT.** The examiner's rejection writes itself.

### Claim Sketch 2 — Version-bound dyadic memoization (requires a code change to support)

> The method of Claim 1, further comprising maintaining a memoization store keyed on a tuple consisting of the version identifier of the first user's behavioral specification record, the version identifier of the candidate's behavioral specification record, a scenario identifier, a random seed, and a language-model identifier, wherein:
> (i) the store is append-only and no entry is ever invalidated or expired;
> (ii) appending a new immutable behavioral specification version for a user causes subsequent simulations involving that user to miss the store and be recomputed, while every transcript computed against prior versions of that user remains retrievable and remains a valid record of the behavior of those prior versions; and
> (iii) no time-to-live, no embedding-similarity threshold, and no content-similarity comparison participates in store lookup.

*Note:* limitation (ii)'s second half — retaining old-version transcripts as valid historical evidence rather than treating them as stale — is the only limitation in this whole document that a generic LLM cache genuinely lacks, because a generic cache has no notion that the superseded entry is still *true of something*. **This requires changing `SimCacheKey` to carry `PersonaVersion.version` instead of (or alongside) `persona_hash`.** As written today the code does content-hash caching and reads directly on US12259913 and on content-addressed build caches. If any claim here is worth pursuing, this is the one, and the code change is small.

### Claim Sketch 3 — Prefix amortization across a fan-out (weakest; include only as a dependent)

> The method of Claim 1, wherein the behavioral specification of the first user is supplied to the second language model as a system-prompt segment marked with a cache-retention directive, such that the segment is billed at a cache-write rate for a first simulation in a batch and at a cache-read rate for each subsequent simulation in the same batch, and wherein candidate simulations for the first user are scheduled contiguously so as to maximize reuse of the retained segment.

*The only arguably non-obvious sliver is the last clause: **scheduling the fan-out to keep the prefix hot** — an ordering decision, not merely a flag. Even that is standard practice in batch-inference serving.* Do not file as an independent claim. Our code does not currently implement the scheduling clause; `asyncio.gather` over candidates gives it incidentally.

---

## Design-Around Notes

### Prior art to cite if we ever prosecute (cite it ourselves; do not let an examiner find it first)

**Cascade / coarse-to-fine:** Viola & Jones, CVPR 2001; arXiv:1008.3742; arXiv:1301.2032.
**Multi-stage ranking patents:** US20220335274A1 (Google, 2021); US11568309B1 (Meta, 2019); US11194819B2, US11704600B2, US12387149 (LinkedIn/Microsoft); US20110125732A1 (Microsoft).
**LLM cascades:** arXiv:2305.05176 (FrugalGPT); RouteLLM (ICLR 2025); arXiv:2406.00060; arXiv:2511.07396 (C3PO); arXiv:2402.10866 (EcoRank); arXiv:2403.13835 (SMART).
**Blocking:** arXiv:1905.06167; VLDB 9:684.
**Prompt/prefix caching:** arXiv:2311.04934 (Prompt Cache); arXiv:2402.14808 (RelayAttention); Anthropic prompt-caching documentation; vLLM automatic prefix caching.
**LLM response caching:** US12259913; arXiv:2403.02694 (MeanCache); GPTCache; LiteLLM.
**Cache invalidation:** US6934720B1; US7120747B2; EP3053323B1 / US20150100664A1.
**Domain:** arXiv:2512.11844 (Love First, Know Later) — the closest domain art; US9679259; WO2007118202A2; US7203674B2; US20020040310A1; US11455827; US11257496.

### Freedom-to-operate posture

Low risk, for the counter-intuitive reason that the art is *too* dense: the general cascade is unenforceably well-known, and each granted patent above is tied to its own domain (image classification for Google, feed ranking for Meta/LinkedIn, document retrieval for Microsoft). None claims LLM-simulated conversation or pairwise persona compatibility. Keep it that way — see below.

### How to stay clear

1. **Do not adopt a confidence-threshold escalation gate.** US20220335274A1 claims escalation triggered by a confidence measure on the cheap model's output (margin between top scores, an abstain class, hard-category membership). Our gate is a fixed top-K on a ranking, which is structurally different and reads on Meta/LinkedIn instead — and those are ranking patents in a different domain. If someone proposes "only escalate when the cheap score is ambiguous," that change *moves us toward* Google's claims for no cost benefit. Keep the top-K gate.
2. **Do not train a learned router.** FrugalGPT-style learned triage (a DistilBERT scoring function choosing among APIs) is the most heavily patented and published sub-area. Our ranking is produced by the simulation itself, not by a separate router model. Keep it that way.
3. **Do not add semantic/embedding similarity to the cache.** Semantic caching (US12259913, MeanCache) is where the LLM-caching patents concentrate. Exact-key lookup on a hash is both safer and, given immutable personas, strictly correct. `SimCacheKey` should stay an exact-match key.
4. **Canonicalize pair order in the cache key if and only if the simulation is order-symmetric.** It is not — `run_pair` assigns roles and the first speaker differs — so the current asymmetric key is correct. Document that, because it looks like a bug and someone will "fix" it.

### If the goal is purely defensive

File a **defensive publication** (e.g. IP.com, or simply a dated public technical post) describing the cascade at the level of §8. Cost: near zero. Effect: bars a competitor from later patenting the same combination against us. But weigh this against the trade-secret value of the tuned constants — publish the *shape* (prefilter → cheap pass → top-K frontier pass → prefix cache → memoize), never the *numbers* (200, 40, 1×1, 3×5, measured hit rates, cost-per-match). The shape is already unpatentable by anyone; the numbers are the asset.

### Where the real IP actually is

This mechanism is the cost floor that makes the product viable, not the product's defensible core. If there is a patentable invention anywhere near here it is **upstream** — in how a `PersonaSpec` is elicited and versioned (the negative-space fields `friction_points` and `drains`, without which the architecture doc notes "two LLM personas are pleasant to each other and the sim has no variance"), and in how simulated transcripts are scored into a compatibility signal. Those should be assessed on their own merits in the mechanisms covering persona extraction and judging. This document's mechanism should be explicitly **excluded** from any independent claim in those filings, because including it would import the entire prior-art wall catalogued above into an otherwise cleaner application.

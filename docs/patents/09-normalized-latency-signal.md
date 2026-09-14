# 09 — Per-User, Per-Hour Normalized Response Latency as a Dyadic Connection Signal

**Status:** prior-art research complete, patentability verdict below
**Subject code:** `regulars/features/real.py` (`LatencySample`, `UserBaseline`, `fit_user_baseline`, `normalize`, `latency_features`)
**Spec reference:** `docs/technical-architecture.md` §4, "Real chat only (the irreplaceable ones)" → *Response latency*
**Date of search:** 2026-09-14
**Verdict in one line:** Weak-to-moderate patentability — the general idea is obvious and the dating-app incumbent (Plentyoffish/Match Group) already owns reply-latency-as-a-matching-signal, so only a narrowly-drafted combination claim survives, and that claim carries severe §101 (Alice) exposure; defensive publication plus trade secret is the better economic choice.

---

## 1. Mechanism Summary

### What the system actually does

Regulars ingests real iMessage transcripts and extracts a *directed, pairwise* feature describing how quickly person A answers person B, relative to A's own habits. The mechanism has five distinct steps, and the novelty argument — such as it is — lives entirely in their combination, not in any single step.

**Step 1 — Reply-gap extraction with explicit reply resolution.** For a directed pair (A replying to B), the system walks the transcript and identifies each turn by A that answers a turn by B. An explicit `reply_to` pointer wins; absent one, the answered turn is the immediately preceding spoken turn. Tapbacks (iMessage reactions) are neither turns nor reply targets. The gap is the difference of `sent_at_ms` timestamps.

**Step 2 — Disengagement-threshold exclusion.** Any gap exceeding `MAX_GAP_MS = 4 hours` is dropped before any statistic is computed, on the stated theory that a four-hour gap encodes sleep or work, not disengagement. Negative gaps (clock skew, out-of-order delivery) are also dropped.

**Step 3 — Per-user, per-hour-of-day baseline.** Over a user's *entire* message history — deliberately not the conversation being scored — the system bins every reply gap by local hour of day (0–23), takes `log(raw_ms)`, and stores the median per bucket. A bucket with fewer than `MIN_HOUR_SAMPLES = 5` observations is discarded and the user's overall log-median is used instead. This sparse-bucket fallback is the documented mechanism that keeps a thin 4 a.m. bucket from producing a nonsense baseline.

```
latency_norm = log(raw_ms) − median(log latency of this user, this hour-of-day)
```

**Step 4 — Normalization against that specific user's own baseline.** Each observed gap becomes a signed deviation. Negative means "A answered B faster than A normally answers anyone at this hour." The baseline is per-*replier*, not per-dyad and not per-population, which is the point: raw response time is dominated by a person's general responsiveness temperament and their time-of-day availability, and an absolute threshold measures those rather than the connection in the exchange.

**Step 5 — Within-conversation distribution as the feature, not the raw mean.** Two features are emitted per direction:

- `latency_norm_median` — the median of the normalized deviations; below zero means A answers B faster than A's own habit.
- `latency_norm_fast_share` — the share of replies that beat A's own baseline.

Both are directed (A→B and B→A are computed separately and can disagree) and both feed the ranking/matching judge described in §5 of the architecture doc, alongside reply-graph weight and tapback rate.

### The academic finding being operationalized

Templeton, Chang, Reynolds, Cone LeBeaumont, Wheatley, *Fast response times signal social connection in conversation*, **PNAS 119(4), 2022** ([doi:10.1073/pnas.2116915119](https://www.pnas.org/doi/10.1073/pnas.2116915119)). Across conversations, within conversations, and at the partner level, faster response predicted felt connection; sub-250ms responses precede conscious control and so read as honest signals even to third-party observers. Critically, the paper's *within-conversation* result is that connected moments are fast **relative to that conversation's own baseline** — which is exactly what step 5 operationalizes, and exactly what an absolute-threshold implementation would miss.

The finding itself is published science and therefore unpatentable and, worse, is prior art against any claim whose inventive contribution is "use the Templeton finding." The only defensible position is that the *computable normalization that makes the finding usable in asynchronous text* is the invention.

---

## 2. Prior Art Found

Ordered by threat level. Everything below was located and read during this search; assignees and dates are as reported by Google Patents / USPTO full-text.

### Tier 1 — Direct threat: a Match Group subsidiary already owns latency-as-a-dating-signal

**US 10,540,607 B1 — "Apparatus, method and article to effect electronic message reply rate matching in a network environment"**
Assignee: **Plentyoffish Media ULC** (acquired by Match Group, 2015). Priority 2013-12-10, filed 2014-12-08.
[patents.google.com/patent/US10540607B1](https://patents.google.com/patent/US10540607B1/en)

This is the single most dangerous reference in the set, and the answer to the brief's question "has a major dating app already patented something functionally similar": **substantially yes, for the general mechanism; no, for the specific normalization.**

What it claims: generating a machine-learned *response predictive model* from a training set comprising user profile information, **user behavioral information**, and message data indicative of messaging activity; then using the model to identify matches with the greatest likelihood of responding to a message from a given user.

What its specification enumerates as behavioral parameters: "the **shortest, average, and longest times required to respond** to a received inbound electronic message." Response latency is therefore expressly disclosed, by a dating platform, as a behavioral input to a machine-learned matching model.

Why it is not fatal: latency appears in the specification as one enumerated behavioral parameter among many, not in the independent claims; the predicted quantity is *reply propensity* (will this person answer at all), used for **recipient selection** — showing you people likely to write back — not *interpersonal enjoyment* between two people already conversing. Nothing in it computes a per-user baseline distribution, normalizes against it, buckets by hour of day, or excludes long gaps. Its use of latency is absolute and population-comparative; ours is explicitly and only relative to the individual's own habit.

Practical consequence: any claim that reads on "use response time as an input to a dating/matching model" is dead on arrival. Match Group's examiner-cited art and its portfolio make that space theirs. A claim must recite the normalization as a positive, essential limitation, not as a dependent-claim refinement.

Related, same family / same assignee, worth reading before filing: **US 10,769,221** (POF, matching on actual actions and likelihood of receiving a response) and **US 10,387,795 B1** (POF, ML for service-level upgrade offers — shows the same behavioral-feature vocabulary).

### Tier 2 — Anticipates the baseline-construction step almost exactly (but for a different purpose)

**US 8,862,519 B2 — "Predicting email response patterns"**
Assignee: originally **International Business Machines**; **assigned to Snap Inc. in 2018**. Priority 2008-12-29.
[patents.google.com/patent/US8862519B2](https://patents.google.com/patent/US8862519)

Builds, per user, an average time-to-read/time-to-respond that is **conditioned on both the sender and the time window** — the specification's worked example gives twenty minutes for Monday–Friday 08:00–17:00 and one hundred eighty minutes for Saturday–Sunday, for the same recipient. It also detects *change* in a user's time-to-act relative to their established figure. Output is a probabilistic prediction ("response by 9am — 95%").

This is the closest art to steps 3 and 4 considered in isolation. It establishes that "a per-user, time-of-day-conditioned historical baseline of message response latency, and deviation from it" was known in 2008. What it does not do: use the deviation as an *affinity* or *compatibility* signal, aggregate it over a conversation's distribution, log-transform, use medians and a sparse-bucket fallback, or exclude long gaps.

The Snap Inc. ownership is a live commercial risk independent of validity: a messaging company now holds the foundational reference in this space.

**US 8,831,571 B2 — "Dynamic text communication administration"**
Assignee: **AT&T Intellectual Property I, L.P.** Priority 2010-09-28.
[patents.google.com/patent/US8831571B2](https://patents.google.com/patent/US8831571B2/en)

Explicitly computes "an average response time by user for text messages **based on the time of day**," with morning / afternoon / evening buckets, then auto-replies to a sender with that figure ("I typically respond to messages received during this time in about 45 minutes"). Purpose is sender-facing transparency, not scoring. But element (a) of our claim — a per-user, per-time-of-day baseline response latency for **SMS/text specifically** — is disclosed here in plain terms.

### Tier 3 — Anticipates the deviation-from-own-baseline step

**US 9,137,181 B2 / US 8,352,561 / US 8,046,418 / US 7,921,174 — "Electronic communication reminder technology"**
Assignee: **Google LLC**. Priority 2009-07-24.
[patents.google.com/patent/US9137181B2](https://patents.google.com/patent/US9137181B2/en)

Maintains per-user *response pattern data* keyed to message characteristics including time/date of receipt and sender identity, learns when the user habitually responds ("90% of the user's past responses have occurred between 10pm and 11pm"), and fires a reminder when a response is **overdue relative to that user's own established pattern**. The claims do not require matching the same hour-of-day bucket, but the concept "compare this response against this user's own time-conditioned history and flag the deviation" is squarely disclosed.

**US 8,095,612 B2 — "Ranking messages in an electronic messaging environment"**
Assignee: Google. Measures how quickly a user acts on a message and correlates that speed with the message's importance *to that specific user*, feeding a ranking. Speed-of-action → relative importance → ranking is the same logical arc as ours.

### Tier 4 — Generic baseline-and-deviation with time-of-day normalization (kills "the technique is novel" arguments)

Baseline-then-normalized-deviation, including explicit time-of-day normalization to suppress false positives, is textbook in security analytics and APM:

- **US 10,645,109** — anomalous user network activity; baseline model of a user's daily pattern, reconstruction error vs. threshold.
- **US 10,437,831** — insider-threat detection; per-employee baselines with explicit hour-of-day normality windows ("17:00–19:00 is normal office exit time").
- **US 10,205,736 / US 10,666,673** — behavioral baselining of network systems, temporal boundaries.
- **US 7,676,706** — baselining backend component response time; expressly uses *"normalization"* at differing activity periods "to reduce false alarms that appear as deviations but are actually just reflections of expected increased activity."

An examiner reaching for an obviousness rejection will pair one of these with US 10,540,607 and argue the combination is a predictable application of a known normalization technique to a known signal. That is the rejection to draft around.

Also relevant on time-of-day-conditioned responsiveness: **US 8,712,943 / US 8,712,944 / US 9,135,558** (Q&A expert routing; base response rates per time-of-day-of-week bucket, with pseudo-counts for sparse buckets — note that even the *sparse-bucket smoothing* idea has an analogue here).

### Tier 5 — Academic / non-patent literature

- **Templeton et al., PNAS 2022** — the underlying finding. Public 2022-01-18; unpatentable itself and prior art against any claim reading on the finding.
- **Florian, Hakobyan & Drimalla, "Sorry for the late reply: Response times and reciprocity in WhatsApp and Instagram chats," arXiv:2605.03687 (2026-05-05)** — 3.4M messages, 889 dyadic chats, 97 participants. Highly relevant and very recent, but methodologically *distinct in exactly our direction of novelty*: it bins response times into seven fixed absolute categories (1min, 2min, 3–5min, …, >60min), compares distributions with Jensen-Shannon distance, and explicitly does **not** normalize per-user, per-hour-of-day, or log-transform. It is evidence that serious researchers working this exact problem in 2026 were *not* doing our normalization — useful for non-obviousness — while also being evidence that the field is now crowded.
- **Lognormal response-time modeling** (van der Linden and the psychometric RT literature) — log-transforming response times to normalize an inherently lognormal quantity is textbook. Do not attempt to claim the log transform as such.
- Interactivity/response-latency in CMC: *Journal of Computer-Mediated Communication* 23(4):201 (2018) on conversational contingency and response latency; ACM CSCW `10.1145/3361115` on interpretations of response delays.

### What was searched and not found

No patent or published application was located that recites, in any combination: (i) a per-user **hour-of-day-bucketed median of log response latency**, (ii) an explicit **sparse-bucket fallback to an overall median**, (iii) exclusion of gaps above a **disengagement threshold** framed as sleep/context-switch, and (iv) use of the **within-conversation distribution of the normalized deviations** as a **directed pairwise** input to a social matching or group-assembly system. Searches covered Google Patents, USPTO full-text, Justia assignee listings, arXiv, and Scholar across the query set in the brief. Justia's Match Group assignee pages returned HTTP 403 to automated fetch; **a manual review of the full Match Group and Bumble assignee listings, plus a professional FTO search, is required before any filing decision** — absence of a hit in this search is not evidence of absence.

---

## 3. Novelty Analysis

### Element-by-element

| Claim element | Closest art | Novel? |
|---|---|---|
| (a) Per-user, per-time-of-day baseline of message response latency | US 8,831,571 (AT&T, SMS, time-of-day buckets); US 8,862,519 (IBM/Snap, per-sender + time-window) | **No.** Squarely disclosed. |
| (a′) …as a **median of log** latency, with **sparse-bucket fallback** to overall median | Log-transform is textbook psychometrics; sparse-bucket smoothing has an analogue in US 8,712,943's pseudo-counts | **Formally novel, substantively obvious** as a design choice. Cannot carry a claim alone. |
| (b) Normalize an observed latency against that same user's own baseline | US 9,137,181 (Google, overdue-vs-own-pattern); US 8,095,612; the entire Tier-4 anomaly-detection corpus | **No.** |
| (c) Exclude gaps above a threshold as non-signal | Routine session segmentation (30-min / 1-hour inactivity thresholds are standard); no exact 4-hour analogue found | **Weakly novel**, obvious as routine optimization. The *4 hours* number is arbitrary and unsupported by data in the spec — an examiner will treat it as result-effective-variable tuning. |
| (d) Use the **within-conversation distribution** of normalized deviations (median and below-baseline share) rather than raw mean | Templeton PNAS 2022 teaches the within-conversation-baseline concept; no patent found implementing it | **Novel in the patent art**, but anticipated *conceptually* by a 2022 publication. |
| (e) Emit it as a **directed pairwise** feature into a matching/ranking algorithm | US 10,540,607 (POF/Match: latency as behavioral input to a matching model) | **No, for the general idea.** Yes for the specific directed-deviation formulation. |

### Where the case actually stands

The honest reading: **no single element is novel, and the general idea ("fast repliers are more engaged, feed it to the matcher") is both intuitive and already in a Match Group subsidiary's portfolio.** The only thing that could support a claim is the *specific ordered combination* of (a′)+(b)+(c)+(d)+(e) — and combination claims of this shape face two compounding problems.

**Problem 1 — §103 obviousness.** The likely rejection writes itself: POF '607 teaches latency as a behavioral matching input; AT&T '571 teaches a per-user per-time-of-day latency baseline; IBM/Snap '519 teaches conditioning that baseline and detecting change against it; the log transform is conventional for lognormal RT data; threshold-based exclusion of inactive gaps is conventional session segmentation. A person of ordinary skill combining them would arrive at the claim with a reasonable expectation of success. Our best rebuttals are (i) *teaching away* — every cited reference uses latency to predict **future behavior of a recipient** (will they reply, when will they reply), whereas this system uses latency to measure **the affective quality of a relationship already in progress**, a different technical purpose that none of the references contemplate; and (ii) *unexpected result* — that per-hour-own-baseline normalization materially changes which dyads rank highest versus an absolute threshold. Rebuttal (ii) is the stronger one, and **it does not exist yet.** It requires measured data from the judge.

**Problem 2 — §101 / Alice, and this is the bigger one.** Stripped to essentials, the claim is: collect timestamps, compute a median, subtract, count how many are negative, use the number to sort people. That is data gathering + mathematical calculation + using the result for an economic/social purpose — the canonical *Electric Power Group* fact pattern. There is no improvement to the functioning of a computer, no unconventional data structure, no networking improvement. Art Unit 3629/3689 will issue a §101 rejection with high probability, and the mathematical-formula framing (`latency_norm = log(raw_ms) − median(...)`) actively invites it.

Mitigations, none of them strong: recite on-device extraction from an encrypted local message store as a technical constraint (ties to §10 of the architecture doc, the privacy architecture — a genuine technical limitation, not a field-of-use fig leaf); recite the sparse-bucket fallback as solving a concrete failure mode of the computation itself; recite that the baseline is fit over a history *disjoint from* the scored conversation, which is a real leakage-prevention property (§6 of the architecture doc) and the most "technical" thing in the whole mechanism.

### Strength summary

- As a **broad** claim ("latency as an engagement/compatibility signal"): worthless. Obvious, and POF '607 is there first.
- As a **narrow combination** claim (the five elements above): obtainable with effort, narrow enough that a competitor implementing per-dyad or per-population normalization, or a mean instead of a distribution, or no gap exclusion, walks around it freely.
- As a **commercial deterrent**: low. The design-arounds are cheap and numerous (see §6).
- As **defensive prior art** preventing someone else from patenting this and blocking us: **good, and cheap.** This is the highest-value use of the work.

---

## 4. Patentability Verdict

**Weak-to-moderate. File only as a narrow dependent layer inside a larger system application; do not spend a standalone filing on it.**

Reasoning:

1. The general mechanism is anticipated in substance by **US 10,540,607** (Plentyoffish / Match Group), which expressly discloses message response times as behavioral inputs to a machine-learned matching model in a dating platform. The answer to "has a major dating app already patented something functionally similar" is yes at the level of idea, no at the level of implementation.
2. Each individual normalization element is disclosed somewhere: the per-user per-time-of-day baseline in **US 8,831,571** and **US 8,862,519**, the deviation-from-own-baseline in **US 9,137,181** and the anomaly-detection corpus, the log transform in textbook psychometrics.
3. The combination is formally novel and the within-conversation directed-deviation formulation appears in no patent found — but it is taught by **Templeton et al. 2022**, a public paper, which converts the inventive step into "implement a published finding competently."
4. **§101 is the binding constraint, not §102/§103.** The claim as naturally drafted is a mathematical formula applied to social data for a business purpose.

**Recommended action, in order:**

1. **Defensive publication now.** Publish the formula, the sparse-bucket fallback, the 4-hour exclusion, and the within-conversation-distribution framing — as a blog post, an arXiv preprint alongside the judge evaluation, or an IP.com disclosure. This costs nearly nothing and permanently blocks a competitor from patenting it and asserting against us. Given the design-around surface, this captures most of the available value.
2. **Keep the tuned constants as trade secrets.** `MIN_HOUR_SAMPLES = 5`, `MAX_GAP_MS = 4h`, the choice of median over mean, the 24-bucket granularity, and — most importantly — the *learned weights* the judge places on `latency_norm_median` versus `latency_norm_fast_share` are not observable from the product and are protectable indefinitely without disclosure. A patent would force disclosure of all of it in exchange for a claim that is easy to design around.
3. **If filing anyway**, roll this in as claims 8–14 of a broader application covering the whole judge — feature extraction from real transcripts + simulation harness + Social Relations Model decomposition (§5) + group assembly (§7). The latency normalization is far more defensible as one recited step of a larger technical pipeline than as its own invention.
4. **Before any filing, run a professional FTO search** on the Match Group and Bumble Inc. portfolios. This search could not complete automated retrieval of Justia's Match Group assignee listing (HTTP 403) and did not locate a Bumble Inc. assignee listing at all. Continuations in the POF '607 family are the specific risk: a continuation filed after 2022 could have been drafted with Templeton in view.
5. **Generate the unexpected-result evidence** regardless of the filing decision: run the judge with (i) raw latency, (ii) population-normalized latency, and (iii) own-hour-baseline-normalized latency, and measure the ranking difference on the known-good/known-bad friend pairs from §5. If the difference is large, it is both the §103 rebuttal and a far better reason to believe the feature is worth having. If it is small, the patent question is moot because the feature is not carrying the prediction.

---

## 5. Draft Claims

Drafted narrow deliberately. Claim 1 recites the normalization as essential, not optional, because anything broader reads on POF '607.

### Claim 1 (independent, method)

> A computer-implemented method for generating a directed interpersonal-connection signal from an asynchronous text-message conversation, the method comprising:
>
> **(a)** accessing, from a message store, a plurality of historical messages sent by a first user, each historical message having a send timestamp, and identifying for each a reply gap comprising the elapsed time between the send timestamp of the historical message and the send timestamp of an immediately preceding message from a different participant;
>
> **(b)** **excluding**, from the identified reply gaps, each reply gap exceeding a disengagement threshold, whereby elapsed intervals attributable to sleep or activity context switches are prevented from contributing to the signal;
>
> **(c)** assigning each remaining reply gap to one of a plurality of time-of-day buckets according to the local hour of day of its send timestamp, and computing, for each time-of-day bucket containing at least a minimum number of reply gaps, a **bucket baseline comprising a median of the logarithms** of the reply gaps assigned to that bucket, thereby forming a per-user, per-time-of-day baseline latency distribution for the first user;
>
> **(d)** computing an overall baseline comprising a median of the logarithms of all remaining reply gaps of the first user, and **assigning the overall baseline as the bucket baseline for each time-of-day bucket containing fewer than the minimum number of reply gaps**;
>
> **(e)** for a target conversation between the first user and a second user, identifying a plurality of observed reply gaps in which a message by the first user answers a message by the second user, excluding those exceeding the disengagement threshold, and computing for each a **normalized latency equal to the logarithm of the observed reply gap minus the bucket baseline of the first user for the time-of-day bucket of that observed reply gap**;
>
> **(f)** deriving, from the **distribution** of the normalized latencies across the target conversation, a directed pairwise feature characterizing the first user with respect to the second user, the directed pairwise feature comprising at least one of a central tendency of the normalized latencies and a proportion of the normalized latencies that are less than zero; and
>
> **(g)** providing the directed pairwise feature as an input to a ranking model that orders candidate pairings of users,
>
> wherein the per-user, per-time-of-day baseline latency distribution of step (c) is computed over historical messages **disjoint from the target conversation**.

Notes on drafting choices. Step (b) before step (c) matters: the threshold exclusion must apply to baseline fitting as well as to scoring, otherwise the baseline is polluted by overnight gaps and the deviation is meaningless — recite it once, ahead of both. Step (d) is the sparse-bucket fallback recited as an affirmative, essential step rather than an optional refinement; it is the one implementation detail with no clean analogue in the located art. Step (f) recites the *distribution*, which is what distinguishes from every "average response time" reference. The `wherein` clause is the strongest §101 hook available: computing the baseline over a disjoint history is a concrete anti-leakage property of the computation, not an abstract idea.

### Claim 2 (independent, symmetric-asymmetry variant)

> A computer-implemented method for detecting asymmetry of interpersonal engagement between two participants in an asynchronous text conversation, comprising: performing the method of claim 1 to compute a first directed pairwise feature characterizing a first user with respect to a second user; performing the method of claim 1 with the roles of the first and second users exchanged, using a **separate per-user, per-time-of-day baseline latency distribution fitted to the second user**, to compute a second directed pairwise feature; computing an engagement-asymmetry measure from a difference between the first and second directed pairwise features; and providing the engagement-asymmetry measure to a group-assembly process that selects a subset of candidate participants for a co-located gathering.

This is the claim worth having. Directed asymmetry — A answers B faster than A's habit while B answers A slower than B's habit — is a genuinely different quantity from anything in the art, cannot be computed at all without per-user baselines, and maps to a real product decision (do not seat these two together). Nothing found comes close.

### Claim 3 (dependent, tapback/multimodal)

> The method of claim 1, further comprising: identifying reaction events in the target conversation, each reaction event associating a first user with a specific prior message of the second user and being excluded from both the identifying of reply gaps and from serving as an answered message; computing a reaction rate as a count of the first user's reaction events targeting the second user's messages normalized by a count of the second user's messages; and combining the directed pairwise feature with the reaction rate in the ranking model.

Recites the tapback handling in `real.py` — reactions are neither turns nor reply targets — which is a real implementation subtlety of iMessage-derived data and adds a platform-specific technical limitation that helps against §101.

### Claim 4 (dependent, on-device / privacy constraint)

> The method of claim 1, wherein the message store is a locally-encrypted message database resident on a user device, and wherein steps (a) through (f) are performed on the user device such that message content is not transmitted from the user device, only the directed pairwise feature being transmitted to a server hosting the ranking model.

Not novel by itself — on-device processing is everywhere — but it is the most technical limitation available and ties to §10 of the architecture doc. Include it as a fallback position for the §101 fight, not as a novelty argument.

---

## 6. Design-Around Notes

### Prior art to cite in an IDS

Citing these is mandatory under the duty of disclosure and strategically sound — better to have the examiner allow over them than have a defendant find them later:

- **US 10,540,607 B1** (Plentyoffish Media ULC) — the one that matters. Distinguish on: reply-propensity prediction for recipient selection vs. affective-quality measurement of an existing exchange; no normalization of any kind.
- **US 8,862,519 B2** (IBM → Snap Inc.) — distinguish on: sender-facing prediction of when a recipient will act vs. a dyadic affinity feature; arithmetic mean over coarse weekday/weekend windows vs. median of logs over 24 hourly buckets with a sparse-bucket fallback; no gap exclusion.
- **US 8,831,571 B2** (AT&T) — distinguish on: auto-reply transparency to a sender vs. ranking input; no deviation computed at all, the baseline *is* the output.
- **US 9,137,181 B2** and family (Google) — distinguish on: reminder generation from an overdue response vs. a signed deviation distribution; no logarithm, no median, no per-dyad direction.
- **US 8,095,612 B2** (Google) — distinguish on: message importance ranking vs. interpersonal connection between participants.
- **US 8,712,943 / 8,712,944 / 9,135,558** — cite for the time-of-day response-rate buckets with pseudo-count smoothing; distinguish on domain and on the absence of any per-observation deviation.
- **Templeton et al., PNAS 2022** — cite in the background section as the motivating finding and state plainly that the invention is the computable normalization, not the finding. Failing to cite the paper the mechanism is openly built on is both an inequitable-conduct risk and, given that the architecture doc names it, indefensible.
- **Florian, Hakobyan & Drimalla, arXiv:2605.03687** — cite affirmatively: contemporaneous researchers with 3.4M messages chose absolute bins and no normalization, which is evidence that the normalization was not the obvious path.

### Design-arounds a competitor gets for free

Be clear-eyed: this claim is porous. A competitor avoids claim 1 by doing any one of the following.

1. **Normalize per-dyad instead of per-user.** Baseline against how A usually answers *B specifically*. Arguably a better feature (it controls for relationship tenure), and it reads outside a claim reciting a per-user baseline over the user's whole history.
2. **Normalize per-population-cohort.** Baseline against all users of similar age/timezone/platform rather than the individual. Weaker signal, zero infringement.
3. **Use a mean, a geometric mean, or a quantile other than the median.** Claim 1 recites "median of the logarithms" — note a geometric mean is the exponentiated mean of logs, and a claim reciting *median* specifically does not cover it. Consider drafting (c) as "a measure of central tendency of the logarithms" and accepting the broader-hence-weaker position, or claiming both.
4. **Learn the baseline continuously instead of batch-fitting it.** An exponentially-weighted moving baseline updated per message never computes a bucket median at all.
5. **Skip the gap exclusion** and instead winsorize, cap, or model long gaps explicitly as a mixture component. Cleaner statistics, and step (b) is not met.
6. **Bucket by something other than hour of day** — by the user's detected activity state, calendar busy/free, screen-on state, or a learned circadian phase. All plausibly better, all outside the claim.
7. **Use the raw mean** of normalized latencies instead of a distributional statistic — this specifically evades step (f), which is why step (f) should be drafted with care if it is the crown jewel.
8. **Use continuous-time or point-process models.** A Hawkes process on the message stream captures the same burstiness and never computes a per-observation deviation.

Items 1, 4, and 6 are the ones a competent team would land on independently, which is itself an obviousness signal worth weighing.

### What to protect instead

- **The 4-hour threshold, the 5-sample minimum, and the 24-bucket granularity** are unobservable from the product. Keep them as trade secrets; a patent discloses them for a claim that item 5 above walks around.
- **The judge's weights** on `latency_norm_median` versus `latency_norm_fast_share`, and their interaction with reply-graph weight and tapback rate, are where the actual competitive value sits (§5 of the architecture doc). None of it is patentable as such and all of it is protectable by not publishing it.
- **The directed-asymmetry formulation (claim 2)** is the one piece worth genuinely pursuing. If only one claim is filed from this document, file that one.

### Open items before any filing decision

1. Professional FTO search over the **Match Group LLC** and **Bumble Inc.** assignee portfolios, including live continuations in the POF '607 family — this research could not complete automated retrieval of the Justia assignee listings and the negative result here is not reliable.
2. Search international filings (EPO, WIPO) for the POF family, which was prosecuted internationally; **EP 4302242 A1** (ML match recommendation from empirical relationship data) surfaced in this search and should be pulled in full.
3. Run the three-way ranking comparison described in §4 item 5. Without a measured effect, there is no unexpected-result argument and no reason to spend money on prosecution.

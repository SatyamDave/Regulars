# Technical Architecture v1

Companion to `product-plan.md`. This is the buildable spec.

---

## 0. System overview

```
voice interview ──> PERSONA (structured behavioral spec)
                        │
                        ├──> SIM HARNESS (persona × persona → transcript)
                        │           │
                        │           └──> FEATURE EXTRACTOR ──> JUDGE ──> ŷ (predicted fit)
                        │
real group chat ──> FEATURE EXTRACTOR ──────────────────────────────────┐
                                                                         │
post-dinner DMs ──> GROUND TRUTH y ──────────────────────────────────────┤
                                                                         │
                                        CALIBRATION + RETRAIN <──────────┘
                                                    │
                                        GROUP ASSEMBLY (optimizer)
```

Two feature extractors, one codebase. Simulated transcripts and real chat transcripts go through the same pipeline. That symmetry is what lets you check the sim against reality at the feature level, not just the outcome level.

---

## 1. Data model

```sql
users(id, phone_hash, zip, created_at, status, exploration_flag)
persona_versions(id, user_id, version, spec_json, source, created_at)
  -- source: 'voice_interview' | 'chat_behavior' | 'chatlog_import'
  -- never mutate; always append a new version. Personas must be
  -- reproducible at the time a prediction was made.

sims(id, user_a, user_b, persona_ver_a, persona_ver_b,
     scenario_id, seed, transcript_json, created_at)
sim_features(sim_id, feature_name, value)
predictions(id, user_a, user_b, judge_version, y_hat,
            y_hat_calibrated, created_at)

groups(id, zip, created_at, assembly_method, status)
  -- assembly_method: 'model' | 'random'  <-- the exploration arm
memberships(group_id, user_id, joined_round, left_round, exit_reason)

plans(id, group_id, proposed_at, locked_at, venue, starts_at)
rsvps(plan_id, user_id, state, deposit_intent_id, paid_at)
attendance(plan_id, user_id, attended, source)

messages(id, group_id, user_id, sent_at_ms, body, reply_to,
         tapback_target, char_count)
ratings(plan_id, rater_id, ratee_id, would_see_again,
        felt_connected_1_7, submitted_at)
  -- DIRECTED. a→b is a different row from b→a. Never collapse.
```

**Three rules that matter more than the schema:**

1. **Personas are immutable versions.** If you overwrite a persona, you can no longer reproduce why the model made a call last Tuesday. Your entire moat is the pairing of prediction to outcome; losing reproducibility destroys it.
2. **Ratings are directed.** Kenny's Social Relations Model needs a→b and b→a separately. Collapsing them throws away the actor/partner decomposition, which is the part that is actually predictable.
3. **`assembly_method` exists from day one.** Explained in §6. Without it your labels are biased and the correlation you report is meaningless.

---

## 2. Persona

Not embeddings. A structured text spec fed into context. Embeddings compress away exactly the idiosyncrasies you need.

```json
{
  "conversational": {
    "opener_style": "...",
    "avg_turn_length": "short|medium|long",
    "humor_register": "dry|absurdist|self-deprecating|none",
    "question_tendency": "high|low",
    "interrupt_tendency": "...",
    "silence_tolerance": "..."
  },
  "content": {
    "unprompted_topics": ["..."],
    "avoided_topics": ["..."],
    "escalates_on": ["..."],
    "deflects_on": ["..."]
  },
  "social": {
    "energy": "initiator|responder|observer",
    "disclosure_speed": "fast|slow",
    "friction_points": ["..."],
    "drains": ["..."]
  },
  "stated": {
    "exclusions": ["..."],
    "seeking": "..."
  },
  "provenance": {"sources": [...], "confidence_by_field": {...}}
}
```

**`friction_points` and `drains` are load-bearing.** Without explicit negative space, two LLM personas will be pleasant to each other and your sim will have no variance (§4).

### Base model

**Centaur** (`marcelbinz/Llama-3.1-Centaur-70B`, open weights). Finetuned on Psych-101: 10M+ human choices, 160 experiments, 60k participants; outperforms domain cognitive models on held-out humans.

Use it for persona *enactment*, not scoring. Centaur is trained on decision tasks, not dyadic conversation, so it gives you human-shaped behavior under pressure, not conversational competence. Hosted frontier model for the scoring pass.

### Extraction

Voice memos → Whisper → transcript **plus prosody features** (speech rate, pause distribution, pitch variance, laughter). Do not discard the audio-derived features. Speech rate and pause structure are cheap signals that a text transcript destroys, and they are your only pre-meeting proxy for the latency signal that dominates §5.

---

## 3. Simulation harness

```
for each candidate pair (a, b):
  for scenario in scenarios:        # 3 scenarios
    for seed in seeds:              # 5 seeds
      run(persona_a, persona_b, scenario, seed) -> transcript(40 turns)
```

**Scenarios must vary in social difficulty.** A single easy scenario produces uniformly pleasant transcripts.

| Scenario | Purpose |
|---|---|
| `cold_open` | First 10 min at the table, drinks not ordered. Baseline. |
| `dead_air` | A topic has just died. Who repairs? Tests §5 repair signal. |
| `mild_friction` | A disagreement surfaces. Tests accommodation vs. escalation. |

**Anti-sycophancy measures** (see §4 for why this is the top technical risk):
- Temperature 0.9 to 1.0, not 0.3. You want variance.
- System prompt explicitly instructs the persona to disengage when bored and to be as uninterested as the real person would be.
- Inject `friction_points` into the scenario for at least one seed per pair.
- Never let both agents see the instruction "have a good conversation."

---

## 4. Feature extraction

Same code path for simulated and real transcripts. Real transcripts have three extra features that simulation cannot produce.

### Simulated + real

**Follow-up question rate.** Huang, Yeomans, Brooks, Minson and Gino (JPSP 2017) trained an NLP follow-up-question detector and applied it to speed dating: daters asking more follow-ups elicited more second-date agreements. Independently audited in 2025; conclusions held. Classify each turn: no question / new-topic question / follow-up / mirror question / rhetorical. Feature = follow-ups as a share of that speaker's turns.

**Linguistic style matching.** Per Ireland and Pennebaker, over nine function-word categories (auxiliary verbs, articles, common adverbs, personal pronouns, impersonal pronouns, prepositions, negations, conjunctions, quantifiers):

```
LSM_c = 1 - |p_a,c - p_b,c| / (p_a,c + p_b,c + 0.0001)
LSM   = mean over the nine categories
```

Also compute turn-by-turn LSM trajectory, not just the aggregate. **Slope matters more than level:** convergence over a conversation is the signal; two people who happen to start similar is not.

Critical caveat, hold it honestly: Ireland et al. (2011) found LSM in speed dating predicted romantic interest and relationship stability, but Niederhoffer and Pennebaker found LSM unrelated to participant- and judge-rated interaction quality, and argued it indexes engagement rather than rapport. **Feature, never target.**

**Topic survival.** Segment into topics; measure turns-per-topic distribution and share of topics surviving past three turns. Long-tailed distribution = flow. Uniform short = interview mode.

**Disclosure reciprocity.** Score each turn's self-disclosure depth 0 to 3. Feature = correlation of a's depth with b's depth at lag 1, plus the asymmetry (did one person carry it).

**Repair rate.** Detect flat moments (topic death, one-word turn, deflection). Measure who recovers and how fast. This is what `dead_air` exists to elicit.

**Question balance.** Gini over who asks. Heavy imbalance means interview, not conversation.

### Real chat only (the irreplaceable ones)

**Response latency.** The strongest finding available. Templeton et al. (PNAS 2022): faster response times predicted enjoyment and connection across conversations, within conversations, and at the partner level; responses under 250ms preclude conscious control, making them an honest signal that even outside observers read.

Text-chat latency is noisy in ways spoken latency is not. Normalize before use:

```
latency_norm = log(raw_ms)
               - median(log latency of this user, this hour-of-day)
```

Drop any gap over 4 hours (that is sleep or work, not disengagement). Use the **within-conversation** distribution, not the mean: connected moments are faster than that same conversation's own baseline, which is what Templeton found and what makes the normalization defensible.

**Reply graph.** Who responds to whom, directed and weighted. This is the real social graph inside a group of six, and it diverges sharply from the roster.

**Tapbacks.** Cheap, high-frequency affect signal. Directed.

---

## 5. The judge

### Build order: judge first, sim second

Take 5 to 10 friend pairs where you already know the answer. Run extraction over their real chat history. If the feature set does not rank known-good above known-bad, no amount of simulation fixes it. One weekend, zero dollars.

### v1: transparent, not learned

```
y_hat = w · z(features)
```

Hand-set weights, z-scored features. Do **not** fit a model on 50 labels. You have more features than data and you will fit noise. Interpretability now is worth more than accuracy, because you need to know which signal is carrying the prediction.

### v2 (from ~1,000 labeled directed dyads): Social Relations Model

This is the right statistical frame and almost nobody in this category uses it. Kenny's SRM decomposes round-robin rating data into:

- **actor** variance: how positively this person rates others
- **partner** variance: how positively this person is rated
- **relationship** variance: the dyad-specific residual

This mirrors the Joel/Eastwick/Finkel decomposition exactly, where actor and partner were predictable and relationship was not. Fitting SRM tells you **which component your model is actually predicting.**

This is the most important diagnostic you will run. If your sim only predicts the partner effect (some people are just likeable), you have built a charisma detector, not a matching engine. Useful, but a totally different product, and you need to know which one you have before you pitch it.

Your dinners are natural round-robins: six people, everyone rates everyone, 30 directed dyads per dinner. That is exactly the design SRM was built for. Use the `TripleR` R package or port it.

### Calibration

Raw judge scores are not probabilities. Fit isotonic regression from `y_hat` to observed `P(would_see_again)`. Report calibrated values only. An uncalibrated 0.8 means nothing to you or an investor.

---

## 6. The evaluation design (read this twice)

This section is the difference between a real result and a number you fooled yourself with.

### Problem 1: selection bias

You only observe outcomes for pairs you chose to put together. If the model picks the pairs and you then score the model on those pairs, the correlation is contaminated. This is the single most common way ML-driven matching companies produce fake results.

**Fix: a permanent randomized exploration arm.** From week one, ~30% of group assemblies are random (`assembly_method = 'random'`). Those groups are your unbiased evaluation set. The other 70% are model-assembled and serve the users.

This also gives you the only comparison that matters commercially:

> Does round-2 retention differ between model-assembled and randomly-assembled groups?

That is an A/B test with a real business outcome, and it is far more convincing to an investor than a correlation coefficient.

### Problem 2: leakage

Split by **person**, never by pair. If Priya appears in both train and test, the model learns Priya's partner effect and you score her twice. Group-aware cross-validation, grouped on user_id.

### Problem 3: power

Per dinner of six: 15 undirected pairs, 30 directed ratings. Ten groups a week is 300 directed observations per week; six weeks is roughly 1,800.

Detecting ρ = 0.35 at n = 300 is comfortably powered. But observations are clustered within person and within group, so effective n is materially lower than raw n. **Report the correlation with clustered standard errors, or a mixed model with random intercepts for rater, ratee and group.** A raw Pearson r on clustered data overstates precision, and a good investor's technical diligence partner will catch it.

### Problem 4: the sim collapses to agreeable

The top technical risk. Two LLMs role-playing people are pleasant to each other and score everything highly.

**Diagnostic, run before anything else:** compute the standard deviation of `y_hat` across pairs. If it is near zero, the sim has no discriminative power and the mean is irrelevant. **Watch variance before mean.**

**Second diagnostic:** AUC on your known-pairs dev set. Under 0.6 means the harness is broken, not that everyone is compatible.

### The success bar

| Metric | Target | Meaning |
|---|---|---|
| Spearman ρ (sim score vs. real rating), exploration arm only, clustered SEs | > 0.35 | Something nobody else has |
| Variance of y_hat across pairs | non-degenerate | Sim discriminates at all |
| SRM relationship-variance explained | > 0 and significant | You are predicting fit, not just charisma |
| Round-2 retention, model arm vs. random arm | +15pp | The business case |

---

## 7. Group assembly

Group score is the **minimum** pair score, not the mean. One bad dyad kills a table of six.

Partitioning n candidates into groups of six maximizing the minimum within-group pair score is a max-min set-partitioning problem and is NP-hard. At your scale you do not need to care:

1. Prefilter to plausible pairs (zip, availability, stated exclusions). Never simulate all pairs.
2. Greedy seed: pick the highest-scoring pair, grow the group by whichever candidate maximizes the resulting minimum.
3. Local search: swap members between groups while the global minimum improves. 200 iterations.
4. Hard constraints as filters before the optimizer, never as score penalties.

Add a **continuity bonus** for keeping surviving members together across rounds. The product thesis is a converging group, not a fresh table. The optimizer must encode that or it will churn groups every week and you will have built 222.

---

## 8. Cost model

All-pairs is quadratic and will bankrupt you at exactly the moment things start working. At 10,000 users, all-pairs is ~50M sims.

| Lever | Effect |
|---|---|
| Prefilter to ~200 candidates per user | 50M → 1M |
| 3 scenarios × 5 seeds only on the top 40 after a cheap first pass | order of magnitude |
| Two-stage: Haiku-class for the first pass, frontier model on the top 40 | ~10x on the dominant cost |
| Cache persona prefix across all sims for a user | large, and nearly free to implement |
| Recompute only on persona version bump or new candidates | amortizes to near zero in steady state |

At 60 users with prefiltering, this is under $50 total. Build the prefilter anyway. Retrofitting it under load is how you get a surprise five-figure bill.

---

## 9. Agent runtime

Five tools. Resist adding a sixth.

```
send_message(group_id, body)
create_plan(group_id, options[], deadline)
collect_deposit(plan_id, user_id, amount)
remove_member(group_id, user_id, reason)
log_signal(user_id, signal_type, value)
```

**The plan is a structured record, not a message.** `{proposer, constraints, candidate_slots, votes, commitment, stake, outcome}`. The group chat is a *rendering* of that object. If the source of truth is message text, you have a chatbot. If it is the object, you have a protocol with a chat frontend, and the transport becomes swappable.

**Transport abstraction from day one.** One `Transport` interface; BlueBubbles behind it now. iMessage has no public bot API, and the Mac-server providers are gray market that Apple can cut off. When that happens you want a one-day swap, not a rewrite.

**Every bot decision writes an audit row.** Why this group, why this removal, which persona version, which judge version. Needed for reproducibility (§1), for debugging, and for the day a user asks what the bot knows about them.

---

## 10. Privacy architecture

Not a compliance chore. It is a product surface and a diligence gate.

- **First-party and consented only.** No scraping of non-users. Building profiles on people who never signed up drags you toward California's data-broker regime, which is now live and enforced. You are probably not a broker if you never sell data, but "probably" is a legal opinion you should buy for an hour rather than assume.
- **Chat-log import is opt-in, revocable, never the gate.** Encrypt at rest with a per-user key. Derive persona fields, then discard raw logs on a short TTL. Store the derived spec, not the corpus. Reduces breach surface by an order of magnitude and costs you nothing analytically.
- **Legibility as a feature.** A `what do you know about me` command that prints the persona in plain English and lets the user correct it. Corrections are free high-quality labels, and legibility is the difference between "thoughtful" and "creepy" in the article someone eventually writes.
- **Removals are explainable.** A user who asks why they were dropped gets a real answer. The audit row already exists.

---

## 11. Build order

| | Ship | Gate |
|---|---|---|
| 1 | Feature extractor + judge v1, on known friend pairs | Ranks known pairs correctly |
| 2 | Bot: 5 tools, plan object, transport abstraction, deposits | Stranger → locked paid plan |
| 3 | Ratings, latency logging, prune/backfill, audit trail | Composition changes after a dinner |
| 4 | Sim harness + variance diagnostic | σ(y_hat) non-degenerate, AUC > 0.6 on dev set |
| 5 | Exploration arm (30% random) turned on | Unbiased labels accruing |
| 6 | Calibration + SRM decomposition | Know which variance component you predict |

**Steps 1 and 4 are the only ones that can kill the thesis, and both are cheap.** Everything else is engineering. Do them early and in that order: judge before sim, variance check before you trust a single score.

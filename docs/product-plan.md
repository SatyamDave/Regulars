# Regulars — Product Plan v1

## 1. The problem

Moving to a new city does not leave you with zero people. It leaves you with the wrong people. Coworkers, roommates, friends-of-friends, the guy from the gym. You have a calendar that looks social and a life that feels lonely, and admitting that sounds ungrateful so nobody says it out loud.

Three things are broken:

1. **Finding.** No mechanism exists for meeting people who actually fit you. Apps optimize for volume of matches, not quality of fit.
2. **Planning.** Adult friendship dies in logistics. Six people, five calendars, nobody wants to be the one who nags.
3. **Pruning.** Groups have no way to remove someone. It is socially expensive, so instead the whole group quietly dies.

Every existing product attacks #1 and ignores #2 and #3. That is why they produce first meetings and not friendships.

## 2. The insight

**Chemistry between two specific people cannot be predicted from what they say about themselves.**

Joel, Eastwick and Finkel (2017) ran machine learning over 100+ self-reported traits from hundreds of speed daters. They decomposed the outcome into three parts:

- **Actor effect** (does this person tend to like others): predictable
- **Partner effect** (do others tend to like this person): predictable
- **Relationship effect** (do these two click): essentially unpredictable

Every app in this category, including 222, is built on the premise that a good enough questionnaire produces compatible tables. That premise failed in the literature nine years ago.

**So stop trying to predict better. Win on trial rate instead.**

If dyadic fit can only be found by testing, the winner is whoever runs the most trials at the lowest cost and prunes fastest.

- 222 runs one trial per user every two weeks at $22 each, because a human curates and a venue gets booked.
- We run thousands per night for cents, in simulation, then one real trial per week.

We are not building a better predictor. We are building a search algorithm with cheap steps.

## 3. The product

### Onboarding (5 minutes, in iMessage)

User texts a number. No app, no download.

The bot conducts a **voice interview**. Three prompts, ninety seconds each:

1. Walk me through your last genuinely good weekend.
2. Tell me about someone you stopped hanging out with, and why.
3. What are you actually looking for right now?

Freeform voice, not a form. Voice carries pacing, humor timing, what they raise unprompted, what they skip. This is the Stanford methodology (Park et al., 2024, *Generative Agent Simulations of 1,000 People*), which validated that agents built from two-hour interviews replicate the real person's behavior.

Optional deep mode, opt-in and revocable: share your AI chat export for a better persona. Strong actor-level signal. Not required, never the gate.

### Placement

Bot simulates this person against every plausible candidate in their zip code, overnight. Scores each pair. Assembles a group of six where **the worst pair is still good** (group score = minimum pair score, not average — one bad person kills a table of six).

### The group chat opens

Bot introduces all six in one message with a specific hook per person, then immediately proposes three dated options. It does not say "get to know each other." That is how group chats die. It goes straight to a plan.

### Lock

Poll closes in 24h. Bot picks the winner, books it, collects **$20 deposits**. No payment in 24h means you are dropped and backfilled. This is the flake filter and it runs from day one.

### Meet

Real dinner. Six people. No staff, no conversation cards, no curation. The group owns the night.

### The engine

Next morning, the bot privately DMs each person: who would you see again, who was off, was it worth it. Private, so people are honest.

Then it acts. Drops the people who did not land. Adds new ones from the queue. The bot absorbs the social cost of exclusion, which is the thing humans structurally cannot do to each other.

### Convergence

Repeat weekly. By round five the group has stabilized and nobody wants to leave, because leaving means restarting the search.

**Success condition: the group graduates.** They start planning without the bot. That is the win, and it is the best marketing we will ever have.

## 4. The technology

Four layers. The moat is the fourth, not the first.

### Layer 0 — Persona

Behavioral spec, not a profile. Built from voice interview, then continuously updated from real group chat behavior and post-dinner DMs.

Base model: **Centaur** (open weights, Llama finetune, Nature 2025) trained on Psych-101: 10M+ human choices across 160 psychology experiments, 60,000 participants. Beats existing cognitive models on held-out humans. Gives personas human-like behavior under decision pressure instead of the default agreeable-LLM voice.

### Layer 1 — Simulation

Two persona agents, specific scenario ("first ten minutes at a dinner table, drinks not ordered"). 40 turns, 5 runs with different openers to measure variance.

**Failure mode to watch: if every pair scores highly, the sim has no discriminative power. Watch variance before mean.**

Cost control: never simulate all pairs. Prefilter with cheap heuristics (geography, availability, stated exclusions) down to ~200 plausible pairs per user.

### Layer 2 — Features

Computable, evidence-backed signals. Not "did they gel, 1-10" — that returns 7 every time.

| Signal | Evidence |
|---|---|
| **Response latency** | Templeton et al., PNAS 2022. Faster responses predicted enjoyment and connection across conversations, within conversations, and at partner level. Sub-250ms precludes conscious control, so it is an honest signal. Strongest single finding in the literature. |
| **Follow-up question rate** | Huang, Yeomans, Brooks, Minson, Gino, JPSP 2017. Speed daters who asked more follow-ups got more second-date agreements. NLP follow-up detector already exists in the paper. Independently audited 2025, conclusions held. |
| **Linguistic style matching** | Ireland et al., Psych Science 2011: LSM in speed-dating predicted romantic interest and relationship stability. Caveat: Niederhoffer and Pennebaker found LSM unrelated to rated interaction quality and argued it indexes engagement, not rapport. **Use as a feature, never as a target.** |
| Topic survival | Threads lasting past three turns |
| Disclosure reciprocity | Did both escalate, or did one carry it |
| Repair | After a flat moment, who recovered it |

### Layer 3 — The judge

Composite score over the features. Group score = worst pair.

**Build the judge before the sim.** Take five friend pairs where the answer is already known. Run feature extraction on their real chat history. If the score does not rank them correctly, nothing downstream works.

### Layer 4 — The label (this is the moat)

Every dinner produces ground truth: sim predicted 0.8, reality said no.

Plus the one signal simulation can never produce: **real reply latency from real humans in the real group chat.**

The mapping from simulated text features to real-world outcomes is a dataset that does not exist anywhere and cannot be downloaded, bought, or scraped. It only accrues to whoever runs the dinners.

**Key metric: correlation between sim score and real post-meeting rating.**
- Above 0.35 → something nobody else has
- Below 0.15 → an expensive random shuffler, and we know in week 6 instead of month 18

## 5. The moat, ranked

1. **Labeled outcomes.** Compounds. Cannot be acquired without running dinners.
2. **The judge.** Anyone can run two LLMs at each other. Almost nobody can build a scorer that correlates with reality, because that requires #1.
3. **Density.** Live zip codes. Least technical, most durable.

Explicitly **not** moats: the models (commodity), open source tooling (same-day download), the idea (describable in a tweet), design polish (cheap in 2026).

Any fine-tune we eventually run is the *output* of the moat, not the moat.

## 6. Business model

- **Now:** $20 deposit per plan, redistributed to people who showed. Small platform cut.
- **Later:** take rate on everything the bot books. Restaurants, tickets, activities, travel. We sit upstream of the reservation instead of competing for it. OpenTable without the search, because the booking is created before the decision is made.

Stack cost at 60 users: roughly $30/month. BlueBubbles on existing hardware, Supabase, Railway, Claude API, Stripe (manual-capture PaymentIntent, no cash payouts — credit toward next plan avoids money transmission).

## 7. Go to market

**One vertical, one zip code.** Indians in SF tech who moved in the last 18 months. Founder is the ICP and is one degree from hundreds of them.

Density is per-neighborhood, not global. A viral spike across 90 cities produces zero usable groups. Meetup has 60M members and never compounded, for exactly this reason.

**So: story goes wide, product goes deep.**

Growth loops, ranked:

1. **The deposit argument.** "We charge $20 to show up. Flake and the people who came split it." That is a debate, not a feature. Fights travel.
2. **The receipt.** Post-dinner recap to the group, designed to be screenshotted. Six people, weekly, free.
3. **Graduation announcements.** "Group 41 graduated after 6 weeks." Counterintuitive, unfakeable, proves the thesis.
4. **Referral unlock.** Waitlist moves only when you bring two people within a mile. Converts hype into density instead of decay.
5. **Published numbers.** Show rate, round-2 attendance, sim-vs-reality correlation. Competitors refuse to publish real numbers. Rigor as positioning.

Not doing: TikTok, influencers, multi-city launch, paid ads, Product Hunt before retention data.

## 8. Positioning

Not "AI friendship app." Every competitor already claims AI matching and anti-attention-economy virtue. That is table stakes, not differentiation.

**"We measure the thing nobody can fake."**

Public line: chemistry cannot be predicted from a personality quiz, so we stopped trying. We simulate the conversation, then check ourselves against what actually happened in the room.

Landing page copy is **restrained**, design is **obsessive**. Understate, then overdeliver at the dinner. Consumer social gets one shot per friend group; the gap between promise and reality becomes the story.

## 9. Build sequence

| When | What | Done when |
|---|---|---|
| Week 1 | 60 individual texts, one zip code. Build the judge on known friend pairs. | 40 yes. Judge ranks known pairs correctly. |
| Weekend 1 | Bot ships. 5 tools: send, create_plan, poll, collect_deposit, remove_member. | A stranger texts the number and ends up in a locked, paid plan. |
| Weekend 2 | Feedback loop. Morning-after DMs, prune/backfill, latency logging. | Group composition changes automatically after a dinner. |
| Weekend 3 | Sim on Centaur personas. Score all plausible pairs. **Match randomly anyway.** | Correlation measured against real outcomes. |
| Week 6 | **Gate.** | See below. |
| Week 7 | Landing page, zip-locked waitlist, referral unlock. | Backed by a real number. |
| Week 8 | Raise. | — |

### The week 6 gate

| Metric | Go | Stop and fix |
|---|---|---|
| Round-2 attendance | 60%+ | Under 30% → matching is broken |
| Sim vs. reality correlation | 0.35+ | Under 0.15 → thesis needs rework |
| Show rate | 85%+ | Under 70% → deposit is not working |
| Groups alive at week 6 | Any | Zero → product is wrong |

**Round-2 attendance is the only number that matters.** Show rate measures the deposit. Round-2 measures whether people liked each other.

Do not raise on a broken loop. It buys 18 months to fail publicly.

## 10. The investor story

Four beats, in order:

1. **The dead premise.** The category's core assumption was falsified in 2017. Every competitor is built on it.
2. **The reframe.** Stop predicting, win on trial rate. Thousands of simulated trials per night vs. one real trial per fortnight.
3. **The wedge.** Behavioral signals nobody collects (sub-second latency, follow-up rate, style convergence) plus outcome labels from real dinners.
4. **The number.** Sim-vs-reality correlation, whatever it is.

## 11. Risks and how they kill us

| Risk | Kill condition | Mitigation |
|---|---|---|
| Sim has no discriminative power | Every pair scores the same | Measure variance before mean. Build judge on known pairs first. |
| Cold start | Empty feed, no groups form | Hand-seed. Founder personally assembles first groups. |
| Deposit friction | People bail at payment | Measure deposit conversion separately. Test $10 vs $20. |
| iMessage platform risk | Apple kills the gray-market providers | Abstract the send layer from day one. Swap to WhatsApp in a day. |
| Privacy backlash | "AI reads your chat to decide who to kick" | Make signals user-visible from day one. Legible beats creepy. |
| Data provenance | Scraped or uploaded data poisons diligence | First-party and consented only. Chat-log sharing is opt-in, revocable, never the gate. |
| No network effects | Density is per-neighborhood, not global | Zip-locked growth. Prove the affinity graph transfers before scaling cities. |

## 12. The long game

Nobody owns the **affinity graph**: who actually works with whom, validated by repeated real-world contact and honest post-hoc feedback. It is the only one with ground truth, because every edge gets tested at a dinner table and rated the next morning.

Friendship is the entry point because it is the only place people will honestly tell you who they liked.

**Mission: rebuild the infrastructure that produces strong ties, after the third place died.**

## Open items

- **The bot's voice.** The single highest-leverage unbuilt thing. It is writing, it is free, and it is the entire perceived product.
- **First-message copy.** The first thing a user ever receives from the bot.
- **Does the Joel/Eastwick/Finkel result replicate for platonic ties?** If yes, it is also a fundraising slide.

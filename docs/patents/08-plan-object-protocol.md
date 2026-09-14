# 08 — Plan-Object Protocol: Structured Commitment Record with Chat as Rendering

Prior-art and patentability assessment for the Regulars agent runtime (`docs/technical-architecture.md` §9).

Prepared: 2026-09-14. Searches run against Google Patents, USPTO full-text (image-ppubs), Justia, arXiv, and general web. This is an engineering-grade prior-art read, not a legal opinion or a freedom-to-operate clearance.

---

## Mechanism Summary

### What the code actually does

The mechanism under review is implemented in three files:

- `regulars/bot/plan.py` — the `Plan` record and its transitions
- `regulars/bot/transport.py` — the messaging `Transport` seam
- `regulars/bot/tools.py` — the five-tool action surface

**1. The commitment record is the source of truth.** `Plan` is a frozen Pydantic model carrying the complete state of a negotiated multi-party dinner commitment:

```
{id, group_id, proposer, constraints, candidate_slots[], votes{user→slot},
 commitment{user→invited|deposit_requested|paid|dropped}, stake_cents,
 deadline, outcome{open|locked|cancelled}, locked_slot_id, poll_closed_at,
 deposit_requested_at, deposit_intents{user→payment_intent}, locked_at}
```

**2. Every state change is a pure function `Plan → Plan`.** `vote`, `close_poll`, `mark_deposit_requested`, `mark_paid`, `drop_unpaid`, `lock`, `cancel` all return a new frozen record via `model_copy`. There is no mutation and no branch that reads message text to decide outcome. Poll resolution is a deterministic rule, not a model call: plurality of `votes`, ties broken by earliest `starts_at`, and with zero votes the earliest slot still wins.

**3. Chat messages are a projection of the record.** `render_proposal`, `render_poll_closed`, `render_lock`, `render_receipt` are pure functions of `(Plan, names)` plus, for the receipt, the attendance outcome. They are the only way plan state reaches the group. Inbound chat is an *event that triggers a transition*, never the store of record — `parse_bluebubbles_webhook` normalizes a provider payload into an `InboundMessage`, the router maps it to a transition, and the transition decides what is true.

**4. The financial commitment is inside the record, not a side system.** `stake_cents` (default $2000 = $20), a 24-hour `DEPOSIT_WINDOW`, per-member `deposit_intents` holding the payment-processor intent id, and a `commitment` state machine where `paid` is a precondition of `lock()`. `drop_unpaid(plan, now)` is the enforcement transition: after the window closes, every member still in `deposit_requested` moves to `dropped` and is returned as a list, which `flows.py` turns into a `remove_member` call against the group chat plus a `BackfillRequest`. `lock()` refuses to fire unless `all_paid()` over the surviving members. Forfeited stake is redistributed in the receipt: `share = stake_cents * len(forfeited) // max(len(attended), 1)`.

**5. Transport is an abstract seam.** `Transport` is a five-method `Protocol` (`send`, `create_group_chat`, `send_dm`, `add_participant`, `remove_participant`). The `Plan` object model imports nothing from it. `BlueBubblesTransport` and `FakeTransport` are interchangeable implementations; BlueBubbles/iMessage payload knowledge is confined to `transport.py` and the ingest module.

**6. The action surface is hard-capped at five tools.** `TOOL_NAMES = ("send_message", "create_plan", "collect_deposit", "remove_member", "log_signal")`, with a module docstring reading "Resist adding a sixth." Every tool writes exactly one audit row (`actor`, `decision_type`, `reason`, `payload`, `subject_user_id`, `group_id`, `plan_id`) through the store before returning. `collect_deposit` validates plan state and member state *before* calling the payment collector, so no hold is opened that cannot be recorded.

### The combination being assessed

> A group-coordination agent in which a multi-party commitment with a per-participant monetary stake is held as a structured record; all state changes are deterministic pure functions on that record; every chat message is a deterministic rendering of that record; non-payment within a fixed window deterministically removes the participant from the underlying messaging group and emits a backfill request; the messaging provider sits behind an abstract transport interface; and the agent's entire capability is a fixed five-function set, each of which writes an audit row.

---

## Prior Art Found

### A. State-machine-backed chatbots — heavily anticipated, oldest and most damaging

| Reference | Date | Owner | Relevance |
|---|---|---|---|
| [US7454469](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7454469) — *Method and system for instant messaging Bots specification using state transition methodology and XML* | filed ~2003 | IBM | **Closest art to the core architecture.** Bot operation is defined by a state machine whose states are declared in XML under a "Bot Transition Definition Language." An IM bot driven by an explicit external state definition, with messages emitted per state. This alone anticipates "chatbot whose behavior is a state machine separate from message text." |
| [US8074199B2](https://patents.google.com/patent/US8074199B2/en) — *Unified messaging state machine* | 2007 | Microsoft | "The user is always in one conversational state, or dialog, at a time, and each dialog determines the next dialog to transition to." |
| [US20150019662A1](https://patents.google.com/patent/US20150019662A1/en) — *Notification Message Generation* | 2013 | — | Message templates with parameter placeholders filled from structured per-recipient data — i.e., deterministic rendering of a message from a record. |
| [EP3378204B1](https://patents.google.com/patent/EP3378204B1/en) — *Bot interaction* | 2015 | Microsoft | Bots maintain "state data structures that maintain the state of dialogs"; natural-language templates with placeholders filled from dialog-specific content. |
| [US20240394285](https://patents.justia.com/patent/20240394285) | 2023 | — | Modern LLM-era chatbot patent separating factual source data from generated text. |

**Honest read:** "chat is a view over a structured state object" is a decades-old, well-documented pattern, and the IBM 2003 filing plus the Microsoft dialog-state filings put it squarely in the prior art. It cannot be claimed at the architectural level.

### B. Group-based communication platforms with structured objects and workflows

| Reference | Owner | Relevance |
|---|---|---|
| US11636260 — parsing/formatting messages into a **structured message object** for rendering in a group-based communication interface | Slack Technologies | Directly on "structured object rendered as a chat message." |
| US11706043 — rendering **workflow interface objects**, detecting a workflow trigger event, retrieving an operational sequence set by workflow identifier, initiating it | Slack Technologies | Directly on "inbound chat event triggers a transition in a stored workflow object." |
| US11763258 — generating documents from workflows; structured input drives a workflow step whose result is saved into the document | Slack Technologies | Structured input → workflow step → persisted result. |
| [US11704098](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11704098) / [US12260192](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12260192) — *Workflow application and user interface builder integrating objects, relationships, and actions* | Salesforce/Slack | Object model + actions as the unit of workflow. |
| [US11943180B2](https://patents.google.com/patent/US11943180) — draft message object collaboration | Slack Technologies | Message-as-object in a group platform. |

Slack's portfolio is the most commercially dangerous cluster for anything phrased as "group-based communication platform + structured object + workflow trigger." Claim language should avoid those terms of art.

### C. Multi-party scheduling / poll negotiation

| Reference | Date | Owner | Status | Relevance |
|---|---|---|---|---|
| [EP2724298A1](https://patents.google.com/patent/EP2724298A1) — *A communication platform for iterative multiparty convergence towards a microdecision* | prio. 2011-06-23 | GIG Technology NV (orig. Mobicage) | **Ceased** | **The single closest scheduling reference.** A structured, modifiable message object with predetermined response options inside a messaging platform; participants iteratively change selections; live aggregated display of who picked what; explicitly exemplified by *"Where are we going for dinner?"* This is the vote/candidate-slot half of the Plan, in chat, from 2011. |
| [US8180663B2](https://patents.google.com/patent/US8180663B2/en) / [US20060293943A1](https://patents.google.com/patent/US20060293943) — *Facilitating automated meeting scheduling* | 2005 | — | Constraint input, candidate-time evaluation, suggested times. |
| [US20140229560A1](https://patents.google.com/patent/US20140229560) — *Appointment negotiation systems and methods* | 2013 | — | Two-or-more-party negotiation over an appointment time and attributes. |
| [US8965972B2](https://patents.google.com/patent/US8965972) | — | — | Automated planning of a meeting among ≥2 participants from calendar data. |
| [US8346590](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8346590), [US9760870](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9760870) | — | — | Scheduling/rescheduling and event scheduling generally. |
| [US8817663](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8817663) — ad-hoc groups linked to an event spanning multiple communication modes | — | — | Event-scoped group creation across channels. |

No Calendly-assigned patent surfaced; Calendly's moat appears to be product/brand rather than patent. No x.ai (Amy/Andrew) or Doodle patent surfaced in searching — both appear to have operated without a patent position on the negotiation mechanic.

### D. Group coordination combined with payment

| Reference | Date | Owner | Status | Relevance |
|---|---|---|---|---|
| [WO2020112614A1](https://patents.google.com/patent/WO2020112614A1/en) — *Event management and coordination platform* ("Smart Chats") | prio. 2018-11-26 | Tickitin Experiences LLC | **Ceased** | **Closest single reference to the overall product.** Group chat connecting user accounts, an event assigned to the group communication connection, collection of price-range preferences from participants, **cost splitting**, automatic ticket purchase / restaurant booking / taxi request. Verified against the published claims: it does **not** claim deposits, non-payment consequences, participant removal, or backfill. |
| [US8280788B2](https://patents.google.com/patent/US8280788B2) / [US20110106675A1](https://patents.google.com/patent/US20110106675A1/en) — *Peer-to-peer and group financial management systems and methods* | 2009 | — | Group financial obligations and settlement among peers. |
| [US9773237B2](https://patents.google.com/patent/US9773237B2/en) — *Synchronous split payment transaction management* | 2013 | — | Splitting one transaction across multiple payers synchronously. |
| [US8712859B2](https://patents.google.com/patent/US8712859) — configuration and incentive in an event management environment with automated segmentation of consideration | — | — | Automated splitting of consideration in event management. |
| [US9479602B1](https://patents.google.com/patent/US9479602B1/en) — event platform for peer-to-peer digital gaming competition | — | — | Entry-stake pooling and payout among participants (gaming framing). |
| [US12190310](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12190310) — automatic check-in at an event via e-wallet transaction | — | — | Attendance verification tied to a payment record. |
| [US10614479](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10614479) — *Attendance verification* | — | — | Incentivizing attendance and verifying it, with a reward message on valid attendance. |

Commercial (non-patent) prior art on the deposit mechanic is extensive and public: **Tock** built its business on prepaid/deposit reservations explicitly to suppress no-shows; **OpenTable** ships deposits, card holds, and no-show fees as productized features; **Meetup** charges member dues and *automatically removes members who have not paid after the trial period* — which is publicly documented, non-patent prior art directly on "automatic removal from a social group for non-payment."

### E. Transport abstraction

| Reference | Date | Owner | Status | Relevance |
|---|---|---|---|---|
| [US10594840](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10594840) — *Bot framework for channel agnostic applications* | prio. 2018-02-23 | West Technology Group LLC | **Active to 2038** | **Anticipates the transport-seam element outright.** Abstraction layer letting a single bot application run across IVR, SMS, web chat, and social channels; per-channel controllers sharing a common parent performing core logic; normalization of inbound and reformatting of outbound. |
| [US20040249950A1](https://patents.google.com/patent/US20040249950A1/en) | 2003 | Microsoft | Messaging layer abstracting transport protocols; channel layer above it. |
| [US10708360](https://patents.justia.com/patent/10708360) | 2020 | — | Transport-agnostic communication via a transport abstraction layer. |
| [US10896077B2](https://patents.google.com/patent/US10896077B2/en) | — | — | Vendor-agnostic messaging abstraction layer. |
| [US10659403](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10659403) | — | — | Channel-based chatbot node navigation. |

**The transport interface has no independent patentable weight.** Treat it as a described feature, never as a claim limitation carrying novelty.

### F. Constrained agent action surfaces and event-sourced agent state

No patent was located claiming "an agent limited to a fixed enumerated set of N tools." The field is instead dense with very recent non-patent literature, which is prior art under §102(a)(1) regardless of patent status:

- Least-privilege tool binding, tool allowlists, and per-tool audit logging are established security practice (Microsoft Security, OWASP AI Agent Security Cheat Sheet).
- [STARS: Skill-Triggered Audit for Request-Conditioned Invocation Safety in Agent Systems](https://arxiv.org/pdf/2604.10286) — request-conditioned gating of tool invocation.
- [Beyond Static Sandboxing: Learned Capability Governance for Autonomous AI Agents](https://arxiv.org/pdf/2604.11839) — a Capability Governor restricting agent awareness, a Safety Router intercepting every tool call, and training on audit logs to learn "the minimum viable tool set per task type." This explicitly teaches minimizing the tool surface for safety.
- [The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems](https://arxiv.org/html/2605.21997) — append-only log as source of truth, working state as a deterministic projection.
- [ESAA-Conversational](https://arxiv.org/html/2606.23752) — event-sourced memory for conversational agents: append-only log of record, derived read models.
- [OpenHands Software Agent SDK](https://arxiv.org/pdf/2511.03690) — event sourcing at the core, all interactions as immutable events.

This is the most important honest finding in this report: **"structured record is the source of truth, the visible surface is a deterministic projection, the agent's tool set is deliberately minimized and every call is audited" is, as of 2026, published state of the art in the agent-systems literature.** It is good engineering. It is not, standing alone, novel.

---

## Novelty Analysis

### Not novel (do not claim)

1. **Chatbot backed by a state machine / structured dialog state.** US7454469 (2003), US8074199B2, EP3378204B1.
2. **Rendering a chat message deterministically from a structured record via templates.** US20150019662A1, Slack US11636260.
3. **Inbound message as a trigger event for a stored workflow object.** Slack US11706043.
4. **Multi-party poll over candidate options inside a messaging platform, resolved to one outcome.** EP2724298A1 — with a dinner example, no less.
5. **Transport/channel abstraction so bot logic is provider-independent.** US10594840 (live to 2038), US20040249950A1.
6. **Group chat plus cost splitting plus automated booking.** WO2020112614A1.
7. **Deposits/prepayment to suppress no-shows.** Tock, OpenTable, Resy — massive public commercial prior art.
8. **Automatic removal from a group for non-payment.** Meetup member dues, publicly documented.
9. **Waitlist backfill of a vacated slot.** US20190318276A1, EP2775428A1, and every appointment-scheduling product.
10. **Minimizing and auditing an agent's tool surface.** Published 2026 arXiv work explicitly teaches learning the minimum viable tool set and intercepting every tool call for audit.

### Plausibly novel — the specific combination

What no located reference combines is the **closed enforcement loop that runs entirely inside one commitment record**:

**(a) The payment state is a field of the same record that holds the negotiation state, and the lock transition is gated on it.** In the prior art, negotiation lives in a scheduling system and money lives in a payment system; reconciliation is external and usually human. Here `commitment[user]` is the *same* per-user field that moves `invited → deposit_requested → paid → dropped`, `lock()` raises unless `all_paid()` over the surviving membership, and `deposit_intents` binds the processor's intent id into the record. Both the scheduling references (§C) and the payment references (§D) hold one half.

**(b) Non-payment deterministically mutates the underlying social graph, not just the booking.** `drop_unpaid` → `dropped` → `remove_member` → an actual participant removal from the messaging group, plus a `BackfillRequest` to source a replacement. Meetup removes non-payers from a *membership roster*; nothing found removes a non-payer from the *live group conversation of a specific negotiated event* as a pure consequence of a record transition, and simultaneously triggers replacement sourcing. The tight coupling of *deposit window expiry → chat-group eviction → backfill* inside one transition is the sharpest non-obvious element.

**(c) Forfeited stake is redistributed to verified attendees, and redistribution is rendered as the group-visible artifact.** `render_receipt` computes `share = stake_cents * len(forfeited) // max(len(attended), 1)` and the credit carried in `DepositCollector` reduces a future stake. Gaming-stake pooling (US9479602B1) and attendance rewards (US10614479) exist separately; peer-to-peer forfeiture-to-attendees for a social dinner, surfaced back into the same chat thread as the settlement record, was not found.

**(d) Money movement and social eviction are reachable only through a fixed, enumerated, individually audited action set.** The novelty here is not "few tools" in the abstract — that is published. It is the *pairing*: the two irreversible real-world powers (debit a person; eject a person from their friends' chat) are each exactly one function in a five-function closed set, each function writing a one-row audit record with actor, decision type, human-readable reason, subject, group, and plan before returning, and each function validating record state before touching the external effect (`collect_deposit` checks plan and member state before the collector is called, so no hold exists that is not recorded). Framed as a *safety-enforcement architecture for financially- and socially-consequential agent actions*, this is a narrower and better-supported story than "minimal tool surface."

### Obviousness exposure (§103)

The honest risk: an examiner combines **EP2724298A1** (chat poll converging on a dinner slot) + **Tock/OpenTable deposits** (deposit to hold a booking) + **Meetup dues removal** (remove non-payer from group) + **any waitlist backfill patent**, and calls the combination obvious to a person of ordinary skill. That rejection is foreseeable and must be answered up front by claiming the *mechanism-level specifics* the combination does not supply: the single record whose lock transition is gated on the per-member payment field, the deposit-window expiry as the sole trigger for both eviction and backfill, the pure-function/no-message-parsing constraint, and the audited fixed-arity action set. A claim written at the level of "collect deposits and kick non-payers" will not survive.

---

## Patentability Verdict

**Verdict: narrowly patentable at best. File only if the claims are drawn tightly to the closed record-driven enforcement loop — payment-gated lock, deposit-window-triggered group eviction plus backfill, and stake redistribution — with the audited fixed action surface as a dependent-claim safety limitation. Any claim drawn to "chat as a view over a structured state object," to the transport abstraction, or to "an agent with a minimal tool set" will be rejected on prior art.**

Supporting judgment:

- **§102 (novelty):** the specific closed loop (b)+(c) above was not located in any single reference. Survivable.
- **§103 (obviousness):** **this is the real risk.** Every ingredient exists commercially; the combination is assemblable from four public sources. Survival depends entirely on claiming mechanism, not outcome.
- **§101 (eligibility):** **this is the second real risk, and possibly the larger one.** *Recentive Analytics v. Fox Corp.* (Fed. Cir., April 2025) held that applying generic machine learning to a new data environment without improving the model itself is ineligible; the broader line is that automating a business process with generic computing stays abstract, while a specific technological improvement to computer functionality survives. "Collect deposits, remove non-payers, refill the seat" is squarely a business practice — organizing human activity plus a financial transaction, the Alice heartland. The *only* credible Step-Two argument is technological: a bounded, deterministic, replayable, fully-audited execution substrate for an agent authorized to move money and alter group membership, in which no critical state is derived from parsed natural language. Claims must be drafted so that this is the claimed improvement, in the claim body, not merely argued in the spec.
- **Commercial value:** modest as an exclusion right, real as a defensive and diligence asset. The nearest product-level reference (WO2020112614A1, Tickitin) is **ceased**, and EP2724298A1 is **ceased** — the closest-fitting art is abandoned, so a narrow grant here would sit in genuinely unoccupied space.
- **Cost/benefit:** a provisional is worth filing. A full non-provisional is worth filing only if the deposit/lock/backfill loop proves out as the retained differentiator in production (the product plan's own success metric — show rate ≥85% — is exactly the test of whether this mechanic works). Do not spend on continuations for the architectural framing.

---

## Draft Claims

Illustrative sketches for counsel, not filing-ready text. Claim 1 is the anchor; it deliberately buries the business outcome inside mechanism limitations.

### Claim 1 (independent — record-gated commitment with automatic membership enforcement)

A computer-implemented method for coordinating a multi-party commitment over a group messaging channel, comprising:

- (a) storing, in non-transitory memory, an immutable commitment record comprising a plurality of candidate time-slot entries, a vote map associating each of a plurality of participant identifiers with at most one candidate time-slot entry, a **commitment map associating each participant identifier with a per-participant commitment state selected from at least {invited, deposit-requested, paid, dropped}**, a per-participant stake amount, and an outcome field;
- (b) receiving, from a messaging transport interface, an inbound event derived from a message in the group messaging channel, and mapping the inbound event to one of a predetermined finite set of transition functions, **each transition function computing a successor commitment record as a pure function of the current commitment record and the inbound event, without deriving any field of the successor record from unstructured text of the message**;
- (c) responsive to a poll-close transition, selecting a locked time-slot entry by a deterministic resolution rule applied solely to the vote map and the candidate time-slot entries, and writing the locked time-slot entry into the successor record;
- (d) for each participant identifier, invoking a payment-authorization action for the per-participant stake amount, recording the returned authorization identifier in the commitment record, and advancing that participant's commitment state to deposit-requested, **the authorization action being invoked only after the commitment record is validated as eligible, such that no authorization exists that is not recorded in the record**;
- (e) upon expiry of a predetermined deposit window measured from the record's deposit-request timestamp, executing a single enforcement transition that (i) advances to dropped every participant identifier whose commitment state is deposit-requested, (ii) **emits, for each such participant identifier, a participant-removal instruction to the messaging transport interface causing removal of that participant from the group messaging channel**, and (iii) emits a backfill request identifying the number of vacated positions;
- (f) transitioning the outcome field to locked **only when every participant identifier not in the dropped state has commitment state paid**; and
- (g) generating each message transmitted to the group messaging channel as a deterministic rendering function of the commitment record alone, such that the group messaging channel is a projection of the commitment record and is not a store of record for any field thereof.

### Claim 2 (independent — stake settlement and redistribution)

A method as in claim 1, further comprising, after an occurrence time of the locked time-slot entry:

- receiving an attendance determination partitioning the paid participant identifiers into an attended set and a forfeited set;
- computing a per-attendee redistribution amount from the aggregate stake amounts of the forfeited set divided over the cardinality of the attended set;
- crediting each attended participant identifier with the redistribution amount as a reduction against a stake amount of a subsequent commitment record for the same group; and
- transmitting to the group messaging channel a settlement rendering generated deterministically from the commitment record, the attended set, and the forfeited set.

### Claim 3 (dependent — audited fixed-arity action surface)

The method of claim 1, wherein all external effects of the coordinating system are performed exclusively by a fixed set consisting of exactly five action functions — a message-transmission function, a commitment-record-creation function, a payment-authorization function, a participant-removal function, and a signal-recording function — and wherein each invocation of each action function writes exactly one audit record comprising an actor identifier, a decision type, a human-readable reason string, a subject participant identifier, a group identifier, and a commitment-record identifier, prior to returning; such that every debit of a participant and every removal of a participant from the group messaging channel is individually attributable to one audit record.

### Claim 4 (dependent — transport independence and replay)

The method of claim 1, wherein the commitment record and the transition functions are free of any data structure specific to a messaging provider, the messaging transport interface exposing provider-independent send, direct-message, group-create, participant-add, and participant-remove operations; and wherein, the transition functions being pure, the sequence of messages transmitted to the group messaging channel is reproducible by replaying the stored sequence of inbound events against an initial commitment record.

**Drafting notes.** Limitation (b)'s negative element ("without deriving any field ... from unstructured text") and (d)'s ordering element ("only after ... validated, such that no authorization exists that is not recorded") are the two limitations doing the most §101 and §103 work — they describe a concrete technical constraint on the system, not a business goal. Limitation (e) must remain a *single* transition producing both eviction and backfill; split across two steps it reads as ordinary automation. Avoid the phrase "group-based communication platform" throughout (Slack's term of art). Avoid "channel agnostic" (US10594840's term of art).

---

## Design-Around Notes

### Art to cite affirmatively in an IDS

Citing the strongest art and distinguishing it is better than having it found later. At minimum: **US7454469** (IBM, IM bots as XML state machines), **EP2724298A1** (Mobicage/GIG, multiparty dinner micro-decision in chat), **WO2020112614A1** (Tickitin, Smart Chat event coordination with cost splitting), **US10594840** (West, channel-agnostic bot framework), **US8280788B2** (peer-to-peer and group financial management), **US9773237B2** (synchronous split payment), **US10614479** (attendance verification with reward), Slack **US11636260** / **US11706043**, and the 2026 arXiv agent-safety and event-sourcing papers listed in §F. Also disclose the non-patent commercial art: Tock/OpenTable deposits and no-show fees, and Meetup's automatic removal of unpaid members.

### Live rights to route around

- **US10594840 (West Technology Group, active to 2038)** — the only *live* patent found that reads on an element of this system. Mitigation is straightforward: never recite the transport abstraction as a source of novelty, and do not implement multi-channel *normalization and per-channel response reformatting*, which is what its claims actually reach. A single-provider `Transport` protocol with one live implementation is outside the claimed subject matter and should be described as an engineering seam, not an invention.
- **Slack/Salesforce workflow portfolio (US11636260, US11706043, US11763258, US11704098, US12260192)** — live and well-funded. Avoid "group-based communication platform," "workflow trigger event," "operational sequence set," and "structured message object" as claim vocabulary. Regulars' object is a *commitment record with a monetary stake*, not a workflow definition, and the claims should keep that distinction visible.
- **WO2020112614A1 and EP2724298A1 are ceased** — no infringement exposure, full §102/§103 force as art. Distinguish on the enforcement loop: neither teaches a payment-gated lock, non-payment eviction, or backfill.

### Product-side design-arounds a competitor would attempt

A competitor avoids Claim 1 by breaking any one of these, so each is a place where the moat is thin:

1. **Keep the deposit outside the commitment record** (a separate payments service reconciled by a job). Defense: the dependent claims on the single-record `deposit_intents` + `all_paid()`-gated lock; and practically, the reconciliation bug surface is the reason this design wins.
2. **Do not evict from the chat** — mark the non-payer inactive but leave them in the thread. This is the easiest design-around and it is also a product weakening: the eviction is what makes the stake credible. Consider a second independent claim covering *suppression of the non-payer from the rendering* as an alternative to transport-level removal.
3. **Parse chat text as the state of record** with an LLM. Trivially outside the claims, and worse-engineered; not a real competitive threat but it does mean the claims cover only the good design, which is acceptable.
4. **Use more than five tools.** Claim 3's "exactly five" is fragile by construction. Redraft as "a fixed, enumerated set of action functions, wherein the set is closed at system-configuration time and no action function is added at runtime, and wherein each invocation writes exactly one audit record" — the *closure and audit* properties are the defensible part; the integer five is a design decision, not a limitation worth claiming literally.

### Open gaps to close before filing

- **Attendance determination is not implemented in the reviewed code.** `render_receipt` takes `attended` and `forfeited` as parameters supplied by the caller. Claim 2 presumes an attendance determination step; the specification needs a concrete enabling disclosure of how attendance is established (venue check-in, transaction record per US12190310, group confirmation) or Claim 2 risks a §112 enablement/written-description rejection.
- **Backfill is only a request.** `flows.py` emits a `BackfillRequest`; the candidate-selection and re-invitation path should be described in the specification if Claim 1(e)(iii) is to carry weight.
- **Priority timing.** The closest live art is a 2018 filing and the closest agent-architecture art is 2026 non-patent literature that is still accumulating. Every additional month of published agent-safety work erodes the "minimal audited action surface" story specifically. If this is filed at all, the provisional should go in promptly and Claim 3 should be treated as a dependent-claim hedge, never as the basis for the filing.

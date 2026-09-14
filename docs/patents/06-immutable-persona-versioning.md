# 06 — Immutable Persona Versioning and Reproducible Decision Pinning

**Status:** prior-art research complete, patentability verdict below
**Repo artifacts:** `docs/technical-architecture.md` §1 rule 1 and §2; `regulars/core/persona.py`;
`regulars/persona/versioning.py`; `regulars/persona/legibility.py`; `regulars/db/schema.py`
(`persona_versions`, `PERSONA_IMMUTABLE_DDL`); `regulars/db/repo.py`
(`append_persona_version`, `write_audit`); `regulars/pipeline.py`
(`legibility_reply`, `apply_persona_correction`).

---

## Mechanism Summary

The system stores an inferred behavioral profile ("persona") for each user and never mutates it.
Four elements combine.

**(a) Database-enforced append-only profile history.** `persona_versions(id, user_id, version,
spec_json, source, created_at)` carries a unique key on `(user_id, version)` and a monotonic
per-user version counter computed inside the writing transaction, so two concurrent appends
cannot both win — the loser gets an `IntegrityError` rather than silently overwriting. On top of
the application guarantee (the repository layer deliberately exposes no update function), the
schema installs a `BEFORE UPDATE` trigger per dialect that aborts any `UPDATE` against the table:
`RAISE(ABORT, 'persona_versions is append-only')` on SQLite, a `plpgsql` `RAISE EXCEPTION` on
Postgres, both attached via a SQLAlchemy `after_create` event listener so they exist wherever the
schema is created.

**(b) Decision records pinned to the exact profile version.** `sims` carries
`persona_ver_a` and `persona_ver_b` as foreign keys into `persona_versions` — not user ids — so a
simulated transcript records the precise behavioral input it was generated from. `audit_log`
carries `persona_version_id` (FK) alongside `judge_version`, `decision_type`, `reason` and
`payload_json`, and is written at every consequential decision point in `regulars/pipeline.py`
(group assembly, removal, persona update). The intent is that any historical algorithmic decision
is re-derivable against the byte-exact input that produced it.

**(c) Structured human-readable specification, not an embedding, with per-field provenance and
confidence.** `PersonaSpec` is four typed sections (`conversational`, `content`, `social`,
`stated`) of short text and enumerated fields, plus a `Provenance` block holding
`sources: list[PersonaSource]` and `confidence_by_field: dict[str, float]` keyed by dotted field
path (`"social.energy"`). `PersonaSource` is one of `voice_interview`, `chat_behavior`,
`chatlog_import`, `user_correction`. Because the spec is text and field-addressable,
`regulars/persona/legibility.py` renders every populated field as a numbered plain-English
sentence — the "what do you know about me" feature — and maps a line number back to a field path.

**(d) Corrections as versioned, high-confidence, provenance-tagged writes that double as labels.**
`apply_correction` sets the field, appends `user_correction` to `provenance.sources`, and pins
that field's confidence to `1.0`; `apply_persona_correction` then calls `append_persona_version`,
so the user's rebuttal becomes version N+1 rather than an edit. `merge_update` folds observed chat
behavior into a base spec under asymmetric rules — list fields union and never shrink, scalars
change only when the observing pass explicitly set them, and user-stated `exclusions` can only
grow, so behavioral observation can never overrule something the user said about themselves.

**The product claim this supports:** a person removed from a group or matched into one can ask
why, and the system can reproduce the answer exactly — which persona version, which judge version,
which stated exclusions were in force at that moment.

> **Implementation gap relevant to drafting.** The `predictions` table currently stores
> `user_a`, `user_b`, `judge_version`, `y_hat`, `y_hat_calibrated`, `created_at` — it has **no**
> foreign key to `persona_versions` and no FK to `sims`. Pinning today runs through `sims` and
> `audit_log` only. Any claim reciting "the prediction record stores a reference to the persona
> version" is not literally practiced by that table as written. Either add
> `persona_ver_a`/`persona_ver_b` (or `sim_id`) to `predictions` before filing, or draft around
> "decision record" rather than "prediction record."

---

## Prior Art Found

### Tier 1 — directly anticipatory

**US 11,409,990 B1 — "Machine learning archive mechanism using immutable storage"** (Bottomline
Technologies, priority 1 Mar 2019). Claim 1 recites a server with a blockchain attached, an
application, an ML engine, and an ML model that "updates periodically," wherein the model "is
stored in the blockchain by the special purpose multi-core server when the machine learning model
is updated." Dependent disclosure covers write-once-read-many hardware and cloud immutable storage
as alternatives, preserves customer data alongside model versions, and — critically — stores
records in immutable storage that include "a link to the immutable storage record for the model,"
i.e. per-decision linkage from a result back to the exact model version that produced it. The
stated motivation is giving bank auditors the reason a loan or account decision was made. This is
element (a) + element (b) with the roles of model and input swapped, in the same
explainability-for-regulators framing.

**US 11,784,821 B2 — "Cognitive process data and decision tracking to facilitate artificial
intelligence operations compliance"** (IBM, priority 11 Mar 2020). Claim 1: obtain an AI decision
model; receive inputs each having "a data value, a data type, and a geographical source"; execute
the model on those inputs; **record, in a ledger, the result of the execution, the plurality of
inputs, and an identifier of the artificial intelligence decision model**; perform a compliance
check against compliance criteria; store a compliance verification indication in the ledger. The
specification states the ledger is a blockchain-backed distributed ledger. This is element (b)
essentially verbatim: a per-decision record binding result → exact inputs → exact model identifier,
in an immutable store, for compliance auditing.

**US 11,915,236 B2 / US 2022/0284422 / US 12,481,990 — "Immutable database for bitemporal
analysis"** (Capital One, priority 5 Mar 2021). A bitemporal ledger "configured as an event store
that only allows appending new data," with `created_at` and `valid_from` temporal metadata, that
"prevents data overwrites" — records cannot be modified or deleted retroactively. Corrections do
not modify existing entries; they create a separate alternate timeline. That last point is element
(d)'s "a correction is a new version, not an edit," claimed at the database level and not tied to
any particular domain.

### Tier 2 — the human-readable, user-correctable profile

**Judy Kay's scrutable user models / the Personis family** (University of Sydney, ~1995–2010;
*Personis: A Server for User Models*, AH 2002; *PersonisAD*, Pervasive 2007; Holden & Kay, *The
Scrutable User Model and Beyond*). This is the strongest non-patent art against element (c) and
much of (d). Personis holds the user model as **evidence** — each piece tagged with its source and
the process that produced it — rather than as an opaque vector, resolves evidence to a value at
query time via pluggable resolvers, lets the user inspect the details of what is held about them,
the processes used to gather it, and how it drove personalization, and lets the user **alter the
model at any point in the interaction**. Open-learner-model and scrutable-adaptation literature
generalizes this: learners inspect and edit the observations, inferences and assumptions the
system holds. Field-level source attribution + human inspection + user editing of an inferred
profile is roughly thirty years old and thoroughly published.

**US 9,251,468 B2 — "Inferring user profile attributes from social information"** (Facebook).
Inferred attributes carry per-attribute confidence scores, and conflicts between sources are
resolved by attaching confidence to the source — GPS location weighted above self-reported
location. That is per-field confidence keyed to provenance.

**US 2015/0347917 A1 — "Inferred identity"** (LinkedIn). Predicts a missing profile attribute and
generates a confidence score for that specific member/attribute pair.

**US 11,799,974 — "User profile aggregation and inference generation."** Accuracy confidence
levels are set by whether the user has **confirmed or validated** the attribute, and the UI
presents attributes for the user to confirm or update. That is element (d)'s "a user correction is
the highest-confidence source," minus the versioning.

### Tier 3 — the architectural pattern itself

Event sourcing and CQRS: every state change captured as an immutable event appended to an
append-only store which is the authoritative source of truth, with current state derived by replay
and a complete audit trail as a free consequence. Documented as a named pattern by Fowler (2005),
in Microsoft's Azure Architecture Center, and claimed in passing in patents such as US 10,761,861
("event stream modification") and US 11,943,344 ("Mutation processing for events," which recites
append-only write databases and replay-derived state).

Temporal and bitemporal databases: Snodgrass, *Developing Time-Oriented Database Applications in
SQL* (1999); SQL:2011 system-versioned and application-time period tables; IBM's US 8,219,522 /
US 8,713,073 ("Management of temporal data by means of a canonical schema"). Datomic (2012)
ships immutable, monotonically-timestamped facts with as-of query as its core product thesis.

Point-in-time-correct feature retrieval in ML feature stores (Feast, Tecton, Databricks Feature
Store, SageMaker Feature Store) is exactly the "reproduce the decision against the inputs as they
stood at decision time" guarantee, sold commercially since ~2019 and described as the central
correctness problem of the category.

MLOps lineage: MLflow, SageMaker model registry, and the surrounding lineage literature
(US 10,621,033 / US 11,237,900 on dataflow lineage) hold that every model version is immutable
once registered, that overwriting artifacts in place destroys the audit trail, and that
column-level lineage tied to model metadata is what lets a team reproduce an inference from the
exact data version that fed it.

DB-level rejection of `UPDATE` via a `BEFORE UPDATE` trigger is textbook DBA practice for
append-only audit tables and predates all of the above; `RAISE(ABORT, …)` in SQLite triggers and
`RAISE EXCEPTION` in `plpgsql` are documented vendor features used for precisely this.

### Tier 4 — the domain

Matchmaking patents (US 6,272,467; US 7,203,674; US 2002/0040310; US 11,513,666) describe profile
databases with one record per user and weighted compatibility scoring. Notably, **none of the
matchmaking art surveyed stores a version-pinned snapshot of the profile used for a given
recommendation, and none exposes an audit trail for why a match was made.** The domain gap is
real; the mechanisms that fill it are not novel.

### Legal context (not prior art, but load-bearing)

GDPR Arts. 13–15 and 22(3) give a data subject the right to meaningful information about the logic
of a solely automated decision, to human intervention, and to contest it — the compliance driver
this mechanism serves. Scholarship disputes whether a hard "right to explanation" exists
(Wachter/Mittelstadt/Floridi vs. Selbst/Powles), but the record-keeping obligation is not disputed,
and the EU AI Act's logging duties push the same way. This is a **regulatory motivation**, which
under §101 analysis is a business/legal need, not a technical problem — a point that cuts against
eligibility, not for it.

---

## Novelty Analysis

Element by element, against the art above.

| Element | Novel? | Killed by |
|---|---|---|
| (a) Append-only versioned profile rows | **No** | Event sourcing; SQL:2011 system-versioned tables; Datomic; Capital One US 11,915,236 |
| (a′) Enforcement specifically by a DB `BEFORE UPDATE` trigger rather than app logic or a ledger | **Marginal** | Standard DBA practice; likely obvious over US 11,915,236 ("event store that only allows appending"). Arguably the one element *not* explicitly recited in the closest patents, which all reach for blockchain/WORM instead |
| (b) Decision record FK-pinned to the exact input version | **No** | IBM US 11,784,821 claim 1 (records inputs + model identifier per execution in a ledger); Bottomline US 11,409,990 ("link to the immutable storage record for the model"); feature-store point-in-time correctness |
| (c) Structured human-readable text spec instead of an embedding | **No** | Personis / scrutable user models; every pre-2013 attribute-value profile system. "Not an embedding" is a reversion to older art, not an advance over it |
| (c′) Per-field confidence + per-field source provenance | **No** | US 9,251,468 (confidence by source); US 2015/0347917; Personis evidence model |
| (d) User correction as highest-confidence, provenance-tagged input | **No** | US 11,799,974 (confirmed-by-user raises accuracy confidence); Personis user editing; open learner models |
| (d′) Correction creates a new version rather than an edit | **No** | Capital One US 11,915,236 — corrections create alternate timelines, never modify entries |
| (e) Asymmetric merge: user-stated exclusions monotonically grow and behavioral inference can never override them | **Plausibly novel as claimed** | No art found reciting a merge rule where the provenance class of a field determines whether a later inference may overwrite it. Weakest link: it is a conflict-resolution policy, and confidence-weighted source precedence (US 9,251,468) is the obvious generalization |
| (f) The specific combination applied to behavioral matching, where the reproducible artifact is a *behavioral spec* rather than a model, and the contestable decision is group membership | **Novel in combination only** | Combination novelty over four well-known elements in a new field is the textbook §103 "obvious to try / known technique applied to a known system" posture |

The honest read: **every individual element is anticipated, most of them by more than one
reference, and three of the four references teach the elements in the same explainability/audit
motivation.** The only things not found in the art are (e), the provenance-class-gated merge rule,
and (f), the field of use.

An examiner combining US 11,784,821 (per-decision ledger record binding result to exact inputs and
model id) with US 11,915,236 (append-only bitemporal store where corrections spawn versions rather
than edits) and US 9,251,468 / US 11,799,974 (per-field confidence keyed to source, user
confirmation as the strongest source) reaches essentially the whole mechanism under §103 with
three references and a routine motivation to combine — all four are in the "audit inferences about
people" space.

---

## Patentability Verdict

**Do not file this as a standalone patent.** Of the ten mechanisms, this is the weakest candidate
and it is not close. It fails on §102/§103 before §101 is even reached, and it would fail §101 too.

### §103 (the dispositive problem)

Three references — Bottomline, IBM, Capital One — each reach two of the four elements, all filed
2019–2021, all assigned to well-resourced financial/enterprise filers, all motivated by "auditors
and regulators need to know why the machine decided." The combination is not merely available to a
hypothetical examiner; it is the obvious one. The field of use (social matching rather than credit)
does not rescue it: swapping the domain of a known auditing technique is the paradigm §103
rejection, and *KSR*'s "known technique applied to a known device ready for improvement" framing
fits exactly.

### §101 / *Alice* (the secondary problem)

Step one asks what the claim is *directed to*. Stripped to its essence, this claim is: keep a
dated history of what you believed about a person, write down which belief you used when you acted,
and be able to show it later. That is **record-keeping for accountability** — a practice humans
performed with paper ledgers, medical charts and credit files for centuries, and it is squarely the
"collecting information, analyzing it, and displaying the results" abstraction that sank *Electric
Power Group v. Alstom* (Fed. Cir. 2016). Doing it in a relational database with foreign keys is
generic computer implementation of the kind *Alice* holds insufficient.

The strongest available counterargument runs through *Enfish v. Microsoft* (Fed. Cir. 2016) and
*Koninklijke KPN v. Gemalto* (Fed. Cir. 2019): a claim directed to a **specific improvement in the
way a computer stores and retrieves data**, described as such in the specification, can survive
step one. *Enfish*'s self-referential table was eligible because the data structure itself was the
advance. To use that path here the claim must be about the *storage architecture* — the trigger,
the monotonic counter, the concurrency behavior of the unique key, the FK topology — as a technical
improvement over profile stores that overwrite.

That argument is weak here for three reasons. First, the spec does not frame the mechanism as a
database improvement; `docs/technical-architecture.md` §1 rule 1 frames it as a **business
requirement** ("your entire moat is the pairing of prediction to outcome"), and *KPN* turns on the
improvement being "described as such in the specification." Second, the trigger and the unique key
are off-the-shelf DBMS features used exactly as documented — there is no new data structure, no
change to how the database stores or retrieves anything, and therefore nothing like *Enfish*'s
altered table format. Third, the stated benefit is legal and evidentiary (a person can contest a
decision), not technical (nothing runs faster, smaller, or more reliably as a machine).

Under the 2019 PTO Guidance a claim might scrape through prosecution by reciting the trigger and
the concurrency-safe append as a "practical application," and an examiner in art unit 2160 might
allow something narrow. But allowance is not survival: a claim that narrow is trivially designed
around (see below), and in litigation a *Berkheimer*-flavored fight over whether a `BEFORE UPDATE`
trigger is "well-understood, routine, conventional" is one that loses, because the answer is
plainly yes.

### Recommendation

Protect this as **compliance architecture and trade secret**, not patent:

1. **Trade secret / know-how.** The asymmetric merge rules in `merge_update` — union-never-shrink
   on lists, `model_fields_set` gating on scalars, `stated.exclusions` monotonic, observation never
   overriding a user statement — are the genuinely non-obvious part and are invisible from outside
   the product. A patent would publish them for a claim that would not hold. Keep them secret.
2. **Defensive publication** of the append-only-plus-FK-pinning architecture, if anything, to
   foreclose a competitor patenting it against you. Cheap, fast, and it costs nothing you could
   have enforced.
3. **Trust and compliance positioning.** "We can tell you exactly why, and prove it" is a market
   and regulatory asset independent of IP. It also materially de-risks the removal/exclusion
   feature, which is where legal exposure actually lives.
4. **Fold the reproducibility guarantee into stronger filings** as a dependent claim. Where a
   sibling mechanism in this series has a genuinely novel independent claim — the simulation
   harness, the feature extraction, the group-assembly optimizer — "wherein the persona version
   used to generate the prediction is stored as a foreign key in the decision record and the
   decision is re-derivable therefrom" is a fine dependent claim. It adds infringement surface at
   near-zero marginal cost and never has to stand on its own.

**If filing anyway** (e.g. for investor optics), file the narrowest possible claim — element (e),
the provenance-class-gated merge, recited as the point of novelty — and expect a §101 rejection
plus at least one §103 rejection over US 11,784,821 in combination with US 9,251,468.

---

## Draft Claims

Presented weakest-independent-first, with the honest kill shot on each. None of these is
recommended for standalone filing; claims 4 and 5 are the only ones worth drafting, and only as
dependents inside another application.

### Claim 1 — the naive framing (do not file)

> A computer-implemented method comprising: storing, in a table of a relational database, a
> plurality of behavioral profile records for a user, each record comprising a monotonically
> increasing version number, a timestamped creation time, and a structured specification; installing
> on the table a database trigger that aborts any UPDATE statement directed to the table; responsive
> to receiving new information about the user, inserting a new record having a version number one
> greater than the maximum existing version number for that user rather than modifying any existing
> record; storing, in a decision record, a foreign key reference to the profile record active at the
> time the decision was computed; and, responsive to a request to explain the decision, retrieving
> the referenced profile record and re-deriving the decision therefrom.

**Kill shot.** Anticipated in substance by US 11,784,821 (record result + inputs + model identifier
in a ledger, then answer compliance questions from it) combined with US 11,915,236 (append-only
event store that refuses overwrites). The trigger is the only element not expressly recited in
either, and it is the conventional implementation of what both describe. §101 step one: directed to
recording and reporting information. **Dead on arrival.**

### Claim 2 — narrowed to the field-level provenance spec

> The method of claim 1, wherein the structured specification comprises a plurality of
> human-readable natural-language fields addressable by field path, and a provenance structure
> comprising (i) a set of source identifiers each drawn from an enumerated set comprising at least a
> spoken-interview source, an observed-conversation source, an imported-conversation-log source, and
> a user-correction source, and (ii) a mapping from field path to a numeric confidence value;
> wherein the specification is stored as text and is supplied as context to a language model without
> intermediate embedding.

**Kill shot.** Personis and the scrutable-user-model literature (1995–2010) teach evidence-tagged,
source-attributed, user-inspectable profile fields. US 9,251,468 teaches per-field confidence keyed
to source reliability. "Text rather than embedding" is a return to the older art. The only arguably
fresh limitation — "supplied as context to a language model without intermediate embedding" — is a
statement about downstream use that is both obvious post-2023 and easy to design around.

### Claim 3 — the transparency loop

> The method of claim 2, further comprising: rendering each populated field of the specification as
> a numbered natural-language sentence and transmitting the numbered rendering to the user;
> receiving from the user a reply identifying one of the numbers and asserting a corrected value;
> mapping the identified number to the corresponding field path; and inserting a new profile record
> in which the identified field holds the corrected value, the provenance source set includes the
> user-correction source, and the confidence value for that field path is set to the maximum of the
> confidence range.

**Kill shot.** US 11,799,974 presents inferred attributes for user confirmation and raises
confidence on confirmation. Open learner models let learners edit the system's beliefs about them.
The numbered-line ↔ field-path mapping is a UI affordance, not a technical contribution. §101: this
is a conversation, mechanized.

### Claim 4 — provenance-class-gated merge (the only genuinely fresh limitation)

> A computer-implemented method of updating a behavioral profile without loss of user-asserted
> constraints, comprising: maintaining an append-only store of versioned profile records, each
> record comprising a structured specification whose fields are partitioned into at least an
> inferred class and a user-asserted class, and a provenance structure mapping each field to a
> source identifier and a confidence value; receiving an observed specification derived by a machine
> learning process from the user's conversational behavior; and computing a merged specification by
> applying a merge in which (i) each list-valued field of the merged specification is the order-
> preserving union of the base and observed values such that no element present in the base is
> removed, (ii) each scalar-valued inferred field takes the observed value only when the observed
> specification explicitly set that field to a non-empty value, and (iii) each field of the
> user-asserted class is immune to the observed specification, such that a value asserted by the
> user is never removed or overridden by a value derived from observed behavior; and appending the
> merged specification as a new versioned record whose provenance source set is the union of the
> base source set and the source of the observed specification.

**Assessment.** The strongest claim of the five, and the only one where a search turned up no
direct teaching. The novelty is the *ratchet*: a monotonic, one-directional constraint set that
inference may extend but never contract, tied to the provenance class of the field rather than to a
numeric confidence comparison. Risk: an examiner will read this as confidence-weighted source
precedence (US 9,251,468, where GPS beats self-report) with the weights pinned to 1.0 and 0, and
call it obvious. Counterargument to preserve: confidence weighting is a *comparison* that a
sufficiently confident inference always wins; a provenance-class gate is a *prohibition* that no
amount of inferred confidence can defeat — which is what makes the safety property (a stated
exclusion cannot be inferred away) hold. Recite the safety property in the spec, since *KPN*
rewards specifications that describe the improvement as an improvement.

### Claim 5 — the dependent worth keeping

> The method of [a parent claim directed to generating a compatibility prediction from simulated
> dyadic conversation], wherein generating the prediction further comprises storing, in a prediction
> record, a foreign key reference to the specific versioned profile record of each participant that
> was supplied to the simulation, and a version identifier of the scoring model; and further
> comprising, responsive to a request received after the decision, retrieving the referenced
> versioned profile records and the identified scoring model and recomputing the prediction to
> produce a value identical to the stored prediction.

**Assessment.** As a standalone claim this is US 11,784,821. As a dependent hanging off a novel
simulation or assembly claim it costs nothing, adds real infringement surface (a competitor
offering an explain-my-match feature over a versioned store reads on it), and inherits its parent's
§101 posture instead of fighting its own. **This is the claim to actually write.**

> **Drafting prerequisite:** claim 5 requires `predictions` to carry the FK. It does not today.
> Add `persona_ver_a`/`persona_ver_b` (or a `sim_id` FK) to the `predictions` table before any
> filing, or the claim lacks written-description support in the implementation.

---

## Design-Around Notes

### Prior art to cite in an IDS (if any application in this family goes forward)

Failure to cite these, given they surfaced in a first-pass search, is an inequitable-conduct
exposure that costs more than the claims are worth.

- US 11,409,990 B1 — Bottomline Technologies — *Machine learning archive mechanism using immutable
  storage* (prio. 2019-03-01)
- US 11,784,821 B2 — IBM — *Cognitive process data and decision tracking to facilitate artificial
  intelligence operations compliance* (prio. 2020-03-11)
- US 11,915,236 B2, US 2022/0284422 A1, US 12,481,990 — Capital One — *Immutable database for
  bitemporal analysis* (prio. 2021-03-05)
- US 9,251,468 B2 — Facebook — *Inferring user profile attributes from social information*
- US 2015/0347917 A1 — LinkedIn — *Inferred identity*
- US 11,799,974 — *User profile aggregation and inference generation*
- US 12,217,011 — *Generating adaptive textual explanations of output predicted by trained
  artificial-intelligence processes*
- US 8,219,522 B2 / US 8,713,073 B2 — IBM — *Management of temporal data by means of a canonical
  schema*
- US 10,761,861 — *Event stream modification*; US 11,943,344 — *Mutation processing for events*
- US 6,272,467 B1; US 7,203,674 B2; US 2002/0040310 A1; US 11,513,666 B2 — matchmaking/profile
  scoring
- Non-patent: Kay & Kummerfeld, *Personis: A Server for User Models* (AH 2002); Assad et al.,
  *PersonisAD* (Pervasive 2007); Holden & Kay, *The Scrutable User Model and Beyond*; Snodgrass,
  *Developing Time-Oriented Database Applications in SQL* (Morgan Kaufmann, 1999); SQL:2011
  system-versioned tables; Fowler, *Event Sourcing* (2005); Microsoft Azure Architecture Center,
  *Event Sourcing pattern*; Datomic architecture documentation; Feast/Tecton point-in-time-correct
  feature retrieval documentation.

### How a competitor designs around any claim here (i.e. why enforcement fails)

- **Drop the trigger, keep the behavior.** Enforce append-only with `REVOKE UPDATE` on the role, or
  a WORM/object-lock bucket, or an event-sourced log with a materialized read model, or just an ORM
  with no update path. Every one of these avoids a claim reciting a `BEFORE UPDATE` trigger while
  delivering the identical guarantee.
- **Pin by content hash instead of foreign key.** Store `sha256(spec_json)` in the decision record
  and keep specs in content-addressed blob storage. No FK, no version column, same reproducibility.
  This defeats claim 1 and claim 5 as literally drafted, which is an argument for drafting
  "reference that uniquely identifies the profile state" rather than "foreign key."
- **Pin by timestamp against a bitemporal table.** Store `decided_at` and read the profile `AS OF`
  that instant using SQL:2011 system-versioned tables. No explicit version number, no explicit
  reference stored at all — and the art here (Snodgrass, IBM US 8,219,522) predates everything.
- **Use an embedding plus a rendered text view.** Claim 2's "without intermediate embedding"
  limitation is avoided by storing both and rendering text on demand.
- **Invert the merge.** Claim 4's ratchet is avoided by letting inference propose and requiring
  user confirmation, or by a numeric confidence floor on user-asserted fields set to 1.0 — a
  competitor implementing the identical safety property through confidence weighting rather than
  class gating does not read on limitation (iii).

### Defensive posture to take instead

1. **Publish the architecture, keep the policy.** A short defensive publication of "append-only
   behavioral profile with decision-time version pinning" blocks a competitor from patenting it
   against you. Say nothing about `merge_update`'s asymmetry rules, which stay secret.
2. **Do not disclose the merge semantics** in any public repo, blog post, conference talk, or
   investor deck. They are in `regulars/persona/versioning.py`; if this repository ever goes public
   or partially open-source, that module is the one to hold back. Trade secret protection requires
   demonstrable secrecy measures.
3. **Make reproducibility contractually and operationally real** — the FK gap in `predictions`
   noted above means the guarantee is currently narrower than the architecture doc claims. Close it.
   The trust claim is only as good as the weakest un-pinned decision path, and that is a product
   liability question independent of IP.

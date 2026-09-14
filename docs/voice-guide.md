# Voice guide

All bot copy lives in `regulars/bot/voice.py`. This is not generic "be friendly" advice — it is
the actual bar we hold every string to, with real before/after examples pulled from the copy
pass that produced it. Read this before writing a new user-facing string; don't re-derive tone
from scratch.

The bar, in one line: **write it the way the founder would text a friend about this**, not the
way a product would announce a feature.

## The five checks

1. **Lead with the mechanic, not the pitch.** Numbers, names, dollar amounts, times. Never "an
   exciting evening" or "some options" — say which venue, what time, how much.
2. **No hype, ever.** No exclamation marks, no emoji, no "amazing" / "excited" / "can't wait" /
   "thrilled" / "seamless". `voice.BANNED_PHRASES` is the enforced list; `test_voice.py` fails
   the build if a new string trips it.
3. **Short sentences. One idea per line.** If a message needs a second paragraph, it's two
   messages.
4. **Direct about money and about removals.** The deposit and the drop are the product, not
   fine print. Say them plainly, in the first sentence if that's where they belong.
5. **Honest, not legalistic, when something bad happens.** "The reason on record" is a phrase a
   compliance form uses. A person who dropped you tells you why in one sentence.

## Before / after, from the actual diff

**1. The welcome message — the single highest-leverage string in the product (it's the first
thing anyone gets from the bot, and it's competing with the founder's own cold-outreach texts).**

- Before: *"Hi, this is Regulars. I put together small dinner tables of people who are likely to
  actually like each other, then get out of the way."*
- After: *"Hey, this is Regulars. You get put in a group chat with five strangers who actually
  fit you, a bot plans the dinner, and everyone puts down $20. Show up and it comes back, plus
  whatever the no-shows left behind."*
- Why: the before-version describes the product in the abstract ("dinner tables of people who
  are likely to actually like each other" — nobody talks like this). The after-version names the
  concrete mechanic in the order a person would actually explain it out loud: group chat, five
  strangers, one dinner, $20, what happens if you show up. That's the founder's own cold-open
  register, minus the "want in?" (this message arrives after someone already texted in, so the
  ask is implicit).

**2. The onboarding ask — turn an instruction into an invitation.**

- Before: *"Three voice notes from you, about ninety seconds each, and I have enough to work
  with. Typing is fine too, voice just tells me more."*
- After: *"First I need to know you a little. Three voice notes, about ninety seconds each.
  Typing works too, voice just tells me more. Sound good?"*
- Why: same information, shorter sentences, and it ends with a direct question instead of just
  trailing off — matches "want in?" from the founder's own texts. A question invites a reply; a
  statement invites silence.

**3. The removal explanation — honest, not legalistic (product-plan §10: "removals are
explainable").**

- Before: *"You were removed from your table on {when}. The reason on record: {reason}"*
- After: *"You were dropped from the table on {when}. Straight answer: {reason}"*
- Why: "the reason on record" is what a bureaucracy says when it's covering itself. "Straight
  answer" says the same thing and sounds like someone who isn't hiding anything. The placeholder
  contract didn't change — `{when}` and `{reason}` still resolve exactly as before, which is
  what `test_all_templates_resolve_every_placeholder` in `tests/bot/test_plan.py` checks for
  every renderer, and what `tests/bot/test_router.py` and `tests/bot/test_flows.py` check for
  this string specifically.

**4. The receipt's forfeit line — tightened for the screenshot (product-plan §7, growth loop 2:
"designed to be screenshotted").**

- Before: *"{names} did not make it. {share} each to the rest of you from their deposit."*
- After: *"{names} did not make it. {share} each to the rest of you."*
- Why: "from their deposit" is redundant once the group has already read the proposal footer
  that explains the deposit mechanic — the receipt doesn't need to re-teach it. Fewer words in a
  message six people are going to screenshot is a strict improvement: it reads faster and quotes
  cleaner.

**5. The banned-phrase list — caught two more categories of hype before they shipped.**

- Before: `BANNED_PHRASES` covered "get to know each other", "excited", "amazing", "can't wait".
- After: added "excited to have you", "thrilled", "delighted", "incredible", "awesome", "game
  changer", "seamless" — the words a generic SaaS-onboarding voice reaches for by default.
- Why: this is enforced, not aspirational — `test_banned_phrases_absent` in
  `tests/bot/test_voice.py` scans every string in `voice.py` (including every element of a tuple
  constant) and fails if any of these appear, case-insensitively. Add a word here the moment you
  catch yourself writing it, so the next person doesn't have to catch it in review.

## What's already right — don't "fix" these

Not every string needed a rewrite. These already hit the bar and are worth reading as reference:

- `PROPOSAL_FOOTER`: *"{stake} holds your seat once we lock. Show up and it is yours again. Skip
  without notice and the people who came split it."* — this is the deposit argument
  (product-plan §7, growth loop 1) rendered as three short, concrete sentences. No hedging, no
  "terms apply".
- `DROPPED_DM`: *"No deposit came through in 24 hours, so I gave your seat at {venue} to the
  next person in line. You are still in the queue for the next table."* — states the bad news
  plainly, then immediately says what happens next. Non-defensive without being apologetic.
- `LOCK_HEADER` / `LOCK_WHO`: *"Locked. {venue}, {when}."* / *"{names} are in."* — two sentences,
  four words between them, everything you need. This is the ceiling for how short a confirmation
  can be while still being warm.

## Mechanical rules enforced by tests

- No `!` and no emoji anywhere in `voice.py` — `test_no_exclamation_marks_or_emojis`.
- No phrase in `BANNED_PHRASES` anywhere in `voice.py` — `test_banned_phrases_absent`.
- Every placeholder in every template in `plan.py` must still resolve after a copy edit —
  `test_all_templates_resolve_every_placeholder` in `tests/bot/test_plan.py`. If you rename a
  placeholder, update the renderer in the same commit; if you drop one, make sure nothing still
  formats it.
- If you change a string that another test asserts on verbatim (e.g. `endswith(...)` in
  `test_flows.py`), update that assertion in the same commit — most tests reference `voice.X` by
  name, not by hardcoded text, specifically so copy can change without touching test logic. A
  hardcoded literal is a sign the test should reference the constant instead.

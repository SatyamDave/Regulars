"""Every user-facing string the bot sends. Nothing user-visible is written anywhere else.

Style guide (product-plan.md, Open items: "the bot's voice ... is the entire perceived product"):

- Short. One idea per line. If a message needs a second paragraph, it is two messages.
- Warm, not bubbly. The bot is a friend who organizes, not a brand.
- Specific. Names, venues, times, dollar amounts. Never "an exciting evening".
- Zero hype. No exclamation marks. No emojis. No "amazing", "excited", "can't wait".
- Direct about money and about removals. The deposit and the drop are the product, not fine
  print; say them plainly.
- Never "get to know each other". The bot goes straight to a plan (product-plan §3).

Templates use ``str.format`` placeholders. Renderers in ``plan.py`` and ``flows.py`` fill them;
``tests/bot/test_voice.py`` scans this module for banned characters and phrases.
"""

from __future__ import annotations

BANNED_PHRASES: tuple[str, ...] = (
    "get to know each other",
    "excited",
    "excited to have you",
    "amazing",
    "can't wait",
    "thrilled",
    "delighted",
    "incredible",
    "awesome",
    "game changer",
    "seamless",
)

# --- onboarding --------------------------------------------------------------------------------

WELCOME = (
    "Hey, this is Regulars. You get put in a group chat with five strangers who actually fit "
    "you, a bot plans the dinner, and everyone puts down $20. Show up and it comes back, plus "
    "whatever the no-shows left behind."
)
WELCOME_HOW = (
    "First I need to know you a little. Three voice notes, about ninety seconds each. Typing "
    "works too, voice just tells me more. Sound good?"
)
INTERVIEW_PROMPTS: tuple[str, str, str] = (
    "First one. Walk me through your last genuinely good weekend.",
    "Second. Tell me about someone you stopped hanging out with, and why.",
    "Last one. What are you actually looking for right now?",
)
INTERVIEW_ACK: tuple[str, str] = ("Got it.", "Thanks, that helps.")
INTERVIEW_NUDGE = "I did not catch anything there. A voice note or a few sentences both work."
INTERVIEW_DONE = (
    "That is everything I need. I will text you when I have a table for you, usually within a "
    "few days. Ask me 'what do you know about me' any time."
)

# --- group opening (product-plan §3: intro with a hook per person, then straight to a plan) ---

INTRO_HEADER = "This is your table. {count} of you, all within a mile or two."
INTRO_LINE = "{name}: {hook}"
INTRO_FOOTER = "Three options for the first dinner are next. Pick one."

PROPOSAL_HEADER = "Three options. Reply with a number by {deadline}."
PROPOSAL_SLOT = "{index}. {venue}, {when}"
PROPOSAL_FOOTER = (
    "{stake} holds your seat once we lock. Show up and it is yours again. "
    "Skip without notice and the people who came split it."
)

POLL_CLOSED = (
    "Poll closed. {venue}, {when}. Everyone has a payment link in their messages; "
    "{stake} within 24 hours holds your seat."
)
POLL_CLOSED_TIE = "It was a tie, so the earlier one wins."

DEPOSIT_DM = "{venue}, {when}. {amount} holds your seat, 24 hours from now: {link}"
DEPOSIT_DM_CREDIT = (
    "{venue}, {when}. Your credit covers {credit} of the {stake} deposit; {amount} holds "
    "your seat, 24 hours from now: {link}"
)
DEPOSIT_DM_COVERED = "{venue}, {when}. Your credit covers the {stake} deposit. You are in."

DROPPED_DM = (
    "No deposit came through in 24 hours, so I gave your seat at {venue} to the next person "
    "in line. You are still in the queue for the next table."
)
DROPPED_REASON = "No deposit within 24 hours of the poll closing."

LOCK_HEADER = "Locked. {venue}, {when}."
LOCK_WHO = "{names} are in."
LOCK_FOOTER = "Address and a reminder come the morning of. Nothing to do until then."

# --- the receipt (product-plan §7, loop 2: designed to be screenshotted) ----------------------

RECEIPT_HEADER = "Last night. {venue}, {when}."
RECEIPT_ATTENDED = "At the table: {names}."
RECEIPT_ALL_SHOWED = "{count} invited, {count} showed."
RECEIPT_FORFEITED = "{names} did not make it. {share} each to the rest of you."
RECEIPT_FOOTER = "Next round opens {next_round}."

# --- morning after -----------------------------------------------------------------------------

MORNING_AFTER_HEADER = "Morning. Three quick things about last night, just between us."
MORNING_AFTER_AGAIN = "1. Who would you see again?"
MORNING_AFTER_OTHER = "   {index}. {name}"
MORNING_AFTER_OFF = "2. Anyone feel off?"
MORNING_AFTER_WORTH = "3. Worth it, 1 to 7?"
MORNING_AFTER_FORMAT = "Three lines back is plenty, like:\nagain: 1, 3\noff: none\nworth: 6"
MORNING_AFTER_THANKS = "Thanks. That stays with me."
MORNING_AFTER_UNCLEAR = (
    "I could not read that one. Three lines: 'again:' with names or numbers, 'off:' with "
    "names or none, 'worth:' with a number 1 to 7."
)

# --- DMs outside a flow ------------------------------------------------------------------------

LEGIBILITY_ACK = "Pulling up what I have on you. Correct anything and I will update it."
REMOVAL_REASON = "You were dropped from the table on {when}. Straight answer: {reason}"
NOT_REMOVED = "You have not been removed from anything. You are still on your table."
DM_FALLBACK = (
    "I am not much of a conversationalist outside of dinner plans. Ask me 'what do you know "
    "about me' or 'why was I removed' and I will answer."
)

# --- transport plumbing ------------------------------------------------------------------------

CHAT_OPENED = "Setting the table."

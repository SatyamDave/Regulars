"""Three hand-written 20-turn dyadic transcripts with known conversational shapes.

flow       follow-ups both ways, converging style, long topics, mutual disclosure, flats repaired
interview  one person asks everything (new topics), the other gives short answers
dead       one-word turns and deflections from both sides, nothing gets repaired
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from regulars.core.transcript import Transcript, Turn

A = "ana"
B = "ben"


def dyad(lines: Sequence[tuple[str, str]], source: str = "sim") -> Transcript:
    return Transcript(
        participants=[A, B],
        turns=[Turn(speaker_id=s, text=t) for s, t in lines],
        source="sim" if source == "sim" else "real",
    )


FLOW: list[tuple[str, str]] = [
    (A, "I finally started climbing at the bouldering gym near my place last month."),
    (B, "Oh nice, which gym? I've been thinking about climbing for ages."),
    (
        A,
        "The warehouse gym on Fifth Street. Honestly I was so nervous the first week, "
        "I felt like all the regulars were watching me fall off the climbing wall.",
    ),
    (
        B,
        "I get that, I felt exactly the same when I started running with a club. "
        "Did that fade once you knew a few of the regulars?",
    ),
    (
        A,
        "Yeah, mostly. One of the regulars taught me to read the routes and now I really "
        "love the puzzle of it. Do you still run with that club?",
    ),
    (
        B,
        "So the routes are like puzzles? I love a good puzzle. What grade are you climbing now?",
    ),
    (
        A,
        "Around V3, which is nothing, but I'm proud of it. What about you, did you ever "
        "actually go climbing?",
    ),
    (
        B,
        "Not yet, and I'm a little embarrassed about that. I keep telling myself I'll go "
        "climbing once work calms down.",
    ),
    (A, "Work has been rough for you lately? I've been a bit worried about you."),
    (
        B,
        "Rough is generous. I've been so stressed at work that I stopped sleeping properly, "
        "which I don't usually tell people.",
    ),
    (A, "oof."),
    (B, "anyway."),
    (
        A,
        "No wait, I'm glad you told me. My sister went through a stressed stretch like that "
        "last year and talking about it helped her a lot.",
    ),
    (
        B,
        "Did talking to your sister about it change anything for you too? I've never had "
        "that with mine.",
    ),
    (
        A,
        "It did, honestly. Talking with my sister made me realize I'd been hiding how lonely "
        "the move here made me feel.",
    ),
    (
        B,
        "Wait, did you move here for the job or for someone? I moved two years ago and I "
        "still feel that lonely some weeks.",
    ),
    (
        A,
        "For the job, which I love, but I didn't know a single person here for months. "
        "Was it the same for you when you moved?",
    ),
    (
        B,
        "Pretty much. I'm honestly still a little nervous about meeting new people, but I'm "
        "in if you're going to the gym this weekend.",
    ),
    (A, "yes!!"),
    (
        B,
        "Saturday morning then. Show me the routes and I'll bring the coffee, I promise not "
        "to embarrass you too much.",
    ),
]

INTERVIEW: list[tuple[str, str]] = [
    (A, "Where did you grow up?"),
    (B, "Ohio."),
    (A, "What do you do for work?"),
    (B, "Accounting."),
    (A, "Do you enjoy the accounting side of it?"),
    (B, "It pays."),
    (A, "Any hobbies?"),
    (B, "Not many."),
    (A, "Have you been to the new ramen place downtown?"),
    (B, "No."),
    (A, "Do you travel much?"),
    (B, "Sometimes."),
    (A, "Where was your last trip?"),
    (B, "Denver, for a conference."),
    (A, "Do you have siblings?"),
    (B, "One brother."),
    (A, "Are you a morning person?"),
    (B, "Not at all."),
    (A, "What are you reading right now?"),
    (B, "Nothing at the moment."),
]

DEAD: list[tuple[str, str]] = [
    (A, "hey"),
    (B, "hey"),
    (A, "did you see the game last night?"),
    (B, "no"),
    (A, "ok"),
    (B, "yeah"),
    (A, "meh"),
    (B, "same"),
    (A, "anyway"),
    (B, "sure"),
    (A, "work tomorrow"),
    (B, "yep"),
    (A, "cool"),
    (B, "k"),
    (A, "you around this weekend"),
    (B, "maybe"),
    (A, "ok"),
    (B, "cool"),
    (A, "night"),
    (B, "night"),
]


@pytest.fixture
def flow() -> Transcript:
    return dyad(FLOW)


@pytest.fixture
def interview() -> Transcript:
    return dyad(INTERVIEW)


@pytest.fixture
def dead() -> Transcript:
    return dyad(DEAD)

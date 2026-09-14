from __future__ import annotations

from regulars.core.transcript import Transcript, Turn
from regulars.features.repair import (
    FlatMoment,
    analyze_repairs,
    detect_flat_moments,
    find_repairs,
    flat_kind,
    is_deflection,
    repair_by_speaker,
    repair_latency_from,
    repair_latency_turns,
    repair_rate,
    repair_rate_from,
)
from tests.features.conftest import A, B, dyad


def test_deflection_lexicon() -> None:
    assert is_deflection("anyway")
    assert is_deflection("ok cool")
    assert is_deflection("Anyway, moving on.")
    assert is_deflection("i guess")
    assert not is_deflection("Anyway, my sister is visiting from Denver next week.")
    assert not is_deflection("I think that is fine.")


def test_flat_kind() -> None:
    assert flat_kind(Turn(speaker_id=A, text="oof.")) == "one_word"
    assert flat_kind(Turn(speaker_id=A, text="ok cool")) == "deflection"
    assert flat_kind(Turn(speaker_id=A, text="why?")) is None  # questions are never flat
    assert flat_kind(Turn(speaker_id=A, text="I started climbing last month.")) is None


def test_topic_death_flat_uses_segments() -> None:
    turns = [
        Turn(speaker_id=A, text="I started climbing at the gym last month."),
        Turn(speaker_id=B, text="My landlord raised the rent again."),
        Turn(speaker_id=A, text="The heating in my place is broken as well."),
        Turn(speaker_id=B, text="I baked bread on the weekend and it went well."),
    ]
    segments = [0, 1, 1, 2]
    flats = detect_flat_moments(turns, segments)
    assert [(f.index, f.kind) for f in flats] == [(0, "topic_death"), (2, "topic_death")]
    # the final topic has not ended, so it is not a death
    assert all(f.index != 3 for f in flats)


def test_find_repairs_window_and_kinds() -> None:
    turns = [
        Turn(speaker_id=A, text="ok"),
        Turn(speaker_id=B, text="yeah"),
        Turn(speaker_id=A, text="hm"),
        Turn(speaker_id=B, text="Did you ever hear back from the landlord?"),
        Turn(speaker_id=A, text="k"),
        Turn(speaker_id=B, text="sure"),
        Turn(speaker_id=A, text="fine"),
        Turn(speaker_id=B, text="ok"),
        Turn(speaker_id=A, text="So I finally got around to fixing the bike and it rides again."),
    ]
    flats = [
        FlatMoment(index=i, kind="one_word", speaker_id=t.speaker_id)
        for i, t in enumerate(turns)
        if i not in (3, 8)
    ]
    out = find_repairs(turns, flats, window=3)
    by_index = {f.index: f for f in out}
    assert by_index[0].repaired_at == 3 and by_index[0].latency_turns == 3
    assert by_index[2].repaired_at == 3 and by_index[2].latency_turns == 1
    assert by_index[4].repaired_at is None  # 8 is outside the 3-turn window
    assert by_index[5].repaired_at == 8 and by_index[5].repaired_by == A
    # repaired: 0->3, 1->3, 2->3, 5->8, 6->8, 7->8; unrepaired: 4
    assert repair_rate_from(out) == 6 / 7
    assert repair_latency_from(out) == (3 + 2 + 1 + 3 + 2 + 1) / 6


def test_rates_on_no_flats() -> None:
    assert repair_rate_from([]) == 1.0
    assert repair_latency_from([]) == 0.0
    t = dyad(
        [
            (A, "I started climbing at the gym last month and I love it."),
            (B, "Which gym did you end up joining after all that searching?"),
        ]
    )
    assert repair_rate(t) == 1.0
    assert repair_latency_turns(t) == 0.0


def test_repair_rate_flow_beats_dead(flow: Transcript, dead: Transcript) -> None:
    assert repair_rate(flow) > repair_rate(dead)
    assert repair_rate(flow) == 1.0
    assert repair_rate(dead) < 0.25
    assert 1.0 <= repair_latency_turns(flow) <= 3.0


def test_dead_has_many_flats_and_almost_no_repair(dead: Transcript) -> None:
    flats = analyze_repairs(dead)
    assert len(flats) >= 15
    assert sum(f.repaired for f in flats) <= 2
    assert sum(repair_by_speaker(dead).values()) == sum(f.repaired for f in flats)


def test_repair_by_speaker_interview_is_one_sided(interview: Transcript) -> None:
    who = repair_by_speaker(interview)
    assert set(who) == {A, B}
    assert who[A] > 0
    assert who[B] == 0


def test_repair_by_speaker_flow_is_shared(flow: Transcript) -> None:
    who = repair_by_speaker(flow)
    assert who[A] > 0 and who[B] > 0
    assert B in flow.participants

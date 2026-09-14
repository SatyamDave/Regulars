"""Synthetic round-robin dinners with planted rater, ratee, group and relationship effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regulars.eval.records import AssemblyMethod, Attendance, LabeledDyad, Membership, Rsvp

GROUP_SIZE = 6


@dataclass(frozen=True, slots=True)
class SynthDinners:
    dyads: list[LabeledDyad]
    memberships: list[Membership]
    attendance: list[Attendance]
    rsvps: list[Rsvp]
    y_hats_all_pairs: list[float]


def make_dyads(
    n_groups: int = 10,
    seed: int = 0,
    beta: float = 1.0,
    sd_rater: float = 0.8,
    sd_ratee: float = 0.6,
    sd_group: float = 0.8,
    sd_resid: float = 1.0,
    sd_yhat_group: float = 0.6,
    random_share: float = 1.0,
    scale_1_7: bool = True,
) -> list[LabeledDyad]:
    """``n_groups`` dinners of six, everyone rates everyone (30 directed dyads per dinner).

    ``y = beta * y_hat + rater + ratee + group + resid`` with ``y_hat`` carrying a group-level
    component (``sd_yhat_group``) so both regressor and error are clustered within group.
    The first ``random_share`` of groups are the exploration arm.
    """
    rng = np.random.default_rng(seed)
    n_users = n_groups * GROUP_SIZE
    rater_eff = rng.normal(0, sd_rater, n_users)
    ratee_eff = rng.normal(0, sd_ratee, n_users)
    dyads: list[LabeledDyad] = []
    n_random = round(n_groups * random_share)
    for g in range(n_groups):
        method: AssemblyMethod = "random" if g < n_random else "model"
        members = list(range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE))
        g_eff = rng.normal(0, sd_group)
        g_yhat = rng.normal(0, sd_yhat_group)
        for a in members:
            for b in members:
                if a == b:
                    continue
                y_hat = g_yhat + rng.normal(0, 1.0)
                latent = beta * y_hat + rater_eff[a] + ratee_eff[b] + g_eff
                latent += rng.normal(0, sd_resid)
                felt = int(np.clip(np.round(4 + latent), 1, 7)) if scale_1_7 else None
                dyads.append(
                    LabeledDyad(
                        group_id=f"g{g}",
                        plan_id=f"p{g}-1",
                        rater=f"u{a}",
                        ratee=f"u{b}",
                        y_hat=float(y_hat),
                        y_hat_calibrated=float(1 / (1 + np.exp(-y_hat))),
                        would_see_again=bool(latent > 0),
                        felt_connected_1_7=felt,
                        assembly_method=method,
                        round_index=1,
                    )
                )
    return dyads


def make_retention(
    n_groups_per_arm: int = 50,
    p_model: float = 0.75,
    p_random: float = 0.60,
    seed: int = 0,
    show_rate: float = 0.9,
) -> SynthDinners:
    """Two arms of dinners; each round-1 member returns for round 2 with the arm's probability."""
    rng = np.random.default_rng(seed)
    memberships: list[Membership] = []
    attendance: list[Attendance] = []
    rsvps: list[Rsvp] = []
    uid = 0
    for arm, p in (("model", p_model), ("random", p_random)):
        method: AssemblyMethod = "model" if arm == "model" else "random"
        for g in range(n_groups_per_arm):
            gid = f"{arm}-g{g}"
            p1, p2 = f"{gid}-p1", f"{gid}-p2"
            for _ in range(GROUP_SIZE):
                u = f"u{uid}"
                uid += 1
                back = bool(rng.random() < p)
                memberships.append(
                    Membership(
                        group_id=gid,
                        user_id=u,
                        joined_round=1,
                        left_round=None if back else 2,
                        exit_reason=None if back else "no_show",
                        assembly_method=method,
                    )
                )
                showed_r1 = bool(rng.random() < show_rate)
                rsvps.append(Rsvp(plan_id=p1, user_id=u, state="confirmed", paid=True))
                attendance.append(
                    Attendance(
                        plan_id=p1, group_id=gid, user_id=u, round_index=1, attended=showed_r1
                    )
                )
                if back:
                    rsvps.append(Rsvp(plan_id=p2, user_id=u, state="confirmed", paid=True))
                    attendance.append(
                        Attendance(
                            plan_id=p2, group_id=gid, user_id=u, round_index=2, attended=True
                        )
                    )
    dyads = make_dyads(n_groups=10, seed=seed, random_share=0.5)
    return SynthDinners(
        dyads=dyads,
        memberships=memberships,
        attendance=attendance,
        rsvps=rsvps,
        y_hats_all_pairs=[d.y_hat for d in dyads] + list(rng.normal(0, 1, 500)),
    )

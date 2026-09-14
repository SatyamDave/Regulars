"""The five tools (§9). Resist adding a sixth.

Every call writes exactly one audit row through the store before returning. Tools are the only
place the bot touches the transport, the deposit collector, or the plan store with intent; flows
and the router compose these five and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from regulars.bot.clock import Clock
from regulars.bot.deposits import DepositCollector, DepositRequest
from regulars.bot.plan import Plan, Slot, mark_deposit_requested, new_plan
from regulars.bot.store import AuditEntry, BotStore, GroupChat
from regulars.bot.transport import Transport

ACTOR: Final = "bot"
TOOL_NAMES: Final[tuple[str, ...]] = (
    "send_message",
    "create_plan",
    "collect_deposit",
    "remove_member",
    "log_signal",
)


class ToolError(RuntimeError):
    """A tool was called with a target the store does not know."""


class BotTools:
    def __init__(
        self, transport: Transport, store: BotStore, deposits: DepositCollector, clock: Clock
    ) -> None:
        self.transport = transport
        self.store = store
        self.deposits = deposits
        self.clock = clock
        self._plan_counter = 0

    @staticmethod
    def tools() -> tuple[str, ...]:
        """The public tool list. Length five, by design."""
        return TOOL_NAMES

    # --- 1. send_message -----------------------------------------------------------------------

    async def send_message(
        self, group_id: str | None, body: str, *, to_user: str | None = None
    ) -> str:
        """Post to the group chat, or DM one member when ``to_user`` is given."""
        if to_user is not None:
            handle = self.store.handle_for_user(to_user)
            if handle is None:
                raise ToolError(f"no handle for user {to_user!r}")
            message_id = await self.transport.send_dm(handle, body)
        else:
            group = self._group(group_id)
            message_id = await self.transport.send(group.chat_id, body)
        self._audit(
            "send_message",
            "DM" if to_user else "group message",
            {"body": body, "message_id": message_id},
            subject_user_id=to_user,
            group_id=group_id,
        )
        return message_id

    # --- 2. create_plan ------------------------------------------------------------------------

    async def create_plan(
        self,
        group_id: str,
        options: Sequence[Slot],
        deadline: datetime,
        *,
        proposer: str = ACTOR,
        constraints: Mapping[str, str] | None = None,
        plan_id: str | None = None,
    ) -> Plan:
        group = self._group(group_id)
        self._plan_counter += 1
        plan = new_plan(
            plan_id or f"plan-{group_id}-{self._plan_counter}",
            group_id,
            [m.user_id for m in group.members],
            options,
            deadline,
            proposer=proposer,
            constraints=constraints,
        )
        self.store.save_plan(plan)
        self._audit(
            "create_plan",
            f"{len(options)} candidate slots, poll closes {deadline.isoformat()}",
            {"slots": [s.model_dump(mode="json") for s in options], "proposer": proposer},
            group_id=group_id,
            plan_id=plan.id,
        )
        return plan

    # --- 3. collect_deposit --------------------------------------------------------------------

    async def collect_deposit(
        self, plan_id: str, user_id: str, amount_cents: int
    ) -> DepositRequest:
        plan = self.store.get_plan(plan_id)
        if plan is None:
            raise ToolError(f"unknown plan {plan_id!r}")
        if plan.locked_slot_id is None or plan.outcome != "open":
            raise ToolError(f"plan {plan_id!r} is not ready for deposits")
        if plan.commitment.get(user_id) not in ("invited", "deposit_requested"):
            raise ToolError(
                f"{user_id!r} cannot be charged from state {plan.commitment.get(user_id)!r}"
            )
        # Validate before the collector is called: never open a hold we cannot record.
        request = await self.deposits.collect(plan_id, user_id, amount_cents)
        self.store.save_plan(
            mark_deposit_requested(plan, user_id, request.intent_id, self.clock.now())
        )
        self._audit(
            "collect_deposit",
            f"{amount_cents} cents requested; {request.amount_after_credit_cents} after credit",
            {
                "intent_id": request.intent_id,
                "amount_cents": amount_cents,
                "amount_after_credit_cents": request.amount_after_credit_cents,
            },
            subject_user_id=user_id,
            group_id=plan.group_id,
            plan_id=plan_id,
        )
        return request

    # --- 4. remove_member ----------------------------------------------------------------------

    async def remove_member(self, group_id: str, user_id: str, reason: str) -> None:
        group = self._group(group_id)
        member = group.member(user_id)
        if member is None:
            raise ToolError(f"{user_id!r} is not in group {group_id!r}")
        await self.transport.remove_participant(group.chat_id, member.handle)
        self.store.save_group(
            GroupChat(
                group_id=group.group_id,
                chat_id=group.chat_id,
                members=tuple(m for m in group.members if m.user_id != user_id),
                last_message_at_ms=group.last_message_at_ms,
                last_sender=group.last_sender,
            )
        )
        self._audit(
            "member_removed",
            reason,
            {"handle": member.handle},
            subject_user_id=user_id,
            group_id=group_id,
        )

    # --- 5. log_signal -------------------------------------------------------------------------

    async def log_signal(
        self, user_id: str, signal_type: str, value: float, *, group_id: str | None = None
    ) -> AuditEntry:
        return self._audit(
            "log_signal",
            signal_type,
            {"signal_type": signal_type, "value": value},
            subject_user_id=user_id,
            group_id=group_id,
        )

    # --- internals -----------------------------------------------------------------------------

    def _group(self, group_id: str | None) -> GroupChat:
        group = self.store.get_group(group_id) if group_id is not None else None
        if group is None:
            raise ToolError(f"unknown group {group_id!r}")
        return group

    def _audit(
        self,
        decision_type: str,
        reason: str,
        payload: Mapping[str, Any],
        *,
        subject_user_id: str | None = None,
        group_id: str | None = None,
        plan_id: str | None = None,
    ) -> AuditEntry:
        return self.store.write_audit(
            actor=ACTOR,
            decision_type=decision_type,
            reason=reason,
            payload=payload,
            subject_user_id=subject_user_id,
            group_id=group_id,
            plan_id=plan_id,
        )

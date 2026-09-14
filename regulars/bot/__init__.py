"""Agent runtime (§9): five tools, a Plan record, a swappable Transport, audit rows everywhere.

Seams other subpackages implement or consume:

- :class:`BotStore` (persistence; ``regulars.db`` adapts its repo to this)
- :class:`DepositCollector` (``regulars.payments``)
- :class:`Transport` (BlueBubbles today)
- events: :class:`OnboardingEvent`, :class:`PruneRequest`, :class:`BackfillRequest`,
  :class:`LegibilityRequest`
"""

from regulars.bot.clock import Clock, FrozenClock, SystemClock
from regulars.bot.deposits import DepositCollector, DepositRequest, FakeDepositCollector
from regulars.bot.events import (
    BackfillRequest,
    BotEvent,
    DirectedRating,
    InterviewAnswer,
    LegibilityRequest,
    OnboardingEvent,
    PruneRequest,
)
from regulars.bot.flows import GroupOpener, LockFlow, MorningAfter
from regulars.bot.onboarding import Onboarding, OnboardingState
from regulars.bot.plan import Plan, Slot
from regulars.bot.router import Router
from regulars.bot.runtime import BotRuntime, Flows, build_runtime
from regulars.bot.store import BotStore, GroupChat, InMemoryBotStore, Member
from regulars.bot.tools import TOOL_NAMES, BotTools
from regulars.bot.transport import (
    BlueBubblesTransport,
    FakeTransport,
    InboundMessage,
    Transport,
    parse_bluebubbles_webhook,
)

__all__ = [
    "TOOL_NAMES",
    "BackfillRequest",
    "BlueBubblesTransport",
    "BotEvent",
    "BotRuntime",
    "BotStore",
    "BotTools",
    "Clock",
    "DepositCollector",
    "DepositRequest",
    "DirectedRating",
    "FakeDepositCollector",
    "FakeTransport",
    "Flows",
    "FrozenClock",
    "GroupChat",
    "GroupOpener",
    "InMemoryBotStore",
    "InboundMessage",
    "InterviewAnswer",
    "LegibilityRequest",
    "LockFlow",
    "Member",
    "MorningAfter",
    "Onboarding",
    "OnboardingEvent",
    "OnboardingState",
    "Plan",
    "PruneRequest",
    "Router",
    "Slot",
    "SystemClock",
    "Transport",
    "build_runtime",
    "parse_bluebubbles_webhook",
]

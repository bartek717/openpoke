"""Data factories for generating realistic rosters and conversation logs at scale."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

from server.services.execution.roster import AgentRoster


_FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mona", "Nick", "Olivia", "Paul",
    "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zach", "Amara", "Blake", "Cleo", "Devin", "Elara", "Finn",
    "Gemma", "Hugo", "Ivy", "Jasper", "Kaia", "Liam", "Maya", "Noah",
]

_LAST_NAMES = [
    "Anderson", "Baker", "Chen", "Davis", "Evans", "Foster", "Garcia",
    "Harris", "Ibrahim", "Johnson", "Kim", "Lee", "Martinez", "Nguyen",
    "O'Brien", "Patel", "Quinn", "Roberts", "Singh", "Thompson", "Ueda",
    "Vasquez", "Williams", "Xu", "Yamamoto", "Zhang", "Ali", "Brown",
    "Clark", "Diaz", "Edwards", "Freeman", "Green", "Hill", "Ito",
]

_TOPICS = [
    "Monday standup", "Q3 budget review", "project timeline",
    "PTO request", "office supplies reorder", "weekly progress",
    "contract signing", "team lunch", "client onboarding",
    "product demo", "sprint retro", "design review",
    "performance review", "travel reimbursement", "invoice follow-up",
    "partnership proposal", "marketing campaign", "hiring update",
    "security audit", "release planning", "customer feedback",
    "vendor negotiation", "board meeting prep", "training session",
]

_AGENT_TEMPLATES = [
    "Email to {first} {last}",
    "Follow up with {first} {last}",
    "{first} {last} - {topic}",
    "Draft reply to {first}",
    "Reminder: {topic}",
    "{topic} discussion",
    "Meeting notes - {topic}",
    "Schedule {topic} with {first}",
]


def generate_agent_names(count: int, seed: int = 42) -> List[str]:
    """Generate `count` unique realistic agent names."""
    rng = random.Random(seed)
    names: List[str] = []
    seen: set[str] = set()

    while len(names) < count:
        template = rng.choice(_AGENT_TEMPLATES)
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        topic = rng.choice(_TOPICS)
        name = template.format(first=first, last=last, topic=topic)

        if name not in seen:
            seen.add(name)
            names.append(name)

    return names


def populate_roster(roster: AgentRoster, count: int, seed: int = 42) -> List[str]:
    """Populate a roster with `count` agents and return the names."""
    names = generate_agent_names(count, seed=seed)
    for name in names:
        roster.add_agent(name)
    return names


_USER_MESSAGES = [
    "Can you draft an email to {first} about {topic}?",
    "What's the latest on {topic}?",
    "Remind me about {topic} tomorrow at 9am.",
    "Forward the last email from {first} to my manager.",
    "Set up a meeting with {first} {last} for next week.",
    "Did {first} reply to my email about {topic}?",
    "Check my inbox for anything from {first} {last}.",
    "Cancel my reminder about {topic}.",
]

_POKE_REPLIES = [
    "On it, drafting that email now.",
    "Here's what I found - {first} sent an update about {topic} yesterday.",
    "Done! Reminder set for tomorrow at 9am.",
    "I've forwarded that email. Anything else?",
    "Meeting request sent to {first} for next Tuesday.",
    "Looks like {first} hasn't replied yet. Want me to follow up?",
    "Found 3 emails from {first} {last} in the past week.",
    "Reminder cancelled.",
]


def generate_conversation_log(turn_count: int, seed: int = 42) -> str:
    """Generate a realistic conversation log with `turn_count` user/reply pairs."""
    rng = random.Random(seed)
    lines: List[str] = []

    for i in range(turn_count):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        topic = rng.choice(_TOPICS)

        hour = 9 + (i % 10)
        minute = rng.randint(0, 59)
        timestamp = f"2026-04-{10 + i // 20:02d} {hour:02d}:{minute:02d}:00"

        user_msg = rng.choice(_USER_MESSAGES).format(first=first, last=last, topic=topic)
        poke_msg = rng.choice(_POKE_REPLIES).format(first=first, last=last, topic=topic)

        lines.append(f'<user_message timestamp="{timestamp}">{user_msg}</user_message>')
        lines.append(f'<poke_reply timestamp="{timestamp}">{poke_msg}</poke_reply>')

    return "\n".join(lines)


def write_conversation_log(path: Path, turn_count: int, seed: int = 42) -> str:
    """Write a conversation log file and return its content."""
    content = generate_conversation_log(turn_count, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    return content

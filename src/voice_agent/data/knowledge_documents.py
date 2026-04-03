from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)

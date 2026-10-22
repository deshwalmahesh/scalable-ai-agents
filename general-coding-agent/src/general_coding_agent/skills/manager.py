"""Progressive disclosure skill manager for general_coding_agent.

Follows the Agent Skills specification:
- Discovers skills in skills/*/SKILL.md
- Extracts YAML frontmatter for lightweight catalogue prompt (~30 tokens)
- Lazily loads full markdown instructions only when a skill is activated
"""

import logging
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel
from general_coding_agent.config import settings

log = logging.getLogger(__name__)


class SkillMetadata(BaseModel):
    """Metadata extracted from SKILL.md frontmatter."""

    name: str
    description: str
    directory: Path
    file_path: Path


class SkillManager:
    """Manages skill discovery, cataloging, and lazy instruction retrieval."""

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = (skills_dir or settings.skills_dir).resolve()
        self._skills: dict[str, SkillMetadata] = {}
        self.refresh()

    def refresh(self) -> None:
        """Scan skills directory and parse frontmatters."""
        self._skills.clear()
        if not self.skills_dir.exists():
            return

        for skill_file in self.skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8")
                frontmatter, _ = self._parse_frontmatter(content)
                name = frontmatter.get("name", skill_file.parent.name)
                desc = frontmatter.get("description", "No description provided.")
                self._skills[name] = SkillMetadata(
                    name=name,
                    description=desc,
                    directory=skill_file.parent,
                    file_path=skill_file,
                )
            except Exception as e:  # noqa: BLE001 - one bad skill must not break discovery
                log.warning("Failed to parse skill at '%s': %s", skill_file, e)

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Split YAML frontmatter from Markdown body."""
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, content

        frontmatter_lines = []
        body_start = 1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                body_start = i + 1
                break
            frontmatter_lines.append(lines[i])

        try:
            frontmatter = yaml.safe_load("\n".join(frontmatter_lines)) or {}
        except Exception:
            frontmatter = {}

        body = "\n".join(lines[body_start:]).strip()
        return frontmatter, body

    def list_skills(self) -> list[SkillMetadata]:
        """Return list of discovered skills."""
        return list(self._skills.values())

    def get_catalogue_prompt(self) -> str:
        """Generate a minimal skill catalogue for system/supervisor prompt."""
        if not self._skills:
            return "No specialized skills currently loaded."

        lines = ["Available Specialized Skills (activate via name when needed):"]
        for s in self._skills.values():
            lines.append(f"- `{s.name}`: {s.description.strip()}")
        return "\n".join(lines)

    def load_skill_instructions(self, skill_name: str) -> str | None:
        """Lazily load full instructions of a skill by name."""
        skill = self._skills.get(skill_name)
        if not skill or not skill.file_path.exists():
            return None

        content = skill.file_path.read_text(encoding="utf-8")
        _, body = self._parse_frontmatter(content)
        return body


# Default singleton
skill_manager = SkillManager()

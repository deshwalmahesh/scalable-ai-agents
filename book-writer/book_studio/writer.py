"""Research and write a three-chapter book with a checked LangGraph workflow."""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from tavily import TavilyClient

from .config import DEFAULTS, DEFAULT_BRIEF, DEFAULT_ENV_FILE, load_config, load_profiles, source_allowed


LOG = logging.getLogger("book_writer")
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
OFFICIAL_DOMAINS = tuple(DEFAULTS["profiles"]["upi"]["research"]["domains"])


class ChapterPlan(BaseModel):
    title: str
    focus: str
    search_queries: list[str] = Field(min_length=2, max_length=3)


class Outline(BaseModel):
    title: str
    chapters: list[ChapterPlan] = Field(min_length=3, max_length=3)


class ValidationResult(BaseModel):
    passed: bool
    issues: list[str]


class ProseReplacement(BaseModel):
    old: str
    new: str


class ProseFix(BaseModel):
    replacements: list[ProseReplacement]


class FactVerdict(BaseModel):
    supported: bool = Field(description="True only if every factual claim in the passage is cited and supported")
    issues: list[str] = Field(description="Only unsupported, false, or uncited claims; empty when supported")


class CitationNeed(BaseModel):
    factual: bool


class CitationMatch(BaseModel):
    source_number: int | None


class TakeawayChoice(BaseModel):
    index: int


class ResearchReview(BaseModel):
    sufficient: bool
    gaps: list[str]
    follow_up_queries: list[str] = Field(max_length=2)
    usable_source_numbers: list[int]


class Source(BaseModel):
    number: int
    publisher: str
    title: str
    url: str
    evidence: str


class BookState(TypedDict):
    job_id: str
    brief: str
    profile: dict
    outline: Outline | None
    chapter_index: int
    sources: list[Source]
    research_round: int
    search_queries: list[str]
    coverage_ok: bool
    draft: str
    draft_round: int
    review_round: int
    structure_issues: list[str]
    reference_issues: list[str]
    editor_issues: list[str]
    scope_issues: list[str]
    fact_issues: list[str]
    unsupported_sentences: list[str]
    citation_fixes: list[tuple[str, int]]
    takeaway_repairs: int
    expansion_round: int
    editor_repairs: int
    consistency_issues: list[str]
    chapters: list[str]
    human_feedback: list[str]
    human_review_round: int
    human_approved: bool


def publisher_for(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if official_https(url, ("npci.org.in",)):
        return "National Payments Corporation of India (NPCI)"
    if official_https(url, ("rbi.org.in",)):
        return "Reserve Bank of India (RBI)"
    if official_https(url, ("pib.gov.in",)):
        return "Press Information Bureau (PIB)"
    return host.removeprefix("www.")


def official_https(url: str, domains: list[str] | tuple[str, ...] = OFFICIAL_DOMAINS) -> bool:
    return source_allowed(url, domains)


def source_key(url: str) -> str:
    parsed = urlparse(url)
    if official_https(url, ("pib.gov.in",)):
        match = re.search(r"(?i)(?:PRID|NoteId)=(\d+)", parsed.query)
        if match:
            return f"pib.gov.in:{match.group(1)}"
    return url


def source_title(item: dict) -> str:
    title = re.sub(r"\s+", " ", item.get("title") or "Untitled source").strip()
    if publisher_for(item["url"]).startswith("Press Information Bureau") and title.startswith("Press Release"):
        lines = [line.strip() for line in (item.get("raw_content") or "").splitlines() if line.strip()]
        if len(lines) > 1 and lines[0].lower() == "azadi ka amrit mahotsav":
            return lines[1]
    return title


def evidence_packet(sources: list[Source]) -> str:
    return "\n\n".join(
        f"[{source.number}] {source.publisher} — {source.title}\nURL: {source.url}\nEvidence:\n{source.evidence}"
        for source in sources
    )


def prose_word_count(draft: str) -> int:
    prose = draft.split("Takeaway:", 1)[0]
    return len(re.findall(r"\b[\w’'-]+\b", re.sub(r"\[\d+\]", "", prose)))


def format_issues(draft: str) -> list[str]:
    issues = []
    lines = draft.strip().splitlines()
    takeaway_lines = [line for line in lines if line.startswith("Takeaway:")]
    if len(takeaway_lines) != 1 or lines[-1] != takeaway_lines[0]:
        issues.append("End with exactly one Takeaway: line, after the prose")
    prose = draft.split("Takeaway:", 1)[0]
    if len(re.split(r"\n\s*\n", prose.strip())) < 4:
        issues.append("Break the chapter into at least four readable prose paragraphs")
    words = prose_word_count(draft)
    if not 600 <= words <= 900:
        issues.append(f"Chapter prose is {words} words; it must be 600–900")
    if re.search(r"(?m)^\s*(?:[-*]\s|\d+\.\s|#{1,6}\s)", draft):
        issues.append("Use flowing prose; remove lists and headings from the chapter body")
    if "References" in draft or "http://" in draft or "https://" in draft:
        issues.append("Do not write references or URLs in the chapter body")
    return issues


def reference_issues(draft: str, sources: list[Source], domains: list[str] | tuple[str, ...] = OFFICIAL_DOMAINS) -> list[str]:
    issues = []
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", draft)}
    available = {source.number for source in sources}
    if not cited:
        issues.append("Cite factual claims with numbered source markers")
    if cited - available:
        issues.append(f"Unknown citation numbers: {sorted(cited - available)}")
    if len(available) != len(sources):
        issues.append("Source numbers must be unique within a chapter")
    for source in sources:
        if source.number in cited and (not official_https(source.url, domains) or len(source.evidence.strip()) < 500 or not source.title.strip()):
            issues.append(f"Citation [{source.number}] has no validated public source, title, or extractable evidence")
    return issues


def checked_issues(result: ValidationResult, label: str) -> list[str]:
    if not isinstance(result, ValidationResult):
        raise RuntimeError(f"{label} did not return a valid decision")
    if result.passed:
        return []
    if not result.issues:
        raise RuntimeError(f"{label} failed without actionable feedback")
    return result.issues


def remove_unsupported(draft: str, sentences: list[str]) -> str:
    for sentence in sentences:
        if sentence not in draft:
            raise RuntimeError(f"Fact-checker quote was not found in the draft: {sentence[:80]}")
        draft = draft.replace(sentence, "", 1)
    draft = re.sub(r" {2,}", " ", draft)
    return "\n".join(line.strip() for line in draft.splitlines()).strip()

def add_citations(draft: str, fixes: list[tuple[str, int]]) -> str:
    for sentence, number in fixes:
        if sentence not in draft:
            raise RuntimeError(f"Citation repair quote was not found in the draft: {sentence[:80]}")
        draft = draft.replace(sentence, f"{sentence} [{number}]", 1)
    return draft


def render_chapter(number: int, plan: ChapterPlan, draft: str, sources: list[Source]) -> str:
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", draft)}
    references = "\n".join(
        f"[{source.number}] {source.publisher}, *{source.title}*. <{source.url}>"
        for source in sources if source.number in cited
    )
    return f"## Chapter {number}: {plan.title}\n\n{draft.strip()}\n\n### References\n\n{references}"


def initial_state(brief: str, job_id: str = "", profile: dict | None = None) -> BookState:
    return {
        "job_id": job_id, "brief": brief, "profile": profile or {}, "outline": None, "chapter_index": 0, "sources": [],
        "research_round": 0, "search_queries": [], "coverage_ok": False,
        "draft": "", "draft_round": 0, "review_round": 0, "structure_issues": [], "reference_issues": [],
        "editor_issues": [], "scope_issues": [], "fact_issues": [],
        "unsupported_sentences": [], "citation_fixes": [], "takeaway_repairs": 0,
        "expansion_round": 0, "editor_repairs": 0, "consistency_issues": [], "chapters": [],
        "human_feedback": [], "human_review_round": 0, "human_approved": False,
    }


def render_book(state: BookState) -> str:
    outline = state["outline"]
    if outline is None or len(state["chapters"]) != 3:
        raise RuntimeError("Book workflow did not complete three chapters")
    return f"# {outline.title}\n\n" + "\n\n".join(state["chapters"]) + "\n"


class BookWriter:
    def __init__(self, config: dict[str, str], profiles: dict | None = None) -> None:
        self.profiles = profiles or load_profiles(config)
        self.brief = self.profiles["default_brief"]
        self.profile = self.profile_for(self.brief)
        self.model = ChatOpenAI(
            model=MODEL,
            base_url=config["VLLM_BASE_URL"],
            api_key=config["VLLM_API_KEY"],
            temperature=0,
            max_tokens=5000,
            timeout=180,
            max_retries=1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.structured_outline = self.model.with_structured_output(Outline, method="function_calling")
        self.structured_prose_fix = self.model.with_structured_output(ProseFix, method="function_calling")
        self.structured_validation = self.model.with_structured_output(ValidationResult, method="function_calling")
        self.structured_fact_verdict = self.model.with_structured_output(FactVerdict, method="function_calling")
        self.structured_citation_need = self.model.with_structured_output(CitationNeed, method="function_calling")
        self.structured_citation_match = self.model.with_structured_output(CitationMatch, method="function_calling")
        self.structured_takeaway_choice = self.model.with_structured_output(TakeawayChoice, method="function_calling")
        self.structured_research_review = self.model.with_structured_output(ResearchReview, method="function_calling")
        self.search = TavilyClient(api_key=config["TAVILY_API_KEY"])

    def profile_for(self, brief: str) -> dict:
        name = "upi" if brief.strip() == DEFAULT_BRIEF.strip() else "general"
        return self.profiles["profiles"][name]

    def messages(self, name: str, **values) -> list[tuple[str, str]]:
        values = {"brief": self.brief, **values}
        pair = self.profile["prompts"][name]
        return [(role, pair[role].format(**values)) for role in ("system", "human")]

    def plan(self, brief: str) -> Outline:
        self.brief = brief
        outline = self.structured_outline.invoke(self.messages('planner', brief=brief))
        if not isinstance(outline, Outline):
            raise RuntimeError("Planner did not return a valid outline")
        title_match = re.search(r"(?im)^Title:\s*(.+)$", brief)
        if title_match and outline.title.strip() != title_match.group(1).strip():
            raise RuntimeError("Planner changed the requested book title")
        if len({chapter.title.lower() for chapter in outline.chapters}) != 3:
            raise RuntimeError("Planner returned duplicate chapter titles")
        return outline

    def research(
        self,
        plan: ChapterPlan,
        queries: list[str] | None = None,
        existing: list[Source] | None = None,
        chapter_index: int = 0,
    ) -> list[Source]:
        sources = list(existing or [])
        if len(sources) >= 8:
            return sources
        known_keys = {source_key(source.url) for source in sources}
        results: dict[str, dict] = {}
        policy = self.profile["research"]
        for query in queries or plan.search_queries:
            response = self.search.search(
                query,
                search_depth="basic",
                max_results=5,
                include_raw_content="text",
                include_domains=policy["domains"],
                timeout=30,
            )
            for item in response.get("results", []):
                url = item.get("url", "")
                key = source_key(url)
                if official_https(url, policy["domains"]) and key not in results and key not in known_keys:
                    results[key] = item

        ranked = sorted(results.values(), key=lambda item: float(item.get("score") or 0), reverse=True)
        excluded = policy["excluded_titles"][chapter_index]
        if excluded:
            ranked = [item for item in ranked if not re.search(excluded, item.get("title") or "", re.IGNORECASE)]
        starters = policy["seeds"][chapter_index]
        seeded = [{"url": url, "title": title} for url, title in starters if source_key(url) not in known_keys]
        seed_keys = {source_key(item["url"]) for item in seeded}
        candidates = (seeded + [item for item in ranked if source_key(item["url"]) not in seed_keys])[:8]
        if not candidates:
            if sources:
                return sources
            raise RuntimeError(f"No research results for {plan.title}")
        # Keep page order: query-focused chunks can separate figures from table headers.
        extracted = self.search.extract(
            [item["url"] for item in candidates],
            extract_depth="advanced",
            format="text",
            chunks_per_source=5,
            timeout=45,
        )
        passages = {item["url"]: item.get("raw_content", "") for item in extracted.get("results", [])}
        for item in candidates:
            evidence = passages.get(item["url"], "")
            if len(evidence.strip()) < 500:
                LOG.info("Skipping source without extractable evidence: %s", item["url"])
                continue
            sources.append(Source(
                number=len(sources) + 1,
                publisher=publisher_for(item["url"]),
                title=source_title(item),
                url=item["url"],
                evidence=evidence[:4500],
            ))
            if len(sources) >= (8 if existing else 6):
                break
        LOG.info("Research for %s: %d extracted sources", plan.title, len(sources))
        return sources

    def evaluate_research(self, plan: ChapterPlan, sources: list[Source]) -> ResearchReview:
        if len(sources) < 2:
            return ResearchReview(
                sufficient=False,
                gaps=["Fewer than two extractable sources"],
                follow_up_queries=[query.format(title=plan.title, focus=plan.focus)
                                   for query in self.profile["research"]["fallback_queries"]],
                usable_source_numbers=[source.number for source in sources],
            )
        result = self.structured_research_review.invoke(self.messages('research_review', title=plan.title, focus=plan.focus, evidence=evidence_packet(sources)))
        if not isinstance(result, ResearchReview):
            raise RuntimeError("Research evaluator did not return a valid decision")
        if not result.sufficient and not result.follow_up_queries:
            raise RuntimeError(f"Research evaluator found gaps without search queries: {result.gaps}")
        return result

    def draft(
        self,
        plan: ChapterPlan,
        sources: list[Source],
        previous_chapters: list[str],
        previous_draft: str = "",
        feedback: list[str] | None = None,
    ) -> str:
        if previous_draft:
            messages = self.messages('revision', evidence=evidence_packet(sources), previous_draft=previous_draft, feedback='\n'.join(feedback or []))
        else:
            messages = self.messages('writer', title=plan.title, focus=plan.focus, evidence=evidence_packet(sources), chapter_number=len(previous_chapters) + 1)
        response = self.model.invoke(messages)
        if not isinstance(response.content, str):
            raise RuntimeError("Writer returned non-text content")
        draft = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        if not draft:
            raise RuntimeError("Writer returned an empty chapter")
        return draft

    def takeaway(self, draft: str, plan: ChapterPlan) -> str:
        sentences = [sentence.strip() for paragraph in re.split(r"\n\s*\n", draft)
                     for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"₹])|(?<=\])\s+(?=[A-Z“\"₹])", paragraph)
                     if 7 <= len(sentence.split()) <= 40
                     and not re.match(r"(?i)^(?:but|and|however)\b", sentence.strip())]
        if len(sentences) > 1:
            sentences = sentences[:-1]
        if not sentences:
            raise RuntimeError("No sentence available for a Takeaway")
        choices = "\n".join(f"{index}: {sentence}" for index, sentence in enumerate(sentences))
        choice = self.structured_takeaway_choice.invoke(self.messages('takeaway', focus=plan.focus, choices=choices))
        if not isinstance(choice, TakeawayChoice) or not 0 <= choice.index < len(sentences):
            raise RuntimeError("Writer did not select an existing Takeaway sentence")
        return f"Takeaway: {sentences[choice.index]}"

    def expand(self, draft: str, plan: ChapterPlan, sources: list[Source], feedback: list[str] | None = None) -> str:
        marker = re.search(r"(?m)^Takeaway:", draft)
        if marker is None:
            raise RuntimeError("Cannot expand a chapter without its Takeaway")
        body, takeaway = draft[:marker.start()].rstrip(), draft[marker.start():].strip()
        needed = min(260, max(140, 760 - prose_word_count(draft)))
        response = self.model.invoke(self.messages('expansion', title=plan.title, focus=plan.focus, needed=needed, feedback='\n'.join(feedback or []), evidence=evidence_packet(sources), draft=draft))
        paragraph = response.content.strip() if isinstance(response.content, str) else ""
        if not paragraph or "Takeaway:" in paragraph or re.search(r"(?m)^\s*(?:[-*]\s|#{1,6}\s)", paragraph):
            raise RuntimeError("Writer did not return one prose expansion")
        return f"{body}\n\n{paragraph}\n\n{takeaway}"

    def edit(self, draft: str, brief: str, previous_chapters: list[str]) -> list[str]:
        style_example = previous_chapters[0].split("\n\n", 1)[-1].split("\n\n", 1)[0][:800] if previous_chapters else ""
        proof = self.structured_validation.invoke(self.messages('language_editor', brief=brief, style_example=style_example, draft=draft))
        coherence_issues = []
        for paragraph in re.split(r"\n\s*\n", draft.split("Takeaway:", 1)[0].strip()):
            coherence = self.structured_validation.invoke(self.messages('coherence_editor', paragraph=paragraph))
            coherence_issues.extend(checked_issues(coherence, "Coherence editor"))
        takeaway_issues = []
        match = re.search(r"(?m)^Takeaway:\s*(.+)$", draft)
        if match:
            takeaway = re.sub(r"\s+", " ", match.group(1)).strip()
            previous_paragraph = re.split(r"\n\s*\n", draft[:match.start()].strip())[-1]
            previous_sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"₹])|(?<=\])\s+(?=[A-Z“\"₹])", previous_paragraph)[-1]
            if takeaway == re.sub(r"\s+", " ", previous_sentence).strip():
                takeaway_issues.append("Takeaway copies the immediately preceding body sentence verbatim")
            if re.match(r"(?i)^(?:But|And|However)\b", takeaway):
                takeaway_issues.append("Takeaway starts with a contrast connector")
        return checked_issues(proof, "Language editor") + coherence_issues + takeaway_issues

    def repair_prose(self, draft: str, issues: list[str]) -> str:
        result = self.structured_prose_fix.invoke(self.messages('prose_repair', issues='\n'.join(issues), draft=draft))
        if not isinstance(result, ProseFix) or not result.replacements or len(result.replacements) > 5:
            raise RuntimeError("Prose repair did not return focused replacements")
        for fix in result.replacements:
            if not fix.old or fix.old not in draft or fix.old == fix.new or len(fix.old) > 700 or len(fix.new) > 700:
                raise RuntimeError("Prose repair did not quote an existing short passage")
            if fix.new and re.findall(r"\[\d+\]", fix.old) != re.findall(r"\[\d+\]", fix.new):
                raise RuntimeError("Prose repair changed a citation marker")
            draft = draft.replace(fix.old, fix.new, 1)
        return draft

    def check_scope(self, draft: str, plan: ChapterPlan, brief: str) -> list[str]:
        issues = []
        for paragraph in re.split(r"\n\s*\n", draft.split("Takeaway:", 1)[0].strip()):
            verdict = self.structured_validation.invoke(self.messages('scope_validator', brief=brief, title=plan.title, focus=plan.focus, paragraph=paragraph))
            issues.extend(checked_issues(verdict, "Scope validator"))
        return issues

    def check_consistency(self, draft: str, previous_chapters: list[str]) -> list[str]:
        if not previous_chapters:
            return []
        opening = re.split(r"(?<=[.!?])\s+", draft, maxsplit=1)[0].strip().casefold()
        for number, chapter in enumerate(previous_chapters, 1):
            prose = chapter.split("\n\n", 1)[-1]
            earlier_opening = re.split(r"(?<=[.!?])\s+", prose, maxsplit=1)[0].strip().casefold()
            if len(opening.split()) >= 12 and opening == earlier_opening:
                return [f"Opening sentence repeats chapter {number} verbatim; write a distinct opening for this chapter"]
        earlier = "\n\n".join(chapter.split("### References", 1)[0] for chapter in previous_chapters)
        verdict = self.structured_validation.invoke(self.messages('consistency_validator', earlier=earlier, draft=draft))
        return checked_issues(verdict, "Consistency validator")

    def fact_check(self, draft: str, sources: list[Source]) -> tuple[list[str], list[str], list[tuple[str, int]]]:
        issues = []
        unsupported = []
        citation_fixes = []
        paragraphs = re.split(r"\n\s*\n|\n(?=Takeaway:)", draft.strip())
        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"₹])|(?<=\])\s+(?=[A-Z“\"₹])", paragraph)
            for sentence in sentences:
                # Layout labels are not claims; retain the original for exact repairs.
                claim = sentence.removeprefix("Takeaway:").strip()
                cited = {int(number) for number in re.findall(r"\[(\d+)\]", sentence)}
                citation_number = None
                if not cited:
                    need = self.structured_citation_need.invoke(self.messages('citation_classifier', claim=claim))
                    if not isinstance(need, CitationNeed):
                        raise RuntimeError("Citation classifier did not return a valid decision")
                    if not need.factual:
                        continue
                    match = self.structured_citation_match.invoke(self.messages('citation_matcher', evidence=evidence_packet(sources), claim=claim))
                    if not isinstance(match, CitationMatch):
                        raise RuntimeError("Citation matcher did not return a valid decision")
                    if match.source_number is not None:
                        if match.source_number not in {source.number for source in sources}:
                            raise RuntimeError("Citation matcher suggested an unknown source")
                        citation_number = match.source_number
                        cited = {citation_number}
                        claim = f"{claim} [{citation_number}]"
                    else:
                        unsupported.append(sentence)
                        issues.append(f"Unsupported uncited claim: {sentence[:120]}")
                        continue
                relevant = [source for source in sources if source.number in cited]
                verdict = self.structured_fact_verdict.invoke(self.messages('fact_checker', evidence=evidence_packet(relevant), claim=claim))
                if not isinstance(verdict, FactVerdict):
                    raise RuntimeError("Fact-checker did not return a valid verdict")
                if not verdict.supported:
                    issues.extend(verdict.issues or [f"Unsupported claim: {sentence[:120]}"])
                    unsupported.append(sentence)
                elif citation_number is not None:
                    citation_fixes.append((sentence, citation_number))
                    issues.append(f"Missing citation [{citation_number}]: {sentence[:120]}")
        return issues, unsupported, citation_fixes

    def build_graph(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        pause_requested: Callable[[str], bool] | None = None,
        human_review: bool = False,
    ) -> CompiledStateGraph:
        def current_plan(state: BookState) -> ChapterPlan:
            outline = state["outline"]
            if outline is None:
                raise RuntimeError("Missing outline")
            return outline.chapters[state["chapter_index"]]

        def plan_node(state: BookState) -> dict:
            outline = self.plan(state["brief"])
            LOG.info("Planned %d chapters", len(outline.chapters))
            return {"outline": outline}

        def research_node(state: BookState) -> dict:
            plan = current_plan(state)
            round_number = state["research_round"] + 1
            LOG.info("Chapter %d research pass %d: %s", state["chapter_index"] + 1, round_number, plan.title)
            sources = self.research(plan, state["search_queries"] or None, state["sources"], state["chapter_index"])
            return {"sources": sources, "research_round": round_number}

        def research_review_node(state: BookState) -> dict:
            review = self.evaluate_research(current_plan(state), state["sources"])
            available = {source.number for source in state["sources"]}
            if set(review.usable_source_numbers) - available:
                raise RuntimeError("Research evaluator selected an unknown source")
            selected = [source for source in state["sources"] if source.number in review.usable_source_numbers]
            selected = [source.model_copy(update={"number": index}) for index, source in enumerate(selected, 1)]
            sufficient = review.sufficient and len(selected) >= 2
            queries = review.follow_up_queries if not sufficient else []
            if not sufficient and not queries:
                queries = current_plan(state).search_queries[:2]
            LOG.info("Chapter %d evidence sufficient: %s; kept %d of %d sources", state["chapter_index"] + 1, sufficient, len(selected), len(state["sources"]))
            return {"sources": selected, "coverage_ok": sufficient, "search_queries": queries}

        def after_research(state: BookState) -> str:
            if state["coverage_ok"]:
                return "writer"
            if state["research_round"] >= 2:
                raise RuntimeError(f"Research gaps remain for chapter {state['chapter_index'] + 1}")
            return "researcher"

        def needs_expansion(state: BookState) -> bool:
            return (prose_word_count(state["draft"]) < 600
                    and len(state["structure_issues"]) == 1
                    and state["structure_issues"][0].startswith("Chapter prose is ")
                    and not (state["reference_issues"] + state["editor_issues"]
                             + state["scope_issues"] + state["fact_issues"]
                             + state["consistency_issues"]))

        def needs_editor_repair(state: BookState) -> bool:
            return (bool(state["editor_issues"])
                    and not any("missing step" in issue.lower() for issue in state["editor_issues"])
                    and not (state["reference_issues"] + state["scope_issues"]
                             + state["fact_issues"] + state["consistency_issues"])
                    and (not state["structure_issues"]
                         or (len(state["structure_issues"]) == 1
                             and prose_word_count(state["draft"]) < 600)))

        def writer_node(state: BookState) -> dict:
            draft = state["draft"]
            if state["unsupported_sentences"] or state["citation_fixes"]:
                draft = remove_unsupported(draft, state["unsupported_sentences"])
                draft = add_citations(draft, state["citation_fixes"])
                LOG.info("Chapter %d removed %d unsupported sentences and added %d citations", state["chapter_index"] + 1, len(state["unsupported_sentences"]), len(state["citation_fixes"]))
                repairs = state["takeaway_repairs"]
                if not re.search(r"(?m)^Takeaway:\s+", draft):
                    if repairs < 2:
                        draft = draft + "\n\n" + self.takeaway(draft, current_plan(state))
                        return {"draft": draft, "takeaway_repairs": repairs + 1}
                else:
                    return {"draft": draft, "takeaway_repairs": repairs}
            if needs_editor_repair(state) and state["editor_repairs"] < 2:
                LOG.info("Chapter %d focused prose repair %d", state["chapter_index"] + 1, state["editor_repairs"] + 1)
                if any(re.match(r"(?i)^(?:the )?takeaway\b", issue.strip()) for issue in state["editor_issues"]):
                    body = state["draft"].split("Takeaway:", 1)[0].rstrip()
                    draft = body + "\n\n" + self.takeaway(body, current_plan(state))
                    return {"draft": draft, "editor_repairs": state["editor_repairs"] + 1}
                try:
                    draft = self.repair_prose(state["draft"], state["editor_issues"])
                except RuntimeError as exc:
                    LOG.warning("Focused prose repair was rejected: %s", exc)
                    if state["draft_round"] >= 4:
                        raise RuntimeError("Focused prose repair failed after four writer attempts") from exc
                else:
                    return {"draft": draft, "editor_repairs": state["editor_repairs"] + 1}
            if needs_expansion(state) and state["expansion_round"] < 3:
                LOG.info("Chapter %d focused expansion %d", state["chapter_index"] + 1, state["expansion_round"] + 1)
                draft = self.expand(state["draft"], current_plan(state), state["sources"], state["human_feedback"])
                return {"draft": draft, "expansion_round": state["expansion_round"] + 1}
            round_number = state["draft_round"] + 1
            if round_number > 4:
                raise RuntimeError("Chapter failed quality checks after four writer attempts")
            LOG.info("Chapter %d writer attempt %d", state["chapter_index"] + 1, round_number)
            draft = self.draft(
                current_plan(state), state["sources"], state["chapters"],
                draft, state["reference_issues"] + state["structure_issues"]
                + state["editor_issues"] + state["scope_issues"] + state["consistency_issues"]
                + state["fact_issues"] + state["human_feedback"],
            )
            return {"draft": draft, "draft_round": round_number,
                    "takeaway_repairs": 0, "expansion_round": 0, "editor_repairs": 0}

        def structure_node(state: BookState) -> dict:
            issues = format_issues(state["draft"])
            LOG.info("Chapter %d structure issues: %s", state["chapter_index"] + 1, issues)
            return {"structure_issues": issues, "review_round": state["review_round"] + 1}

        def references_node(state: BookState) -> dict:
            issues = reference_issues(state["draft"], state["sources"], self.profile["research"]["domains"])
            LOG.info("Chapter %d reference issues: %d", state["chapter_index"] + 1, len(issues))
            return {"reference_issues": issues}

        def editor_node(state: BookState) -> dict:
            issues = self.edit(state["draft"], state["brief"], state["chapters"])
            LOG.info("Chapter %d editor issues: %s", state["chapter_index"] + 1, issues)
            return {"editor_issues": issues}

        def scope_node(state: BookState) -> dict:
            issues = self.check_scope(state["draft"], current_plan(state), state["brief"])
            LOG.info("Chapter %d scope issues: %s", state["chapter_index"] + 1, issues)
            return {"scope_issues": issues}

        def fact_node(state: BookState) -> dict:
            issues, unsupported, fixes = self.fact_check(state["draft"], state["sources"])
            LOG.info("Chapter %d fact issues: %d", state["chapter_index"] + 1, len(issues))
            return {"fact_issues": issues, "unsupported_sentences": unsupported, "citation_fixes": fixes}

        def consistency_node(state: BookState) -> dict:
            issues = self.check_consistency(state["draft"], state["chapters"])
            LOG.info("Chapter %d consistency issues: %d", state["chapter_index"] + 1, len(issues))
            return {"consistency_issues": issues}

        def after_reviews(state: BookState) -> str:
            issues = (state["structure_issues"] + state["reference_issues"] + state["editor_issues"]
                      + state["scope_issues"] + state["fact_issues"] + state["consistency_issues"])
            if not issues:
                return "review_pause" if human_review else "accept"
            if state["review_round"] >= 40:
                raise RuntimeError(f"Chapter {state['chapter_index'] + 1} failed quality checks after 40 review cycles: {issues[:5]}")
            if state["unsupported_sentences"] or state["citation_fixes"]:
                return "writer"
            if needs_editor_repair(state) and state["editor_repairs"] < 2:
                return "writer"
            if needs_expansion(state) and state["expansion_round"] < 3:
                return "writer"
            if state["draft_round"] >= 4:
                raise RuntimeError(f"Chapter {state['chapter_index'] + 1} failed quality checks after four writer attempts: {issues[:5]}")
            return "writer"

        def human_review_node(state: BookState) -> dict:
            preview = render_chapter(
                state["chapter_index"] + 1, current_plan(state), state["draft"], state["sources"]
            )
            decision = interrupt({"kind": "review", "chapter_index": state["chapter_index"], "chapter": preview})
            if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
                raise ValueError("Chapter review needs an approval decision")
            if decision["approved"]:
                return {"human_approved": True, "human_feedback": []}
            feedback = decision.get("feedback")
            if not isinstance(feedback, str) or not 1 <= len(feedback.strip()) <= 2000:
                raise ValueError("A rejected chapter needs feedback of at most 2000 characters")
            if state["human_review_round"] >= 3:
                raise RuntimeError("Chapter failed after three human revision requests")
            return {
                "human_approved": False, "human_feedback": state["human_feedback"] + [feedback.strip()],
                "human_review_round": state["human_review_round"] + 1,
                "draft_round": 0, "review_round": 0,
            }

        def after_human_review(state: BookState) -> str:
            return "accept" if state["human_approved"] else "writer"

        def accept_node(state: BookState) -> dict:
            number = state["chapter_index"] + 1
            chapter = render_chapter(number, current_plan(state), state["draft"], state["sources"])
            LOG.info("Accepted chapter %d", number)
            return {
                "chapters": state["chapters"] + [chapter],
                "chapter_index": number,
                "sources": [], "research_round": 0, "search_queries": [], "coverage_ok": False,
                "draft": "", "draft_round": 0, "review_round": 0, "structure_issues": [], "reference_issues": [],
                "editor_issues": [], "scope_issues": [], "fact_issues": [],
                "unsupported_sentences": [], "citation_fixes": [], "takeaway_repairs": 0,
                "expansion_round": 0, "editor_repairs": 0,
                "consistency_issues": [], "human_feedback": [],
                "human_review_round": 0, "human_approved": False,
            }

        graph = StateGraph(BookState)
        def add_node(name: str, function: Callable[[BookState], dict]) -> None:
            def guarded(state: BookState) -> dict:
                self.brief = state["brief"]
                self.profile = state.get("profile") or self.profile_for(self.brief)
                if pause_requested is not None and pause_requested(state["job_id"]):
                    interrupt({"kind": "pause", "chapter_index": state["chapter_index"]})
                result = function(state)
                return result if state.get("profile") else {"profile": self.profile, **result}
            graph.add_node(name, guarded)

        add_node("planner", plan_node)
        add_node("researcher", research_node)
        add_node("research_evaluator", research_review_node)
        add_node("writer", writer_node)
        add_node("structure_validator", structure_node)
        add_node("reference_validator", references_node)
        add_node("editor", editor_node)
        add_node("scope_validator", scope_node)
        add_node("fact_checker", fact_node)
        add_node("consistency_validator", consistency_node)
        if human_review:
            # Keep pause and approval interrupts in separate tasks: resume is positional.
            add_node("review_pause", lambda state: {})
            graph.add_node("human_review", human_review_node)
            graph.add_edge("review_pause", "human_review")
        add_node("accept", accept_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "research_evaluator")
        graph.add_conditional_edges("research_evaluator", after_research, ["writer", "researcher"])
        graph.add_edge("writer", "structure_validator")
        graph.add_edge("structure_validator", "reference_validator")
        graph.add_edge("reference_validator", "editor")
        graph.add_edge("editor", "scope_validator")
        graph.add_edge("scope_validator", "fact_checker")
        graph.add_edge("fact_checker", "consistency_validator")
        graph.add_conditional_edges("consistency_validator", after_reviews, ["writer", "review_pause" if human_review else "accept"])
        if human_review:
            graph.add_conditional_edges("human_review", after_human_review, ["writer", "accept"])
        graph.add_conditional_edges("accept", lambda state: "researcher" if state["chapter_index"] < 3 else END, ["researcher", END])
        return graph.compile(checkpointer=checkpointer)

    def run(self, brief: str) -> str:
        final = self.build_graph().invoke(initial_state(brief, profile=self.profile_for(brief)), config={"recursion_limit": 1000})
        return render_book(final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--brief-file", type=Path, help="Plain-text brief; defaults to the included UPI book brief")
    parser.add_argument("--prompts-file", type=Path, help="JSON overrides for agent prompts and research profiles")
    parser.add_argument("--output", type=Path, default=Path("book.md"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; use --overwrite to replace it")
    config = load_config(args.env_file)
    if args.prompts_file:
        config["BOOK_PROMPTS_FILE"] = str(args.prompts_file)
    writer = BookWriter(config)
    brief = args.brief_file.read_text(encoding="utf-8") if args.brief_file else writer.profiles["default_brief"]
    if not 1 <= len(brief.strip()) <= 20000:
        parser.error("Book brief must contain 1–20000 characters")
    LOG.info("Using %s with Tavily research", MODEL)
    book = writer.run(brief)
    args.output.write_text(book, encoding="utf-8")
    LOG.info("Wrote %s", args.output)

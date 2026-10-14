#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# ---------------------------------------------------------------------------
# Student solution: two-stage verifier.
#
# 1. a tool-calling agent resolves *who* the CV belongs to (LinkedIn is the
#    primary source, Facebook is used to cross-check the identity);
# 2. the full profiles are fetched again in code, so the judge always sees the
#    untouched JSON instead of a model-rewritten summary;
# 3. a tool-free judge model compares every discrepancy-capable field against
#    the profile and returns a verdict with a score;
# 4. the score is forced to agree with the verdict (a listed discrepancy can
#    never produce a passing score).
# ---------------------------------------------------------------------------

import os
import sys
import time

CONCURRENCY = 3            # the MCP server is shared by the whole class
MAX_CANDIDATES = 3         # same-name profiles handed to the judge
PER_CV_TIMEOUT = 240.0     # seconds allowed for a single CV
RESOLVER_TIMEOUT = 150.0   # seconds allowed per entity-resolution attempt
RESOLVER_ATTEMPTS = 2      # retry a resolver run that dies on a network blip
JUDGE_TIMEOUT = 150.0      # seconds allowed for the comparison call
RUN_BUDGET = 20 * 60.0     # stop early instead of risking the 30 minute limit
FALLBACK_SCORE = 0.5       # used when a CV cannot be verified at all
DEBUG = bool(os.environ.get("HW2_DEBUG"))

RESOLVER_SYSTEM = """You are the entity-resolution stage of a CV verification
system. You query a mock LinkedIn/Facebook database with your tools and find
which profile belongs to the person who wrote the CV you are given.

Facts you must respect:
- Hundreds of people share the same name. The search tools return up to 20 name
  matches from all over the world, in no particular order, so a name match
  alone proves nothing.
- The CV is truthful except for at most one field, which may be false: city,
  employer, job title, seniority, employment years, degree, school, field of
  study, graduation year, or a skill. Identify the person with the *majority*
  of the evidence (employers, schools, years, city, skills), never with a
  single field.
- The location argument of a search is a filter for narrowing, but a CVs city
  may itself be the false field, so never conclude "not found" because a
  location filter returned nothing.

Method:
1. search_linkedin_people with the CV's full name; repeat with a location or
   industry filter when that helps narrow the candidates.
2. get_linkedin_profile for the candidates whose location, headline or industry
   could fit the CV, and compare their companies, job titles, start/end years,
   schools, degrees, fields, graduation years and skills with the CV.
3. Optionally cross-check the identity on Facebook (search_facebook_users, then
   get_facebook_profile); Facebook agrees with LinkedIn for every candidate.
4. Report up to three ids, best match first. Never report an id that no tool
   returned, and never invent one.

Answer with a single JSON object and nothing else:
{"linkedin_id": <int or null>, "alternative_ids": [<int>, ...],
 "facebook_id": <int or null>, "cv_name": "<name on the CV>",
 "cv_city": "<city on the CV or null>", "identity_reason": "<one sentence>"}"""

RESOLVER_USER = """CV to resolve:

<<<CV>>>

Find this person's profile and answer with the JSON object only."""

JUDGE_SYSTEM = """You are a strict but fair CV auditor. You get one CV and the
LinkedIn profiles of the candidate(s) it may describe, and you decide whether
the CV contains a false claim.

ONLY these fields can produce a discrepancy:
name, city, jobs (company, job title, seniority, start year, end year),
education (degree, school, field of study, graduation year), skills.

NEVER a discrepancy, ignore them completely: job description bullets, the CV
headline, hometown, LinkedIn summary/status/headline/industry text,
years_experience, proficiency numbers, interests, bio, posts, friend counts.

NOT discrepancies (accept them):
- Wording and abbreviation differences: "Bachelor of Science" vs "BSc",
  "Master of Business Administration" vs "MBA", "UI/UX Design" vs "UI/UX",
  "Local State University" vs "LSU", case, punctuation, spacing.
- A CV job title carrying an extra seniority word that matches the profile's
  seniority field (profile title "Engineer" + seniority "senior" vs CV
  "Senior Engineer" is fine).
- Listing fewer skills than the profile, or omitting jobs, education entries or
  skills that the profile contains.
- Extra wording detail that the profile simply does not mention.

DISCREPANCIES (these are false claims):
- A skill listed on the CV that is absent from the profile's skills.
- A company, school, degree or field that contradicts the profile (fake or
  upgraded degree, fake employer, different school).
- A job title inflated beyond what the profile supports, e.g. profile title
  "Analyst" with seniority "junior" while the CV says "Senior Analyst".
- A start year, end year or graduation year that differs from the profile.
- A city that differs from the profile city.

Identity first: choose which profile is the CV owner using the majority of
employers, schools, years and city. One contradicting year does not make a
profile the wrong person when everything else agrees. If no provided profile
can plausibly be the owner, set owner_id to null.

The score must agree with your own findings:
- fully consistent -> 0.9 to 1.0
- consistent, only trivial naming doubts -> 0.7 to 0.9
- at least one discrepancy -> 0.0 to 0.3 (0.0 fabricated employer/school/
  degree, 0.1 shifted years, 0.1-0.2 inflated title or a skill the candidate
  does not have)
- no plausible owner among the profiles -> 0.5

Answer with a single JSON object and nothing else:
{"owner_id": <int or null>, "consistent": <true or false>, "score": <float>,
 "discrepancies": [{"field": "...", "cv": "...", "profile": "..."}],
 "fields_checked": ["name", "city", "jobs", "education", "skills"],
 "notes": "<one or two sentences>"}"""

JUDGE_USER = """CV under audit:

<<<CV>>>

LinkedIn profile(s) that the CV may describe (JSON, untouched from the server):

<<<PROFILES>>>

Audit the CV against the owner profile and answer with the JSON object only."""


def _debug(*parts: Any) -> None:
    if DEBUG:
        print("[hw2]", *parts, file=sys.stderr, flush=True)


def _first_json(text: str) -> Any:
    """Return the first complete JSON value inside *text*, else None."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        return value
    return None


def _tool_payload(result: Any) -> Any:
    """MCP tools answer with [{"type": "text", "text": "<json>"}]; decode it."""
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return _first_json(result["text"]) if result["text"].strip()[:1] in "{[" else result["text"]
        return result
    if isinstance(result, (list, tuple)):
        chunks: list[str] = []
        for block in result:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
            else:
                chunks.append(json.dumps(block, ensure_ascii=False, default=str))
        joined = "\n".join(chunks).strip()
        return _first_json(joined) if joined[:1] in "{[" else joined
    if isinstance(result, str):
        text = result.strip()
        return _first_json(text) if text[:1] in "{[" else text
    return result


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _tool_arg_names(tool: Any) -> list[str]:
    schema = getattr(tool, "args", None)
    if isinstance(schema, dict):
        return [str(key) for key in schema]
    if isinstance(schema, list):
        return [str(key) for key in schema]
    args_schema = getattr(tool, "args_schema", None)
    fields = getattr(args_schema, "model_fields", None)
    if isinstance(fields, dict):
        return [str(key) for key in fields]
    return []


def _cv_name(cv_text: str) -> str:
    """Read the candidate name off the top of a converted CV.

    Most CVs convert to ``Name`` on the first line, but a few PDFs convert to a
    markdown table instead (``| Minh | Pham |     |``), so both shapes are
    handled. Used as a fallback when the resolver agent does not answer.
    """
    for raw_line in (cv_text or "").splitlines()[:12]:
        line = raw_line.strip()
        if not line or set(line) <= set("-|: "):
            continue
        if line.startswith("|"):
            cells = [
                re.sub(r"[*_`]", "", cell).strip()
                for cell in line.strip("|").split("|")
            ]
            cells = [cell for cell in cells if cell and cell.lower() not in {"name", "cv", "resume"}]
            line = " ".join(cells[:3])
        else:
            line = re.sub(r"^[#*\s]+", "", line)
            line = re.sub(r"[*_`]", "", line).strip()
        line = re.sub(r"\s{2,}", " ", line.split("|")[0]).strip()
        words = line.split()
        if not 1 <= len(words) <= 4:
            continue
        if len(line) > 60 or any(char.isdigit() for char in line) or "@" in line:
            continue
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z'’.\-]*", word) for word in words):
            continue
        if line.lower() in {"curriculum vitae", "resume", "cv"}:
            continue
        return line
    return ""


async def _call_tool(tool: Any, args: dict[str, Any], attempts: int = 3) -> Any:
    """Call an MCP tool with retries: ngrok drops a connection now and then."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await tool.ainvoke(args)
        except Exception as exc:  # noqa: BLE001
            last = exc
            _debug(
                "tool",
                getattr(tool, "name", "?"),
                "failed:",
                type(exc).__name__,
                str(exc)[:120],
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(1.0 * (attempt + 1))
    raise last if last is not None else RuntimeError("tool call failed")


class _Verifier:
    """Resolves the candidate behind a CV, then audits the CV against them."""

    def __init__(self, resolve_agent: Any, judge_model: Any, tools: list[Any]) -> None:
        self.resolve_agent = resolve_agent
        self.judge_model = judge_model
        self.linkedin_profile = self._find_tool(tools, "linkedin", "profile")
        self.linkedin_search = self._find_tool(tools, "linkedin", "search")

    @staticmethod
    def _find_tool(tools: list[Any], *keywords: str) -> Any:
        for tool in tools:
            name = str(getattr(tool, "name", "")).lower()
            if all(keyword in name for keyword in keywords):
                return tool
        return None

    async def verify_one(self, cv_text: str) -> float:
        cv_text = (cv_text or "").strip()
        if not cv_text:
            return FALLBACK_SCORE
        resolution = await self._resolve(cv_text)
        profiles = await self._fetch_profiles(resolution["linkedin_ids"])
        if not profiles:
            _debug("no LinkedIn profile for", resolution.get("cv_name"), "->", FALLBACK_SCORE)
            return FALLBACK_SCORE
        verdict = await self._compare(cv_text, profiles)
        return self._score_from(verdict)

    async def _resolve(self, cv_text: str) -> dict[str, Any]:
        """Ask the tool-calling agent which LinkedIn ids belong to this CV.

        The agent is retried once: a dropped MCP connection inside the agent
        used to abort the whole CV.
        """
        data: dict[str, Any] = {}
        request = {"messages": [{"role": "user", "content": RESOLVER_USER.replace("<<<CV>>>", cv_text)}]}
        for attempt in range(RESOLVER_ATTEMPTS):
            try:
                state = await asyncio.wait_for(
                    self.resolve_agent.ainvoke(request, config={"recursion_limit": 40}),
                    timeout=RESOLVER_TIMEOUT,
                )
            except Exception as exc:  # noqa: BLE001 - never let one CV kill the run
                _debug("resolver attempt", attempt + 1, "failed:", type(exc).__name__, str(exc)[:160])
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            parsed = _first_json(_message_text(state["messages"][-1]))
            if isinstance(parsed, dict) and parsed:
                data = parsed
                break
            _debug("resolver attempt", attempt + 1, "returned no usable JSON")

        ids: list[int] = []
        for value in [data.get("linkedin_id"), *(data.get("alternative_ids") or [])]:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                ids.append(value)
            elif isinstance(value, str) and value.strip().isdigit():
                ids.append(int(value.strip()))
        unique_ids = list(dict.fromkeys(ids))
        if not unique_ids:
            unique_ids = await self._fallback_ids(cv_text)
            _debug("resolver gave no id, fallback:", unique_ids)
        return {
            "linkedin_ids": unique_ids[:MAX_CANDIDATES],
            "facebook_id": data.get("facebook_id"),
            "cv_name": data.get("cv_name") or _cv_name(cv_text),
        }

    async def _fallback_ids(self, cv_text: str) -> list[int]:
        """Plain name search, used when the agent does not answer properly."""
        name = _cv_name(cv_text)
        if not name or self.linkedin_search is None:
            return []
        results = await self._search(self.linkedin_search, name)
        return [item["id"] for item in results if isinstance(item.get("id"), int)][:MAX_CANDIDATES]

    async def _search(self, tool: Any, query: str) -> list[dict[str, Any]]:
        args: dict[str, Any] = {}
        arg_names = _tool_arg_names(tool)
        if not arg_names or "q" in arg_names:
            args["q"] = query
        else:
            args[arg_names[0]] = query
        if not arg_names or "limit" in arg_names:
            args["limit"] = 20
        try:
            payload = _tool_payload(await _call_tool(tool, args))
        except Exception as exc:  # noqa: BLE001
            _debug("search failed:", type(exc).__name__, exc)
            return []
        if isinstance(payload, dict):
            payload = payload.get("results") or payload.get("candidates") or payload.get("data")
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    async def _fetch_profiles(self, person_ids: list[int]) -> list[dict[str, Any]]:
        """Fetch the full profile JSON in code, so nothing is paraphrased away."""
        if self.linkedin_profile is None:
            return []
        arg_names = _tool_arg_names(self.linkedin_profile)
        arg_name = "person_id" if (not arg_names or "person_id" in arg_names) else arg_names[0]
        profiles: list[dict[str, Any]] = []
        for person_id in person_ids:
            try:
                payload = _tool_payload(await _call_tool(self.linkedin_profile, {arg_name: person_id}))
            except Exception as exc:  # noqa: BLE001
                _debug("profile fetch failed for", person_id, type(exc).__name__)
                continue
            if isinstance(payload, dict) and not payload.get("error") and payload.get("id"):
                profiles.append(payload)
        return profiles

    async def _compare(self, cv_text: str, profiles: list[dict[str, Any]]) -> dict[str, Any]:
        """Tool-free comparison: full rulebook in the prompt, JSON verdict out."""
        user = (
            JUDGE_USER.replace("<<<CV>>>", cv_text[:9000])
            .replace("<<<PROFILES>>>", json.dumps(profiles, ensure_ascii=False, indent=1)[:24000])
        )
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(self.judge_model.ainvoke(messages), timeout=JUDGE_TIMEOUT)
                parsed = _first_json(_message_text(response))
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:  # noqa: BLE001
                _debug("judge failed:", type(exc).__name__, exc)
            if attempt == 0:
                messages.append({"role": "user", "content": "Answer with the JSON object only."})
        return {}

    @staticmethod
    def _score_from(verdict: dict[str, Any]) -> float:
        """Clamp the model score and force it to agree with the verdict."""
        if not isinstance(verdict, dict):
            return FALLBACK_SCORE
        try:
            score = float(verdict.get("score"))
        except (TypeError, ValueError):
            return FALLBACK_SCORE
        if math.isnan(score) or math.isinf(score):
            return FALLBACK_SCORE
        discrepancies = verdict.get("discrepancies") or []
        owner = verdict.get("owner_id")
        if discrepancies:
            score = min(score, 0.35)          # a false claim can never pass
        elif verdict.get("consistent") is True and owner not in (None, "null", ""):
            score = max(score, 0.65)          # verified consistent stays above 0.5
        elif verdict.get("consistent") is False:
            return FALLBACK_SCORE             # contradictory answer without details
        return max(0.0, min(1.0, score))


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
    from langchain_deepseek import ChatDeepSeek

    # The MCP server sits behind an ngrok tunnel, so tool calls do occasionally
    # fail with a dropped connection. Retry them with backoff, and if they still
    # fail, hand the error back to the model instead of crashing the whole run.
    resolve_agent = create_agent(
        model=ChatDeepSeek(model=MODEL_NAME, temperature=0),
        tools=list(tools),
        system_prompt=RESOLVER_SYSTEM,
        middleware=[
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=15.0,
                on_failure="continue",
            ),
            ModelRetryMiddleware(max_retries=2, initial_delay=1.0, max_delay=20.0),
        ],
    )
    judge_model = ChatDeepSeek(model=MODEL_NAME, temperature=0)
    return _Verifier(resolve_agent, judge_model, list(tools))


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    verifier = agent if isinstance(agent, _Verifier) else _Verifier(agent, agent, [])
    semaphore = asyncio.Semaphore(CONCURRENCY)
    deadline = time.monotonic() + RUN_BUDGET
    total = len(cvs)

    async def audit(index: int, name: str, text: str) -> tuple[str, float]:
        async with semaphore:
            budget = deadline - time.monotonic()
            if budget <= 5.0:
                _debug("out of time budget, skipping", name)
                return name, FALLBACK_SCORE
            started = time.monotonic()
            try:
                score = await asyncio.wait_for(
                    verifier.verify_one(text),
                    timeout=min(PER_CV_TIMEOUT, budget),
                )
            except Exception as exc:  # noqa: BLE001 - one bad CV must not crash the run
                _debug("CV", name, "failed:", type(exc).__name__, exc)
                score = FALLBACK_SCORE
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                score = FALLBACK_SCORE
            score = max(0.0, min(1.0, float(score)))
            print(
                f"[{index}/{total}] {name}: {score:.4f} "
                f"({time.monotonic() - started:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            return name, score

    results = await asyncio.gather(
        *(audit(index, name, text) for index, (name, text) in enumerate(cvs.items(), start=1))
    )
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())

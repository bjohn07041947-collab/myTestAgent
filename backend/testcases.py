"""Loads end-to-end test cases from markdown files in the testcases/ directory.

A test case file looks like:

    # Application flow

    Optional free-text description.

    ## Steps

    1. Click the radio button 
    2. Enter "20000.00" in the amount field
    3. Wait 2 seconds
    4. Verify result is displayed

Steps must be numbered list items. Supported step verbs:
- Click / Select / Choose / Press / Tap / Open <element description>
- Type / Enter / Input "<text>" [in/into <element description>]
- Verify / Check / Confirm / Expect <expectation>
- Wait <n> seconds
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

STEP_PATTERN = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")
QUOTED_PATTERN = re.compile(r'"([^"]*)"')

CLICK_VERBS = ("click", "select", "choose", "press", "tap", "open")
TYPE_VERBS = ("type", "enter", "input", "fill")
VERIFY_VERBS = ("verify", "check", "confirm", "expect", "assert")
WAIT_VERBS = ("wait", "pause")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass
class TestStep:
    number: int
    text: str             # raw step text from the markdown file
    action: str           # click | type | verify | wait
    target: str = ""      # element description (click/type) or expectation (verify)
    input_text: str = ""  # text to type, for type steps
    wait_seconds: float = 2.0


@dataclass
class TestCase:
    name: str
    slug: str
    path: Path
    description: str = ""
    steps: List[TestStep] = field(default_factory=list)


def parse_step(number: int, text: str) -> TestStep:
    words = text.split(None, 1)
    first_word = words[0].lower().strip(":,") if words else ""
    remainder = words[1].strip() if len(words) > 1 else ""

    if first_word in TYPE_VERBS:
        quoted = QUOTED_PATTERN.search(text)
        if quoted:
            input_text = quoted.group(1)
            rest = text[quoted.end():]
        else:
            input_text = remainder
            rest = ""
        target_match = re.search(r"\b(?:into|in|on)\b\s+(.*)$", rest, re.IGNORECASE)
        target = target_match.group(1).strip() if target_match else ""
        return TestStep(number, text, "type", target=target, input_text=input_text)

    if first_word in WAIT_VERBS:
        seconds_match = re.search(r"(\d+(?:\.\d+)?)", remainder)
        seconds = float(seconds_match.group(1)) if seconds_match else 2.0
        return TestStep(number, text, "wait", wait_seconds=seconds)

    if first_word in VERIFY_VERBS:
        return TestStep(number, text, "verify", target=remainder or text)

    # Default action is click; strip the leading verb when it is a known one.
    target = remainder if first_word in CLICK_VERBS and remainder else text
    return TestStep(number, text, "click", target=target)


class TestCaseManager:
    """Loads and indexes test case markdown files."""

    def __init__(self, testcase_dir=None):
        if testcase_dir is None:
            self.testcase_dir = Path(__file__).parent / "testcases"
        else:
            self.testcase_dir = Path(testcase_dir)
        self.test_cases: Dict[str, TestCase] = {}
        self.reload()

    def reload(self) -> None:
        cases: Dict[str, TestCase] = {}
        if self.testcase_dir.exists():
            for path in sorted(self.testcase_dir.glob("*.md")):
                case = self._parse_file(path)
                if case.steps:
                    cases[case.slug] = case
        self.test_cases = cases

    def _parse_file(self, path: Path) -> TestCase:
        name = path.stem.replace("-", " ").replace("_", " ").title()
        description_lines: List[str] = []
        steps: List[TestStep] = []

        for line in path.read_text().splitlines():
            heading = re.match(r"^#\s+(.*\S)\s*$", line)
            if heading and not steps:
                name = heading.group(1)
                continue
            step_match = STEP_PATTERN.match(line)
            if step_match:
                steps.append(parse_step(len(steps) + 1, step_match.group(1)))
            elif not steps and line.strip() and not line.startswith("#"):
                description_lines.append(line.strip())

        return TestCase(
            name=name,
            slug=slugify(name),
            path=path,
            description=" ".join(description_lines),
            steps=steps,
        )

    def list_names(self) -> List[str]:
        return [case.name for case in self.test_cases.values()]

    def find(self, name: str) -> Optional[TestCase]:
        """Find a test case by name, tolerating loose phrasing from voice input."""
        key = slugify(name)
        if key in self.test_cases:
            return self.test_cases[key]
        for case in self.test_cases.values():
            if key in case.slug or case.slug in key:
                return case
        # Match on overlapping words as a last resort ("the new car test").
        key_words = set(key.split("-"))
        best, best_overlap = None, 0
        for case in self.test_cases.values():
            overlap = len(key_words & set(case.slug.split("-")))
            if overlap > best_overlap:
                best, best_overlap = case, overlap
        return best if best_overlap >= 2 else None

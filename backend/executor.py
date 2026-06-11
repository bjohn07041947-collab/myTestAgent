"""Executes parsed test cases against the live screen using the visual locator.

Each click/type step captures a fresh screenshot, locates the target with the
two-pass grid locator, and performs the action with pyautogui. Results are
written to results/<slug>-<timestamp>.md, with a screenshot saved for any
failed step.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pyautogui

import grid
from locator import ElementLocator
from testcases import TestCase, TestStep

logger = logging.getLogger("test-executor")

# Seconds to let the UI settle after a click or typing.
ACTION_SETTLE_SECONDS = 1.5


@dataclass
class StepResult:
    step: TestStep
    status: str  # passed | failed | error | skipped
    detail: str = ""


@dataclass
class TestResult:
    test_case: TestCase
    started_at: datetime
    step_results: List[StepResult] = field(default_factory=list)
    report_path: Optional[Path] = None

    @property
    def passed(self) -> bool:
        return bool(self.step_results) and all(
            r.status == "passed" for r in self.step_results
        )

    def summary(self) -> str:
        passed_count = sum(1 for r in self.step_results if r.status == "passed")
        lines = [
            f"Test case {self.test_case.name}: {'PASSED' if self.passed else 'FAILED'} "
            f"({passed_count} of {len(self.step_results)} steps passed)"
        ]
        for r in self.step_results:
            if r.status not in ("passed", "skipped"):
                lines.append(f"Step {r.step.number} {r.status}: {r.step.text}. {r.detail}")
        if self.report_path:
            lines.append(f"Full report: {self.report_path.name}")
        return "\n".join(lines)


class TestExecutor:
    def __init__(self, locator: ElementLocator, results_dir=None):
        self.locator = locator
        self.results_dir = (
            Path(results_dir) if results_dir else Path(__file__).parent / "results"
        )
        self._run_prefix = ""

    async def run(self, test_case: TestCase) -> TestResult:
        result = TestResult(test_case=test_case, started_at=datetime.now())
        self._run_prefix = (
            f"{test_case.slug}-{result.started_at.strftime('%Y%m%d-%H%M%S')}"
        )
        logger.info(f"Running test case: {test_case.name} ({len(test_case.steps)} steps)")

        aborted = False
        for step in test_case.steps:
            if aborted:
                result.step_results.append(
                    StepResult(step, "skipped", "earlier step failed")
                )
                continue
            step_result = await self._execute_step(step)
            result.step_results.append(step_result)
            logger.info(f"Step {step.number} {step_result.status}: {step.text}")
            # A failed click/type leaves the app in the wrong state for the
            # remaining steps; a failed verify is just a failed assertion.
            if step_result.status != "passed" and step.action in ("click", "type"):
                aborted = True

        result.report_path = self._write_report(result)
        return result

    async def click_described(self, description: str) -> Tuple[bool, str]:
        """Locate an element by description and click it."""
        print (f"click_described -- desc:{description}")
        target = await self.locator.locate(description)
        if target is None:
            return False, f"could not locate '{description}' on the screen"
        x, y = target
        await asyncio.to_thread(self._do_click, x, y)
        await asyncio.sleep(ACTION_SETTLE_SECONDS)
        return True, f"clicked at ({int(x)}, {int(y)})"

    async def type_text(self, text: str) -> None:
        await asyncio.to_thread(pyautogui.write, text, 0.05)
        await asyncio.sleep(ACTION_SETTLE_SECONDS)

    async def _execute_step(self, step: TestStep) -> StepResult:
        try:
            if step.action == "wait":
                await asyncio.sleep(step.wait_seconds)
                return StepResult(step, "passed")

            if step.action == "click":
                ok, detail = await self.click_described(step.target)
                if not ok:
                    await self._save_failure_screenshot(step)
                return StepResult(step, "passed" if ok else "failed", detail)

            if step.action == "type":
                if step.target:
                    ok, detail = await self.click_described(step.target)
                    if not ok:
                        await self._save_failure_screenshot(step)
                        return StepResult(step, "failed", detail)
                await self.type_text(step.input_text)
                return StepResult(step, "passed")

            if step.action == "verify":
                ok, reason = await self.locator.verify(step.target)
                if not ok:
                    await self._save_failure_screenshot(step)
                return StepResult(step, "passed" if ok else "failed", reason)

            return StepResult(step, "error", f"unknown action: {step.action}")
        except Exception as e:
            logger.exception(f"Step {step.number} raised an error")
            return StepResult(step, "error", str(e))

    def _do_click(self, x: float, y: float) -> None:
        pyautogui.moveTo(x, y, duration=0.3)
        pyautogui.click()

    async def _save_failure_screenshot(self, step: TestStep) -> None:
        try:
            screenshot, _ = await asyncio.to_thread(grid.capture_screenshot)
            self.results_dir.mkdir(exist_ok=True)
            path = self.results_dir / f"{self._run_prefix}-step{step.number}.png"
            await asyncio.to_thread(screenshot.save, path, "PNG")
            logger.info(f"Saved failure screenshot: {path}")
        except Exception:
            logger.exception("Failed to save failure screenshot")

    def _write_report(self, result: TestResult) -> Path:
        self.results_dir.mkdir(exist_ok=True)
        path = self.results_dir / f"{self._run_prefix}.md"

        def escape(text: str) -> str:
            return text.replace("|", "\\|")

        lines = [
            f"# {result.test_case.name}",
            "",
            f"- Started: {result.started_at.isoformat(timespec='seconds')}",
            f"- Result: {'PASSED' if result.passed else 'FAILED'}",
            f"- Source: {result.test_case.path.name}",
            "",
            "| # | Step | Status | Detail |",
            "|---|------|--------|--------|",
        ]
        for r in result.step_results:
            lines.append(
                f"| {r.step.number} | {escape(r.step.text)} | {r.status} | {escape(r.detail)} |"
            )
        path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote test report: {path}")
        return path

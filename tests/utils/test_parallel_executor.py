# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from threading import Event
from typing import Any

from kebab.utils.parallel_executor import (
    RESULT_TOTAL_TOKENS_KEY,
    GlobalPause,
    LeakyBucketRateLimiter,
    TokenBucketRateLimiter,
    parallel_execute,
)

# ---------------------------------------------------------------------------
# LeakyBucketRateLimiter tests
# ---------------------------------------------------------------------------

class TestLeakyBucketRateLimiter:
    def test_basic_acquire(self) -> None:
        stop = Event()
        limiter = LeakyBucketRateLimiter(
            max_requests_per_minute=600, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            # Should acquire immediately (bucket starts full).
            t0 = time.monotonic()
            limiter.acquire()
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5
            stop.set()

    def test_rate_limits_requests(self) -> None:
        # 60 RPM => 1 per second.  After exhausting the initial token the
        # second acquire must wait ~1 s.
        stop = Event()
        limiter = LeakyBucketRateLimiter(
            max_requests_per_minute=60, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            limiter.acquire()  # first token — immediate
            t0 = time.monotonic()
            limiter.acquire()  # second token — must wait
            elapsed = time.monotonic() - t0
            assert elapsed >= 0.8, f"Expected ~1s wait, got {elapsed:.2f}s"
            stop.set()

    def test_context_manager(self) -> None:
        stop = Event()
        limiter = LeakyBucketRateLimiter(
            max_requests_per_minute=6000, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            limiter.acquire()
            stop.set()

# ---------------------------------------------------------------------------
# TokenBucketRateLimiter tests
# ---------------------------------------------------------------------------

class TestTokenBucketRateLimiter:
    def test_acquire_succeeds_with_budget(self) -> None:
        stop = Event()
        limiter = TokenBucketRateLimiter(
            max_tokens_per_minute=60_000, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            t0 = time.monotonic()
            assert limiter.acquire() is True
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5
            stop.set()

    def test_blocks_when_budget_exhausted(self) -> None:
        stop = Event()
        limiter = TokenBucketRateLimiter(
            max_tokens_per_minute=1_000, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            # Report enough tokens to exhaust the budget.
            limiter.report_tokens(1_500)
            # Next acquire should block until refill provides budget.
            t0 = time.monotonic()
            assert limiter.acquire() is True
            elapsed = time.monotonic() - t0
            # Budget was -500; refill rate is 1000/60 ≈ 16.7 tokens/s.
            # Need ~30s to refill 500 tokens — but we only need budget > 0,
            # so we just verify it took more than 1s (the refill interval).
            assert elapsed >= 1.0, f"Expected blocking, got {elapsed:.2f}s"
            stop.set()

    def test_report_tokens_decreases_budget(self) -> None:
        stop = Event()
        limiter = TokenBucketRateLimiter(
            max_tokens_per_minute=10_000, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            initial_budget = limiter._budget
            limiter.report_tokens(5_000)
            assert limiter._budget == initial_budget - 5_000
            stop.set()

    def test_shutdown_unblocks_acquire(self) -> None:
        stop = Event()
        shutdown = Event()
        limiter = TokenBucketRateLimiter(
            max_tokens_per_minute=100, time_to_stop=stop, logger=logging.getLogger(__name__)
        )
        with limiter:
            limiter.report_tokens(200)  # exhaust budget
            shutdown.set()
            result = limiter.acquire(shutdown=shutdown)
            assert result is False
            stop.set()

# ---------------------------------------------------------------------------
# GlobalPause tests
# ---------------------------------------------------------------------------

class TestGlobalPause:
    def test_no_pause_initially(self) -> None:
        """A fresh GlobalPause should not block."""
        pause = GlobalPause()
        t0 = time.monotonic()
        pause.wait_if_paused()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1

    def test_activate_blocks_for_delay(self) -> None:
        """After activation the caller should block for approximately the delay."""
        pause = GlobalPause()
        pause.activate(1.5)
        t0 = time.monotonic()
        pause.wait_if_paused()
        elapsed = time.monotonic() - t0
        assert elapsed >= 1.0, f"Expected ~1.5s wait, got {elapsed:.2f}s"
        assert elapsed < 3.0, f"Waited too long: {elapsed:.2f}s"

    def test_activate_extends_not_shortens(self) -> None:
        """A second activation with a longer delay should extend, not shorten, the pause."""
        pause = GlobalPause()
        pause.activate(1.0)
        pause.activate(3.0)
        t0 = time.monotonic()
        pause.wait_if_paused()
        elapsed = time.monotonic() - t0
        assert elapsed >= 2.5, f"Expected ~3s wait, got {elapsed:.2f}s"

    def test_shorter_activate_does_not_shorten(self) -> None:
        """A second activation with a shorter delay should not reduce the existing pause."""
        pause = GlobalPause()
        pause.activate(3.0)
        pause.activate(0.5)
        t0 = time.monotonic()
        pause.wait_if_paused()
        elapsed = time.monotonic() - t0
        assert elapsed >= 2.5, f"Expected ~3s wait (not shortened), got {elapsed:.2f}s"

    def test_shutdown_unblocks(self) -> None:
        """Setting shutdown should unblock wait_if_paused quickly."""
        pause = GlobalPause()
        pause.activate(60.0)  # long pause
        shutdown = Event()
        shutdown.set()
        t0 = time.monotonic()
        pause.wait_if_paused(shutdown=shutdown)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"Shutdown should unblock quickly, got {elapsed:.2f}s"

# ---------------------------------------------------------------------------
# parallel_execute tests
# ---------------------------------------------------------------------------

def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Trivial process function that echoes the payload."""
    return {"value": payload["value"]}

def _failing(_payload: dict[str, Any]) -> dict[str, Any]:
    """Always raises an exception."""
    raise RuntimeError("boom")

def _retry_on_status(result: dict[str, Any]) -> tuple[bool, float]:
    return (result.get("status") == "retry", 0.0)

class TestParallelExecute:
    def test_single_item(self) -> None:
        items = [("0", {"value": 42})]
        results = list(parallel_execute(items=items, process_fn=_identity))
        assert len(results) == 1
        assert results[0] == ("0", {"value": 42})

    def test_multiple_items(self) -> None:
        items = [(str(i), {"value": i}) for i in range(10)]
        results = list(parallel_execute(items=items, process_fn=_identity, num_workers=4))
        assert len(results) == 10
        result_map = dict(results)
        for i in range(10):
            assert result_map[str(i)]["value"] == i

    def test_empty_items(self) -> None:
        results = list(parallel_execute(items=iter([]), process_fn=_identity))
        assert results == []

    def test_exception_marks_failure(self) -> None:
        items = [("0", {"value": 1})]
        results = list(parallel_execute(items=items, process_fn=_failing, max_retries=1))
        assert len(results) == 1
        _id, result = results[0]
        assert result.get("processing_failed") is True

    def test_retries_on_exception(self) -> None:
        call_count = 0

        def _fail_then_succeed(_payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return {"ok": True}

        items = [("0", {})]
        results = list(parallel_execute(items=items, process_fn=_fail_then_succeed, max_retries=5))
        assert len(results) == 1
        assert results[0][1]["ok"] is True
        assert call_count == 3

    def test_should_retry_callback(self) -> None:
        call_count = 0

        def _retry_once(_payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"status": "retry"}
            return {"status": "ok"}

        items = [("0", {})]
        results = list(
            parallel_execute(
                items=items,
                process_fn=_retry_once,
                max_retries=3,
                should_retry=_retry_on_status,
            )
        )
        assert len(results) == 1
        assert results[0][1]["status"] == "ok"
        assert call_count == 2

    def test_multiple_workers_all_results_returned(self) -> None:
        """Verify that num_workers > 1 returns all results correctly."""

        def _slow(payload: dict[str, Any]) -> dict[str, Any]:
            time.sleep(0.1)
            return {"value": payload["value"]}

        items = [(str(i), {"value": i}) for i in range(8)]

        results = list(
            parallel_execute(
                items=items,
                process_fn=_slow,
                num_workers=4,
                max_requests_per_minute=10_000,
            )
        )
        assert len(results) == 8
        result_map = dict(results)
        for i in range(8):
            assert result_map[str(i)]["value"] == i

    def test_token_rate_limiting(self) -> None:
        """Verify that max_tokens_per_minute throttles based on reported tokens."""

        def _with_tokens(payload: dict[str, Any]) -> dict[str, Any]:
            return {"value": payload["value"], RESULT_TOTAL_TOKENS_KEY: 500}

        items = [(str(i), {"value": i}) for i in range(5)]

        results = list(
            parallel_execute(
                items=items,
                process_fn=_with_tokens,
                num_workers=1,
                max_requests_per_minute=10_000,
                max_tokens_per_minute=60_000,
            )
        )
        assert len(results) == 5
        result_map = dict(results)
        for i in range(5):
            assert result_map[str(i)]["value"] == i

    def test_no_token_limit_when_none(self) -> None:
        """Verify parallel_execute works normally when max_tokens_per_minute is None."""
        items = [(str(i), {"value": i}) for i in range(3)]

        results = list(
            parallel_execute(
                items=items,
                process_fn=_identity,
                max_tokens_per_minute=None,
            )
        )
        assert len(results) == 3

    def test_global_pause_on_should_retry(self) -> None:
        """When should_retry triggers, all workers should be paused via GlobalPause.

        We use 2 workers.  Item "0" returns a retryable result with a 2s delay.
        Item "1" is processed by the other worker — it should be delayed by the
        global pause even though it doesn't itself need a retry.
        """
        call_log: list[tuple[str, float]] = []

        call_count = 0

        def _process(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            call_log.append((payload["id"], time.monotonic()))
            if payload["id"] == "0" and call_count <= 1:
                return {"status": "rate_limited", "id": payload["id"]}
            return {"status": "ok", "id": payload["id"]}

        def _should_retry(result: dict[str, Any]) -> tuple[bool, float]:
            if result.get("status") == "rate_limited":
                return (True, 2.0)
            return (False, 0.0)

        items = [("0", {"id": "0"}), ("1", {"id": "1"})]

        t0 = time.monotonic()
        results = list(
            parallel_execute(
                items=items,
                process_fn=_process,
                num_workers=2,
                max_retries=3,
                max_requests_per_minute=10_000,
                should_retry=_should_retry,
            )
        )
        total_elapsed = time.monotonic() - t0

        assert len(results) == 2
        result_map = dict(results)
        assert result_map["0"]["status"] == "ok"
        assert result_map["1"]["status"] == "ok"
        # The run should have taken at least ~2s because of the global pause.
        assert total_elapsed >= 1.5, f"Expected global pause to delay execution, got {total_elapsed:.2f}s"

    def test_shutdown_drains_orphaned_items(self) -> None:
        """Items still in the queue at shutdown are drained so the generator terminates.

        A slow process_fn ensures that only some items complete before the
        caller stops iterating (which triggers generator cleanup / shutdown).
        Without the orphan-drain fix this test would hang indefinitely.
        """

        def _very_slow(payload: dict[str, Any]) -> dict[str, Any]:
            time.sleep(0.5)
            return {"value": payload["value"]}

        items = [(str(i), {"value": i}) for i in range(20)]

        collected: list[tuple[str, dict[str, Any]]] = []
        for rid, result in parallel_execute(
            items=items,
            process_fn=_very_slow,
            num_workers=2,
            max_requests_per_minute=10_000,
        ):
            collected.append((rid, result))
            if len(collected) >= 3:
                break

        # We got at least the 3 we asked for; generator closed cleanly.
        assert len(collected) >= 3


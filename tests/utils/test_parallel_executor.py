# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from threading import Event
from typing import Any

from kebab.utils.parallel_executor import LeakyBucketRateLimiter, parallel_execute


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

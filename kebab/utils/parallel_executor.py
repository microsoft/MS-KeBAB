# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Generic parallel task executor with rate limiting and retry support.

This module provides :func:`parallel_execute`, a reusable producer-consumer
framework for running I/O-bound work items across a thread pool with:

* Configurable concurrency (``num_workers``)
* Leaky-bucket rate limiting (``max_requests_per_minute``)
* Token-bucket rate limiting (``max_tokens_per_minute``)
* Automatic retries with exponential back-off (``max_retries``)
* Streaming results — callers iterate ``(id, result)`` tuples as they become
  available, so memory usage stays bounded.
"""

from __future__ import annotations

import concurrent.futures
import heapq
import logging
import math
import queue
import time
import types
from collections.abc import Callable, Iterable
from threading import BoundedSemaphore, Condition, Event, Lock, Thread
from typing import Any, Self, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class GlobalPause:
    """Shared pause gate triggered by retryable errors (e.g. HTTP 429).

    When any consumer encounters a retryable response the caller activates
    the pause for the ``Retry-After`` duration.  **All** consumers then
    block in :meth:`wait_if_paused` until the pause expires, preventing
    the entire worker pool from continuing to flood a rate-limited endpoint.

    Multiple concurrent activations are safe: the expiry is extended to the
    *latest* requested time (never shortened).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._expiry: float = 0.0  # monotonic timestamp

    def activate(self, delay: float, logger: logging.Logger | None = None) -> None:
        """Pause all consumers for at least *delay* seconds from now."""
        with self._lock:
            new_expiry = time.monotonic() + delay
            if new_expiry > self._expiry:
                self._expiry = new_expiry
                if logger is not None:
                    logger.info(f"Global pause activated for {delay:.1f}s")

    def wait_if_paused(self, shutdown: Event | None = None) -> None:
        """Block the calling thread until the pause expires or *shutdown* fires."""
        while True:
            with self._lock:
                remaining = self._expiry - time.monotonic()
            if remaining <= 0:
                return
            if shutdown is not None and shutdown.is_set():
                return
            time.sleep(min(remaining, 0.5))


# Key used by ``parallel_execute`` to extract the token count from a result
# dict returned by ``process_fn``.  Callers should store the total number of
# tokens consumed by the request under this key so that the token-rate limiter
# can account for it.
RESULT_TOTAL_TOKENS_KEY = "total_tokens"

def _log_token_usage_headers(
    result: dict[str, Any],
    request_id: str,
    logger: logging.Logger,
) -> None:
    """Log token-usage response headers from *result* when a request fails.

    Matches any response header whose lowercased name contains ``"token"``
    or ``"ratelimit"`` so that both OpenAI and Azure OpenAI header
    conventions are covered without hard-coding specific names.
    """
    status_code = result.get("request_status_code", "unknown")
    error_msg = result.get("request_error_message", "")
    headers = result.get("response_headers")
    token_info: dict[str, str] = {}
    if isinstance(headers, dict):
        token_info = {k: v for k, v in headers.items() if "token" in k or "ratelimit" in k}
    logger.warning(
        "Item %s failed (status %s): %s | token/rate-limit headers: %s",
        request_id,
        status_code,
        error_msg,
        token_info if token_info else "none available",
    )


class TokenBucketRateLimiter:
    """A token-bucket rate limiter that throttles based on tokens consumed per minute.

    After each request the consumer reports how many tokens were used via
    :meth:`report_tokens`.  A background thread replenishes the bucket at a
    fixed rate (``max_tokens_per_minute / 60`` tokens per second).
    :meth:`acquire` blocks until enough capacity is available in the bucket.
    """

    _SECONDS_PER_MINUTE = 60

    def __init__(self, max_tokens_per_minute: int, time_to_stop: Event, logger: logging.Logger) -> None:
        self.max_tokens_per_minute = max_tokens_per_minute
        self.time_to_stop = time_to_stop
        self.logger = logger

        # Available token budget (may go negative after a large request).
        self._budget: float = float(max_tokens_per_minute)
        self._lock = Lock()
        self._cv = Condition(self._lock)

        # Refill rate: tokens per second.
        self._refill_rate = max_tokens_per_minute / self._SECONDS_PER_MINUTE
        self._refill_thread = Thread(target=self._refill_loop, daemon=True)

    # -- refill loop ----------------------------------------------------------

    def _refill_loop(self) -> None:
        interval = 1.0  # refill every second
        try:
            while not self.time_to_stop.is_set():
                time.sleep(interval)
                with self._cv:
                    prev = self._budget
                    self._budget = min(self._budget + self._refill_rate * interval, float(self.max_tokens_per_minute))
                    if self._budget > 0 and prev <= 0:
                        self._cv.notify_all()
        except Exception as e:
            self.logger.exception(f"Token rate-limiter refill thread failed: {e}")  # noqa: TRY401

    # -- public API -----------------------------------------------------------

    def report_tokens(self, count: int) -> None:
        """Report *count* tokens consumed by a completed request."""
        with self._cv:
            self._budget -= count
            self.logger.debug(f"Token limiter: consumed {count} tokens, budget now {self._budget:.0f}")

    def acquire(self, shutdown: Event | None = None, timeout: float = 0.5) -> bool:
        """Block until at least 1 token of budget is available.

        Returns ``False`` if *shutdown* is signalled before budget becomes
        available.
        """
        with self._cv:
            while self._budget <= 0:
                if shutdown is not None and shutdown.is_set():
                    return False
                if self.time_to_stop.is_set():
                    return False
                self._cv.wait(timeout=timeout)
        return True

    def __enter__(self) -> Self:
        self._refill_thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.time_to_stop.set()
        if self._refill_thread.is_alive():
            self._refill_thread.join()


class LeakyBucketRateLimiter:
    """A leaky bucket rate limiter that allows for a certain number of requests per minute, with a burst capacity."""

    _SECONDS_PER_MINUTE = 60

    def __init__(self, max_requests_per_minute: int, time_to_stop: Event, logger: logging.Logger) -> None:
        """Initializes the rate limiter."""
        self.max_burst = max(1, math.floor(max_requests_per_minute / self._SECONDS_PER_MINUTE))
        self.semaphore = BoundedSemaphore(value=self.max_burst)
        self.time_to_stop = time_to_stop
        self.logger = logger
        self.max_requests_per_minute = max_requests_per_minute
        self.rate_limiter_thread = Thread(target=self.__rate_limiter, daemon=True)

    def __rate_limiter(self) -> None:
        """The rate limiter thread that releases permits at a fixed rate until signaled to stop."""
        try:
            while not self.time_to_stop.is_set():
                time.sleep(self._SECONDS_PER_MINUTE / self.max_requests_per_minute)
                if self._get_semaphore_value() < self.max_burst:
                    self.semaphore.release()
                    self.logger.debug(f"Released semaphore. {self._get_semaphore_value()} requests remaining.")
                else:
                    self.logger.debug(f"Semaphore is full. {self._get_semaphore_value()} requests remaining.")
        except Exception as e:
            # Must explicitly log the exception for it to appear in logs in Heron.
            self.logger.exception(f"Rate limiter thread failed: {e}")  # noqa: TRY401

    def _get_semaphore_value(self) -> int:
        # There's no public API exposing the value.
        return self.semaphore._value  # noqa: SLF001

    def __enter__(self) -> Self:
        """Enters the context manager, starting the rate limiter thread."""
        self.rate_limiter_thread.start()
        return self

    def acquire(self, shutdown: Event | None = None, timeout: float = 0.5) -> bool:
        """Acquires a permit from the rate limiter, periodically checking shutdown.

        Args:
            shutdown: An optional Event that, if set, causes the method to return False.
            timeout: The polling interval in seconds to check the shutdown event.

        Returns:
            bool: True if a permit was acquired, False if shutdown was signaled.
        """
        while True:
            if shutdown is not None and shutdown.is_set():
                return False
            if self.semaphore.acquire(timeout=timeout):
                return True

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None
    ) -> None:
        """Exits the context manager, signaling the rate limiter thread to stop and waiting for it to finish."""
        self.time_to_stop.set()
        if self.rate_limiter_thread.is_alive():
            self.rate_limiter_thread.join()


def parallel_execute(
    items: Iterable[tuple[str, Any]],
    process_fn: Callable[[Any], dict[str, Any]],
    num_workers: int = 1,
    max_retries: int = 3,
    max_requests_per_minute: int = 60,
    max_tokens_per_minute: int | None = None,
    should_retry: Callable[[dict[str, Any]], tuple[bool, float]] | None = None,
    logger: logging.Logger | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Execute *process_fn* over *items* in parallel with rate limiting and retries.

    This is a generic producer-consumer framework.  Each item is a
    ``(request_id, payload)`` tuple.  *process_fn* receives the payload and
    must return a result dict.  Results are yielded as ``(request_id, result)``
    tuples **in completion order** (not input order).

    Rate limiting is governed by the **stricter** of two limits:

    * ``max_requests_per_minute`` — leaky-bucket permits (1 per request).
    * ``max_tokens_per_minute`` — token-bucket budget.  After each request
      the consumer reports the ``total_tokens`` value from the result dict
      (see :data:`RESULT_TOTAL_TOKENS_KEY`).  If the token budget is
      exhausted, consumers block until it refills.  When *None* (the
      default) token-rate limiting is disabled.

    Args:
        items: An iterable of ``(id, payload)`` tuples to process.
        process_fn: A callable that takes a payload and returns a result dict.
        num_workers: Number of concurrent consumer threads.
        max_retries: Maximum number of retries for failed items.
        max_requests_per_minute: Rate limit for the leaky-bucket limiter.
        max_tokens_per_minute: Optional token-rate limit.  When provided the
            effective throughput is ``min(RPM, TPM / avg_tokens_per_request)``.
        should_retry: An optional callable that inspects a result dict and
            returns ``(retry: bool, delay_seconds: float)``.  When *None*,
            results are never retried based on their content (only on
            exceptions).
        logger: Logger instance; a module-level logger is used when *None*.

    Yields:
        ``(request_id, result_dict)`` tuples as results become available.
    """
    logger = logger or logging.getLogger(__name__)

    start_time = time.monotonic()
    logger.info(
        f"Starting parallel_execute: num_workers={num_workers}, "
        f"max_requests_per_minute={max_requests_per_minute}, "
        f"max_tokens_per_minute={max_tokens_per_minute}, max_retries={max_retries}"
    )

    # -- Shared state ----------------------------------------------------------
    shutdown = Event()
    ready_requests: queue.Queue[tuple[str, Any, int]] = queue.Queue(maxsize=num_workers * 2)
    delayed_requests: list[tuple[float, tuple[str, Any, int]]] = []
    delayed_cv = Condition()
    results: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    pending_count = 0
    pending_lock = Lock()
    all_done = Event()
    producer_done = Event()

    # -- Helper: decrement pending count and signal completion -----------------
    def resolve_request() -> None:
        nonlocal pending_count
        with pending_lock:
            pending_count -= 1
            if producer_done.is_set() and pending_count == 0:
                all_done.set()

    # -- Helper: schedule a delayed retry -------------------------------------
    def schedule_delayed_request(
        request_id: str, payload: dict[str, Any], retry_count: int, retry_delay: float
    ) -> None:
        with delayed_cv:
            heapq.heappush(delayed_requests, (time.monotonic() + retry_delay, (request_id, payload, retry_count)))
            delayed_cv.notify()

    # -- Producer thread -------------------------------------------------------
    def populate_ready_requests(item_iter: Iterable[tuple[str, Any]], shutdown_event: Event) -> None:
        nonlocal pending_count
        logger.debug("Producer: starting to enqueue items")
        enqueued = 0
        for request_id, payload in item_iter:
            if shutdown_event.is_set():
                break
            with pending_lock:
                pending_count += 1
            enqueued_ok = False
            while not shutdown_event.is_set():
                try:
                    ready_requests.put((request_id, payload, max_retries), timeout=0.5)
                    enqueued_ok = True
                    break
                except queue.Full:
                    continue
            if not enqueued_ok:
                with pending_lock:
                    pending_count -= 1
                break
            enqueued += 1
        producer_done.set()
        with pending_lock:
            if pending_count == 0:
                all_done.set()
        logger.debug(f"Producer: finished enqueuing {enqueued} items")

    # -- Delayed-request handler thread ----------------------------------------
    def delayed_requests_handler() -> None:
        try:
            while not shutdown.is_set():
                with delayed_cv:
                    while not delayed_requests and not shutdown.is_set():
                        delayed_cv.wait(timeout=0.5)

                    if shutdown.is_set():
                        break

                    now = time.monotonic()
                    due, request = delayed_requests[0]
                    if due > now:
                        delayed_cv.wait(timeout=due - now)
                        continue

                    heapq.heappop(delayed_requests)

                enqueued_ok = False
                while not shutdown.is_set():
                    try:
                        ready_requests.put(request, timeout=0.5)
                        enqueued_ok = True
                        break
                    except queue.Full:
                        continue
                if not enqueued_ok:
                    request_id = request[0]
                    error_result: dict[str, Any] = {
                        "processing_attempted": True,
                        "processing_failed": True,
                        "error": "Shutdown before retry could be enqueued",
                    }
                    results.put((request_id, error_result))
                    resolve_request()
        except Exception as e:
            logger.exception(f"Delayed handler thread failed: {e}")  # noqa: TRY401
        finally:
            with delayed_cv:
                while delayed_requests:
                    _, req = heapq.heappop(delayed_requests)
                    req_id = req[0]
                    err_result: dict[str, Any] = {
                        "processing_attempted": True,
                        "processing_failed": True,
                        "error": "Shutdown with pending delayed retry",
                    }
                    results.put((req_id, err_result))
                    resolve_request()

    # -- Global back-pressure gate --------------------------------------------
    global_pause = GlobalPause()

    # -- Consumer loop (runs inside ThreadPoolExecutor) ------------------------
    def consumer(
        rq: queue.Queue[tuple[str, Any, int]],
        res: queue.Queue[tuple[str, dict[str, Any]]],
        limiter: LeakyBucketRateLimiter,
        token_limiter: TokenBucketRateLimiter | None,
        shutdown_event: Event,
    ) -> None:
        default_retry_delay = 10  # seconds
        while True:
            try:
                request_id, payload, retry_count = rq.get(timeout=0.5)
            except queue.Empty:
                if shutdown_event.is_set():
                    return
                continue
            try:
                logger.debug(f"Consumer: processing item request_id={request_id}")

                # Honour global back-pressure before acquiring rate-limiter
                # permits.  If another consumer recently received a 429 the
                # pause gate will block us until the Retry-After window has
                # elapsed, preventing the whole pool from flooding.
                global_pause.wait_if_paused(shutdown=shutdown_event)

                if not limiter.acquire(shutdown=shutdown_event):
                    # acquire() only returns False on shutdown — no consumer
                    # will pick up a re-enqueued item, so emit an error
                    # result and resolve to avoid orphaning this request.
                    shutdown_err: dict[str, Any] = {
                        "processing_attempted": False,
                        "processing_failed": True,
                        "error": "Shutdown before request could be processed",
                    }
                    res.put((request_id, shutdown_err))
                    resolve_request()
                    return
                # Also wait for token budget if a token limiter is active.
                if token_limiter is not None:
                    if not token_limiter.acquire(shutdown=shutdown_event):
                        shutdown_err: dict[str, Any] = {
                            "processing_attempted": False,
                            "processing_failed": True,
                            "error": "Shutdown before request could be processed (token budget)",
                        }
                        res.put((request_id, shutdown_err))
                        resolve_request()
                        return
                result = process_fn(payload)

                # Report token usage to the token limiter so it can throttle
                # future requests when the budget is exhausted.
                if token_limiter is not None:
                    token_count = result.get(RESULT_TOTAL_TOKENS_KEY, 0)
                    if token_count > 0:
                        token_limiter.report_tokens(token_count)

                # Check if the caller wants to retry based on the result.
                if should_retry is not None and retry_count > 0:
                    do_retry, retry_delay = should_retry(result)
                    if do_retry:
                        _log_token_usage_headers(result, request_id, logger)
                        # Activate global back-pressure so that *all*
                        # consumers pause for the retry window, not just the
                        # single request that was rate-limited.
                        global_pause.activate(retry_delay, logger=logger)
                        schedule_delayed_request(request_id, payload, retry_count - 1, retry_delay)
                        logger.warning(
                            f"Retryable result. Scheduling retry {max_retries - retry_count + 1} for item {request_id}"
                        )
                        continue

                res.put((request_id, result))
                resolve_request()
                logger.debug(f"Consumer: result for item request_id={request_id} submitted")
            except Exception as e:  # noqa: BLE001
                if retry_count > 0:
                    retry_delay = default_retry_delay * (2 ** (max_retries - retry_count))
                    schedule_delayed_request(request_id, payload, retry_count - 1, retry_delay)
                    logger.info(
                        f"Error processing item {request_id}: {e}. Scheduling retry {max_retries - retry_count + 1}"
                    )
                else:
                    logger.warning(f"Item {request_id} failed after {max_retries} retries with error: {e}")
                    error_result: dict[str, Any] = {
                        "processing_attempted": True,
                        "processing_failed": True,
                        "error": str(e),
                    }
                    res.put((request_id, error_result))
                    resolve_request()
            finally:
                rq.task_done()

    # -- Orchestration ---------------------------------------------------------
    with LeakyBucketRateLimiter(
        max_requests_per_minute=max_requests_per_minute, time_to_stop=shutdown, logger=logger
    ) as limiter:
        # Optionally create a token-rate limiter alongside the request-rate
        # limiter.  The effective throughput is governed by whichever limit is
        # stricter.
        token_limiter_ctx: TokenBucketRateLimiter | None = None
        if max_tokens_per_minute is not None:
            token_limiter_ctx = TokenBucketRateLimiter(
                max_tokens_per_minute=max_tokens_per_minute, time_to_stop=shutdown, logger=logger
            )

        with token_limiter_ctx if token_limiter_ctx is not None else _nullcontext() as active_token_limiter:
            producer_thread = Thread(target=populate_ready_requests, args=(items, shutdown), daemon=True)
            delayed_handler_thread = Thread(target=delayed_requests_handler, daemon=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [
                    ex.submit(consumer, ready_requests, results, limiter, active_token_limiter, shutdown)
                    for _ in range(num_workers)
                ]
                try:
                    producer_thread.start()
                    delayed_handler_thread.start()
                    yielded_count = 0
                    while not all_done.is_set() or not results.empty():
                        try:
                            yield results.get(timeout=0.5)
                            results.task_done()
                            yielded_count += 1
                            if yielded_count % 100 == 0:
                                elapsed = time.monotonic() - start_time
                                logger.info(
                                    f"Progress: {yielded_count} results yielded so far ({elapsed:.1f}s elapsed)"
                                )
                        except queue.Empty:
                            continue
                    shutdown.set()
                    elapsed = time.monotonic() - start_time
                    logger.info(f"Completed: {yielded_count} results yielded in {elapsed:.1f}s")
                finally:
                    shutdown.set()
                    producer_thread.join()
                    for f in futures:
                        f.result()
                    delayed_handler_thread.join()
                    # Drain any items still in ready_requests that no
                    # consumer will ever pick up (all have exited).  This
                    # ensures pending_count reaches zero and results are
                    # emitted for every input item.
                    while True:
                        try:
                            req_id, _payload, _retry = ready_requests.get_nowait()
                            drain_err: dict[str, Any] = {
                                "processing_attempted": False,
                                "processing_failed": True,
                                "error": "Shutdown with request still in queue",
                            }
                            results.put((req_id, drain_err))
                            resolve_request()
                            ready_requests.task_done()
                        except queue.Empty:
                            break


class _nullcontext:
    """Minimal no-op context manager (available in stdlib from 3.7 but
    re-implemented here for type-compatibility with the token limiter)."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        pass
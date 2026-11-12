#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for Perplexity's Decisions API.

One :class:`PerplexityDecisionsClient` holds one HTTP/2 connection pool, adds
the auth header, retries when Perplexity is busy, bounds every request in
time, and counts the tokens every request used. Several
:class:`~pipecat.classifiers.perplexity.decisions.classifier.PerplexityDecisionsClassifier`
instances can share one.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.classifiers.base_classifier import ClassifierError
from pipecat.utils.network import exponential_backoff_time

try:
    import httpx
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Perplexity Decisions, you need to `uv add "pipecat-ai[perplexity]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

#: How long an idle connection is kept open, in seconds. Questions come with
#: gaps between them, and a connection that has closed in the meantime costs a
#: new TLS handshake on the next one.
_KEEPALIVE_EXPIRY = 240.0
#: The longest wait between retries, in seconds, whatever ``Retry-After`` asks.
_MAX_RETRY_WAIT = 2.0

#: Where the API is served.
DEFAULT_BASE_URL = "https://api.perplexity.ai"
#: The decision model to ask: the 4B model, the faster of the two. Pinned so
#: thresholds tuned against it hold; ``pplx-decider-flash-latest`` and
#: ``pplx-decider-latest`` follow Perplexity's newest releases.
DEFAULT_MODEL = "pplx-decider-flash-v1.2"
#: Seconds to wait for an answer, retries included, before giving up.
DEFAULT_TIMEOUT = 10.0


class PerplexityDecisionsUsage(BaseModel):
    """Tokens the Decisions API used: for one request, or over every request.

    :attr:`PerplexityDecisionsClient.usage` holds the total over every request.

    Parameters:
        input_tokens: Tokens sent, web evidence and fetched pages included.
        output_tokens: Tokens received: one per question, and free.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class PerplexityDecisionsClient:
    """HTTP client for Perplexity's ``/v1/decisions`` endpoint.

    Holds one HTTP/2 connection pool, so many small requests share a
    connection and can be in flight at the same time. The connection is kept
    open between questions, so a gap between them does not cost a new TLS
    handshake. Retries with backoff when Perplexity answers 429 or a 5xx other
    than 504, waiting what ``Retry-After`` asks when that is two seconds or
    less, and giving up at once when it asks for longer. Bounds each
    request, retries included, by ``timeout``. Counts the tokens every request
    used in :attr:`usage`. Errors carry the reason Perplexity gave.

    The first question opens the connection, and :meth:`close` releases it.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize the client.

        Args:
            api_key: Perplexity API key. Only a server that needs none, such
                as Perplexity's local Decider at ``http://127.0.0.1:8080``,
                goes without.
            base_url: Where the API is served.
            model: The decision model to ask: ``pplx-decider-flash-v1.2``
                (4B), ``pplx-decider-v1.2`` (27B), or an alias that follows
                the newest release of either.
            timeout: Seconds to wait for an answer, retries included, before
                giving up.
            max_retries: How many times to retry a request that failed for a
                reason that may pass, such as Perplexity being busy.
        """
        if not api_key and base_url == DEFAULT_BASE_URL:
            raise ValueError("PerplexityDecisionsClient needs an API key")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._usage = PerplexityDecisionsUsage()
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
            http2=True,
            limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
        )

    @property
    def model(self) -> str:
        """The decision model the questions go to."""
        return self._model

    @property
    def usage(self) -> PerplexityDecisionsUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def close(self):
        """Close the connection pool."""
        await self._http.aclose()

    async def ask(
        self, state: str | dict[str, Any] | list[Any], questions: Mapping[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], PerplexityDecisionsUsage]:
        """Send questions about one state and return Perplexity's answers.

        Args:
            state: What the questions are about: text, an object or an array,
                with any images as ``image_url`` parts in place.
            questions: The questions by name, each in Perplexity's own format:
                a ``type`` of ``noul``, ``choice`` or ``score``,
                ``instructions``, and ``criteria``.

        Returns:
            The answers by the same names, in Perplexity's own format, and the
            tokens the request used.

        Raises:
            ClassifierError: If Perplexity rejected the request, kept refusing
                it because it was busy, could not be reached, did not answer
                in time, or left a question unanswered.
        """
        body: dict[str, Any] = {"model": self._model, "state": state, "questions": dict(questions)}
        try:
            async with asyncio.timeout(self._timeout):
                data = await self._post(body)
        except TimeoutError as e:
            raise ClassifierError(
                f"Perplexity Decisions did not answer within {self._timeout:g}s"
            ) from e
        usage = data.get("usage")
        request_usage = (
            PerplexityDecisionsUsage.model_validate(usage)
            if isinstance(usage, dict)
            else PerplexityDecisionsUsage()
        )
        self._usage.input_tokens += request_usage.input_tokens
        self._usage.output_tokens += request_usage.output_tokens
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ClassifierError("Perplexity Decisions reply has no answers")
        missing = [name for name in questions if not isinstance(answers.get(name), dict)]
        if missing:
            raise ClassifierError(
                f"Perplexity Decisions reply has no answer for {', '.join(missing)}"
            )
        return {name: answers[name] for name in questions}, request_usage

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """The reply to ``body`` as a JSON object, retrying while Perplexity is busy."""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.post("/v1/decisions", json=body)
            except httpx.TimeoutException as e:
                raise TimeoutError from e
            except (httpx.HTTPError, OSError) as e:
                # Some httpx errors carry no message, and a broken TLS
                # connection can come through as an ssl.SSLError.
                raise ClassifierError(f"Perplexity Decisions request failed: {e!r}") from e
            if self._retryable(response.status_code):
                wait = self._retry_wait(response, attempt)
                if attempt > self._max_retries or wait is None:
                    raise ClassifierError(
                        f"Perplexity Decisions is busy ({self._failure(response)}) "
                        f"after {attempt} attempts"
                    )
                logger.debug(
                    f"Perplexity Decisions answered {response.status_code}, retrying in {wait}s"
                )
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                raise ClassifierError(
                    f"Perplexity Decisions rejected the request: {self._failure(response)}"
                )
            try:
                data = response.json()
            except ValueError as e:
                raise ClassifierError(f"Perplexity Decisions reply is not valid JSON: {e}") from e
            if not isinstance(data, dict):
                raise ClassifierError("Perplexity Decisions reply is not a JSON object")
            return data

    def _retryable(self, status: int) -> bool:
        """Whether a retry may succeed: rate limited, or the service failed.

        A 504 is left out: it comes after about a minute, long past any bound
        a voice agent can wait.
        """
        return status == 429 or (500 <= status < 600 and status != 504)

    def _retry_wait(self, response: httpx.Response, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None if it is too long.

        ``Retry-After`` is honored up to two seconds. A longer wait means
        another attempt within it would be refused too.
        """
        try:
            retry_after = float(response.headers["Retry-After"])
        except (KeyError, ValueError):
            return exponential_backoff_time(
                attempt, min_wait=0.25, max_wait=_MAX_RETRY_WAIT, multiplier=0.25
            )
        return max(retry_after, 0.0) if retry_after <= _MAX_RETRY_WAIT else None

    def _failure(self, response: httpx.Response) -> str:
        """A failed response's HTTP status, the reason Perplexity gave, and its request id."""
        status = f"HTTP {response.status_code}"
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = None
        if message:
            status = f"{status}: {message}"
        if request_id := response.headers.get("x-request-id"):
            status = f"{status} (request {request_id})"
        return status

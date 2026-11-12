#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import json
import os
import ssl
from collections.abc import Callable

import httpx
import pytest
from PIL import Image, ImageDraw

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ClassifierError,
    ClassifierImage,
    ScoreQuestion,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.perplexity.decisions.classifier import (
    PERPLEXITY_DECISIONS_MAX_CHOICE_OPTIONS,
    PERPLEXITY_DECISIONS_MAX_IMAGES,
    PERPLEXITY_DECISIONS_MAX_SCORE_LEVELS,
    PerplexityDecisionsClassifier,
)
from pipecat.classifiers.perplexity.decisions.client import PerplexityDecisionsClient
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData

CLIENT_MODULE = "pipecat.classifiers.perplexity.decisions.client"
USAGE = {"input_tokens": 12, "output_tokens": 1}
NOUL = {"answer": {"type": "noul", "instructions": "?"}}


def _client(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs
) -> PerplexityDecisionsClient:
    """A client whose requests ``handler`` answers instead of Perplexity."""
    client = PerplexityDecisionsClient(api_key="key", **kwargs)
    client._http._transport = httpx.MockTransport(handler)
    return client


def _reply(**answers: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={"model": "pplx-decider-flash-v1.2", "answers": answers, "usage": USAGE},
        headers={"x-request-id": "req-1"},
    )


def _noul(probability: float) -> dict:
    return {"type": "noul", "noul": probability}


def _failure(status: int, message: str, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": message, "type": "invalid_request_error", "code": None}},
        headers=headers,
    )


def _image(width: int, height: int, color: str = "red") -> ClassifierImage:
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).ellipse(
        (width // 3, height // 3, 2 * width // 3, 2 * height // 3), fill=color
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return ClassifierImage(data=buffer.getvalue(), content_type="image/jpeg")


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Retry waits, recorded instead of slept."""
    seen: list[float] = []

    async def no_sleep(seconds):
        seen.append(seconds)

    monkeypatch.setattr(f"{CLIENT_MODULE}.asyncio.sleep", no_sleep)
    return seen


class TestPerplexityDecisionsClient:
    @pytest.mark.asyncio
    async def test_sends_questions_to_the_decisions_endpoint_with_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return _reply(answer=_noul(0.9))

        client = _client(handler)
        answers, usage = await client.ask(
            {"user": "hello"}, {"answer": {"type": "noul", "instructions": "a greeting?"}}
        )

        assert answers["answer"] == _noul(0.9)
        assert (usage.input_tokens, usage.output_tokens) == (12, 1)
        assert seen["url"] == "https://api.perplexity.ai/v1/decisions"
        assert seen["auth"] == "Bearer key"
        assert seen["body"] == {
            "model": "pplx-decider-flash-v1.2",
            "state": {"user": "hello"},
            "questions": {"answer": {"type": "noul", "instructions": "a greeting?"}},
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_web_grounding_sets_the_mode_and_staleness(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return _reply(answer=_noul(0.9))

        grounded = _client(handler, web_grounding=True, max_staleness=86400)
        await grounded.ask("https://example.com", NOUL)
        # Staleness alone does nothing without web grounding, so it isn't sent.
        plain = _client(handler, max_staleness=86400)
        await plain.ask("a", NOUL)

        assert (bodies[0]["mode"], bodies[0]["max_staleness"]) == ("web_grounding", 86400)
        assert "mode" not in bodies[1] and "max_staleness" not in bodies[1]
        await grounded.close()
        await plain.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests(self):
        client = _client(lambda request: _reply(answer=_noul(0.5)))
        await client.ask("a", NOUL)
        await client.ask("b", NOUL)

        assert (client.usage.input_tokens, client.usage.output_tokens) == (24, 2)
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("busy", [429, 500, 501, 502, 503])
    async def test_retries_when_busy(self, waits, busy):
        statuses = iter([busy, busy])

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses, None)
            if status is not None:
                return _failure(status, "Request rate limit exceeded")
            return _reply(answer=_noul(0.7))

        client = _client(handler)
        answers, _ = await client.ask("a", NOUL)

        assert answers["answer"]["noul"] == 0.7
        assert waits == [0.25, 0.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_a_retry_waits_what_retry_after_asks_up_to_two_seconds(self, waits):
        statuses = iter([("0.75",), ("2",)])

        def handler(request: httpx.Request) -> httpx.Response:
            retry_after = next(statuses, None)
            if retry_after is not None:
                return _failure(429, "slow down", **{"Retry-After": retry_after[0]})
            return _reply(answer=_noul(0.7))

        client = _client(handler)
        await client.ask("a", NOUL)

        assert waits == [0.75, 2.0]
        await client.close()

    @pytest.mark.asyncio
    async def test_a_retry_after_over_two_seconds_fails_at_once(self, waits):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return _failure(429, "slow down", **{"Retry-After": "30"})

        client = _client(handler)

        with pytest.raises(ClassifierError, match=r"busy \(HTTP 429: slow down\) after 1"):
            await client.ask("a", NOUL)
        assert (len(calls), waits) == (1, [])
        await client.close()

    @pytest.mark.asyncio
    async def test_a_gateway_timeout_is_not_retried(self, waits):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(504, text="<html>Gateway Timeout</html>")

        client = _client(handler)

        with pytest.raises(ClassifierError, match=r"rejected the request: HTTP 504$"):
            await client.ask("a", NOUL)
        assert (len(calls), waits) == (1, [])
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries_with_perplexitys_reason(self, waits):
        client = _client(
            lambda request: _failure(429, "Request rate limit exceeded"), max_retries=2
        )

        with pytest.raises(
            ClassifierError, match=r"busy \(HTTP 429: Request rate limit exceeded\) after 3"
        ):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_stop_at_the_timeout(self, monkeypatch):
        async def forever(seconds):
            await asyncio.Event().wait()

        monkeypatch.setattr(f"{CLIENT_MODULE}.asyncio.sleep", forever)
        client = _client(lambda request: _failure(503, "overloaded"), timeout=0)

        with pytest.raises(ClassifierError, match="did not answer within 0s"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_an_http_timeout_is_reported_as_not_answering_in_time(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        client = _client(handler, timeout=1.5)

        with pytest.raises(ClassifierError, match="did not answer within 1.5s"):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    async def test_a_rejected_request_carries_perplexitys_reason_and_request_id(self):
        client = _client(
            lambda request: _failure(
                400, "Noul question must have criteria or instructions", **{"x-request-id": "abc"}
            )
        )

        with pytest.raises(
            ClassifierError,
            match=r"rejected the request: HTTP 400: Noul question must have criteria "
            r"or instructions \(request abc\)",
        ):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body, error",
        [
            (httpx.Response(200, text="<html>busy</html>"), "not valid JSON"),
            (httpx.Response(200, json=[]), "not a JSON object"),
            (httpx.Response(200, json={"model": "m"}), "no answers"),
            (_reply(other=_noul(0.5)), "no answer for answer"),
            (_reply(answer="0.5"), "no answer for answer"),
        ],
    )
    async def test_an_unusable_reply_is_an_error(self, body, error):
        client = _client(lambda request: body)

        with pytest.raises(ClassifierError, match=error):
            await client.ask("a", NOUL)
        await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raised", [httpx.ConnectError("down"), ssl.SSLError("bad mac")])
    async def test_unreachable_is_an_error(self, raised):
        def handler(request: httpx.Request) -> httpx.Response:
            raise raised

        client = _client(handler)

        with pytest.raises(ClassifierError, match="failed"):
            await client.ask("a", NOUL)
        await client.close()

    def test_needs_an_api_key_only_for_perplexitys_own_api(self):
        with pytest.raises(ValueError):
            PerplexityDecisionsClient(api_key="")

    @pytest.mark.asyncio
    async def test_a_local_decider_needs_no_key(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            return _reply(answer=_noul(0.9))

        client = PerplexityDecisionsClient(base_url="http://127.0.0.1:8080")
        client._http._transport = httpx.MockTransport(handler)
        await client.ask("a", NOUL)

        assert seen == {"url": "http://127.0.0.1:8080/v1/decisions", "auth": None}
        await client.close()


class TestPerplexityDecisionsClassifier:
    @pytest.mark.asyncio
    async def test_yes_no_maps_both_sides_of_the_criteria(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["questions"] = json.loads(request.content)["questions"]
            return _reply(plain=_noul(0.2), voicemail=_noul(0.93))

        classifier = PerplexityDecisionsClassifier(client=_client(handler))
        results = await classifier.yes_no(
            "Hi, you've reached Sam. Leave a message.",
            {
                "plain": YesNoQuestion(instructions="a greeting?"),
                "voicemail": YesNoQuestion(
                    instructions="is this a voicemail?", yes="a recorded greeting"
                ),
            },
        )

        assert results["voicemail"].probability == 0.93 and results["voicemail"].is_yes
        assert not results["plain"].is_yes
        assert seen["questions"] == {
            "plain": {"type": "noul", "instructions": "a greeting?"},
            "voicemail": {
                "type": "noul",
                "instructions": "is this a voicemail?",
                "criteria": {"true": "a recorded greeting", "false": ""},
            },
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_structured_state_and_instructions_go_as_they_are(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return _reply(answer=_noul(0.5))

        state = [{"role": "assistant", "content": "Where to?"}, {"role": "user", "content": "Rome"}]
        classifier = PerplexityDecisionsClassifier(client=_client(handler))
        await classifier.yes_no(
            state, {"answer": YesNoQuestion(instructions={"question": "complete?"})}
        )

        assert seen["body"]["state"] == state
        assert seen["body"]["questions"]["answer"]["instructions"] == {"question": "complete?"}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_choice(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["turn"]
            return _reply(
                turn={
                    "type": "choice",
                    "choice": "short",
                    "confidence": 0.7,
                    "probabilities": {"complete": 0.1, "short": 0.8},
                }
            )

        classifier = PerplexityDecisionsClassifier(client=_client(handler))
        results = await classifier.choice(
            "I'd go to Japan because I love",
            {
                "turn": ChoiceQuestion(
                    instructions="is the turn complete?",
                    options={"complete": "finished", "short": "cut off", "long": None},
                )
            },
        )

        assert seen["question"] == {
            "type": "choice",
            "instructions": "is the turn complete?",
            "criteria": {"complete": "finished", "short": "cut off", "long": None},
        }
        turn = results["turn"]
        assert (turn.choice, turn.confidence) == ("short", 0.7)
        assert turn.probabilities == {"complete": 0.1, "short": 0.8, "long": 0.0}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_score_maps_positions_back_to_levels(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["mood"]
            return _reply(
                mood={
                    "type": "score",
                    "score": 1.78,
                    "confidence": 0.78,
                    "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                    "probabilities": {"0": 0.01, "1": 0.2, "2": 0.79},
                }
            )

        classifier = PerplexityDecisionsClassifier(client=_client(handler))
        results = await classifier.score(
            "This is the third time I'm calling.",
            {
                "mood": ScoreQuestion(
                    instructions="how upset?", levels=["calm", "frustrated", "angry"]
                )
            },
        )

        assert seen["question"]["criteria"] == ["calm", "frustrated", "angry"]
        mood = results["mood"]
        assert (mood.score, mood.confidence) == (1.78, 0.78)
        assert mood.probability("angry") == 0.79
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_ask_sends_every_question_in_one_request(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content)["questions"])
            return _reply(
                a=_noul(0.9),
                b={"type": "choice", "choice": "x", "confidence": 1.0, "probabilities": {"x": 1}},
            )

        classifier = PerplexityDecisionsClassifier(client=_client(handler))
        results = await classifier.ask(
            "s",
            {
                "a": YesNoQuestion(instructions="a?"),
                "b": ChoiceQuestion(instructions="b?", options={"x": None}),
            },
        )

        assert len(calls) == 1 and set(calls[0]) == {"a", "b"}
        assert results["a"].probability == 0.9 and results["b"].choice == "x"
        await classifier.client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "question, error",
        [
            (
                ChoiceQuestion(
                    instructions="?",
                    options={
                        str(i): None for i in range(PERPLEXITY_DECISIONS_MAX_CHOICE_OPTIONS + 1)
                    },
                ),
                "at most 255 options",
            ),
            (
                ScoreQuestion(
                    instructions="?",
                    levels=[str(i) for i in range(PERPLEXITY_DECISIONS_MAX_SCORE_LEVELS + 1)],
                ),
                "at most 10 levels",
            ),
        ],
    )
    async def test_a_question_over_the_apis_limits_fails_before_any_request(self, question, error):
        calls = []
        classifier = PerplexityDecisionsClassifier(
            client=_client(lambda request: calls.append(1) or _reply())
        )

        with pytest.raises(ClassifierError, match=error):
            await classifier.ask("s", {"q": question})
        assert calls == []
        await classifier.client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer, error",
        [
            ({"type": "choice", "choice": "maybe", "confidence": 1}, "not an option"),
            (
                {"type": "choice", "choice": "x", "confidence": 1, "probabilities": {"x": "hi"}},
                "unusable probability",
            ),
            ({"type": "choice", "choice": "x"}, "no usable 'confidence'"),
        ],
    )
    async def test_an_unusable_answer_is_an_error(self, answer, error):
        classifier = PerplexityDecisionsClassifier(client=_client(lambda request: _reply(q=answer)))

        with pytest.raises(ClassifierError, match=error):
            await classifier.choice(
                "s", {"q": ChoiceQuestion(instructions="?", options={"x": None})}
            )
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_images_go_ahead_of_the_state_as_image_url_parts(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["state"] = json.loads(request.content)["state"]
            return _reply(red=_noul(0.97))

        image = _image(640, 480)
        classifier = PerplexityDecisionsClassifier(client=_client(handler))
        await classifier.yes_no(
            {"caption": "camera"}, {"red": YesNoQuestion(instructions="red?")}, inputs=[image]
        )

        part, state = seen["state"]
        assert state == {"caption": "camera"}
        assert part["type"] == "image_url"
        prefix, data = part["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        assert base64.b64decode(data) == image.data
        await classifier.client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "images, error",
        [
            ([_image(64, 64)] * (PERPLEXITY_DECISIONS_MAX_IMAGES + 1), "at most 8 images"),
            # 1600x1310 is 50x41 tiles, over the 2048 the API reads.
            ([_image(1600, 1310)], "got 1600x1310"),
            # A partial tile counts as a whole one: 2064 is 64.5, so 65x32.
            ([_image(2064, 1024)], "got 2064x1024"),
        ],
    )
    async def test_images_over_the_apis_limits_fail_before_any_request(self, images, error):
        calls = []
        classifier = PerplexityDecisionsClassifier(
            client=_client(lambda request: calls.append(1) or _reply())
        )

        with pytest.raises(ClassifierError, match=error):
            await classifier.yes_no("s", {"q": YesNoQuestion(instructions="?")}, inputs=images)
        assert calls == []
        await classifier.client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [(1440, 1440), (2048, 1024)])
    async def test_the_largest_images_the_api_reads_are_sent(self, size):
        calls = []
        classifier = PerplexityDecisionsClassifier(
            client=_client(lambda request: calls.append(1) or _reply(q=_noul(0.5)))
        )

        await classifier.yes_no("s", {"q": YesNoQuestion(instructions="?")}, inputs=[_image(*size)])

        assert calls == [1]
        await classifier.client.close()

    def test_needs_an_api_key_or_a_client(self):
        with pytest.raises(ValueError):
            PerplexityDecisionsClassifier()

    def test_a_client_of_its_own_takes_its_settings(self):
        classifier = PerplexityDecisionsClassifier(
            api_key="key", model="pplx-decider-v1.2", timeout=1.5, web_grounding=True
        )
        assert classifier.model == "pplx-decider-v1.2"
        assert classifier.client._http.timeout.read == 1.5
        assert classifier.client.web_grounding

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = PerplexityDecisionsClassifier(api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = PerplexityDecisionsClient(api_key="key")
        classifier = PerplexityDecisionsClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()

    @pytest.mark.asyncio
    async def test_search_results_come_to_the_event_only_with_web_grounding(self):
        page = {"title": "Sam", "url": "https://example.com", "snippet": "hi", "date": None}
        reply = {
            "model": "pplx-decider-flash-v1.2",
            "answers": {"answer": _noul(0.9)},
            "usage": USAGE,
            # Malformed pages are left out rather than failing the answers.
            "search_results": [page, {"title": "no url"}, {"title": None, "url": "https://x"}],
            "fetched_results": [],
        }

        async def ask(web_grounding: bool) -> tuple[list, YesNoResult]:
            client = _client(lambda request: httpx.Response(200, json=reply))
            client._web_grounding = web_grounding
            classifier = PerplexityDecisionsClassifier(client=client)
            searched, measured = asyncio.Event(), asyncio.Event()
            seen = []

            @classifier.event_handler("on_search_results")
            async def on_search_results(classifier, search_results, fetched_results):
                seen.append((search_results, fetched_results))
                searched.set()

            @classifier.event_handler("on_metrics")
            async def on_metrics(classifier, data):
                measured.set()

            results = await classifier.yes_no(
                "a", {"answer": YesNoQuestion(instructions="a greeting?")}
            )
            # Without grounding there is no search event: the metrics event,
            # which comes after it, shows the call is over.
            await asyncio.wait_for((searched if web_grounding else measured).wait(), 1)
            await client.close()
            return seen, results["answer"]

        [(search_results, fetched_results)], answer = await ask(web_grounding=True)
        assert [result.url for result in search_results] == ["https://example.com"]
        assert fetched_results == []
        assert answer.probability == 0.9
        seen, _ = await ask(web_grounding=False)
        assert seen == []

    @pytest.mark.asyncio
    async def test_every_call_reports_its_time_and_tokens(self):
        client = _client(lambda request: _reply(answer=_noul(0.9)))
        classifier = PerplexityDecisionsClassifier(client=client, name="voicemail")
        reported = asyncio.Event()
        seen: list = []

        @classifier.event_handler("on_metrics")
        async def on_metrics(classifier, data):
            seen.extend(data)
            reported.set()

        await classifier.yes_no("hello", {"answer": YesNoQuestion(instructions="a greeting?")})
        await asyncio.wait_for(reported.wait(), 1)

        processing, usage = seen
        assert isinstance(processing, ProcessingMetricsData)
        assert (processing.processor, processing.model) == ("voicemail", "pplx-decider-flash-v1.2")
        assert isinstance(usage, LLMUsageMetricsData)
        assert (usage.value.prompt_tokens, usage.value.total_tokens) == (12, 13)
        await client.close()


@pytest.mark.skipif(not os.getenv("PERPLEXITY_API_KEY"), reason="PERPLEXITY_API_KEY not set")
class TestPerplexityDecisionsLive:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["pplx-decider-flash-v1.2", "pplx-decider-v1.2"])
    async def test_three_questions(self, model):
        classifier = PerplexityDecisionsClassifier(
            api_key=os.environ["PERPLEXITY_API_KEY"], model=model
        )
        try:
            results = await classifier.ask(
                "Hi, you've reached Sam. I can't take your call right now, leave a message.",
                {
                    "voicemail": YesNoQuestion(instructions="is this a voicemail greeting?"),
                    "turn": ChoiceQuestion(
                        instructions="is the speaker's turn over?",
                        options={"complete": "finished", "short": "a brief pause"},
                    ),
                    "mood": ScoreQuestion(
                        instructions="how upset is the speaker?",
                        levels=["calm", "impatient", "angry"],
                    ),
                },
            )
            assert results["voicemail"].probability > 0.5
            assert abs(sum(results["turn"].probabilities.values()) - 1.0) < 0.05
            assert 0 <= results["mood"].score <= 2
            assert classifier.client.usage.input_tokens > 0
        finally:
            await classifier.cleanup()

    @pytest.mark.asyncio
    async def test_a_question_about_an_image(self):
        classifier = PerplexityDecisionsClassifier(api_key=os.environ["PERPLEXITY_API_KEY"])
        question = {"red": YesNoQuestion(instructions="Is there a red circle in the image?")}
        try:
            red = await classifier.yes_no("A camera frame.", question, inputs=[_image(640, 480)])
            blue = await classifier.yes_no(
                "A camera frame.", question, inputs=[_image(640, 480, "blue")]
            )
            assert red["red"].is_yes
            assert not blue["red"].is_yes
        finally:
            await classifier.cleanup()

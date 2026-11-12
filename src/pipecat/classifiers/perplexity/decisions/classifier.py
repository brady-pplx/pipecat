#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Perplexity's Decisions API.

:class:`PerplexityDecisionsClassifier` turns each question into a request and
each reply into a result, through a
:class:`~pipecat.classifiers.perplexity.decisions.client.PerplexityDecisionsClient`.
"""

import base64
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierImage,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    ScoreLevel,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.perplexity.decisions.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    PerplexityDecisionsClient,
)
from pipecat.metrics.metrics import LLMTokenUsage

#: The most options the Decisions API takes in one choice question.
PERPLEXITY_DECISIONS_MAX_CHOICE_OPTIONS = 255
#: The most levels the Decisions API takes in one score question.
PERPLEXITY_DECISIONS_MAX_SCORE_LEVELS = 10
#: The most images the Decisions API reads for one question.
PERPLEXITY_DECISIONS_MAX_IMAGES = 8
#: The most 32x32 tiles the Decisions API reads in one image. A larger image is
#: not rejected: the request times out after about a minute.
PERPLEXITY_DECISIONS_MAX_IMAGE_TILES = 2048


class PerplexityDecisionsClassifier(BaseClassifier):
    """Answers questions by asking Perplexity's Decisions API.

    Build one with an API key to get a client of its own, or pass a
    :class:`PerplexityDecisionsClient` to share one between several
    classifiers. A client the classifier created is closed in
    :meth:`cleanup`; a shared one is left to whoever made it.

    The state, instructions, criteria and levels go as they are: text, or
    structured data.

    The Decisions API can see PNG, JPEG and WebP images: up to
    :data:`PERPLEXITY_DECISIONS_MAX_IMAGES` per call, each at most
    :data:`PERPLEXITY_DECISIONS_MAX_IMAGE_TILES` tiles of 32x32 pixels, such
    as 1440x1440. They go ahead of the state, in the order given.

    ``pplx-decider-flash-v1.2``, the default, answers each question on its
    own and bills the state once per question. ``pplx-decider-v1.2`` answers
    the questions of a call together and bills the state once.

    Example::

        classifier = PerplexityDecisionsClassifier(api_key=os.getenv("PERPLEXITY_API_KEY"))
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: PerplexityDecisionsClient | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            api_key: Perplexity API key, when the classifier should have a
                client of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required, unless ``base_url`` points at a server that needs no
                key, such as Perplexity's local Decider.
            model: The decision model a client of its own asks.
            base_url: Where a client of its own sends its questions.
            timeout: Seconds a client of its own waits for an answer.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._owns_client = client is None
        if not client:
            if not api_key and base_url == DEFAULT_BASE_URL:
                raise ValueError(
                    "PerplexityDecisionsClassifier needs an API key or a PerplexityDecisionsClient"
                )
            client = PerplexityDecisionsClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
            )
        self._client = client

    @property
    def client(self) -> PerplexityDecisionsClient:
        """The client this classifier asks through."""
        return self._client

    async def cleanup(self):
        """Close the client if this classifier created it."""
        await super().cleanup()
        if self._owns_client:
            await self._client.close()

    @property
    def model(self) -> str:
        """The decision model the questions go to."""
        return self._client.model

    @property
    def supports_images(self) -> bool:
        """The Decisions API can see images."""
        return True

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage]:
        """Answer the questions in one request."""
        images = [item for item in inputs if isinstance(item, ClassifierImage)]
        if len(images) > PERPLEXITY_DECISIONS_MAX_IMAGES:
            raise ClassifierError(
                f"Perplexity Decisions takes at most {PERPLEXITY_DECISIONS_MAX_IMAGES} images, "
                f"got {len(images)}"
            )
        if images:
            # The model reads each image part where it appears in the state.
            state = [*(self._image_part(image) for image in images), state]
        answers, usage = await self._client.ask(
            state, {name: self._to_perplexity(q) for name, q in questions.items()}
        )
        results = {
            name: self._from_perplexity(question, answers[name])
            for name, question in questions.items()
        }
        return results, LLMTokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    def _image_part(self, image: ClassifierImage) -> dict[str, Any]:
        """An image as an ``image_url`` part, or a ClassifierError if it is too large."""
        with Image.open(io.BytesIO(image.data)) as opened:
            width, height = opened.size
        tiles = math.ceil(width / 32) * math.ceil(height / 32)
        if tiles > PERPLEXITY_DECISIONS_MAX_IMAGE_TILES:
            side = 32 * math.isqrt(PERPLEXITY_DECISIONS_MAX_IMAGE_TILES)
            raise ClassifierError(
                f"Perplexity Decisions reads images up to {PERPLEXITY_DECISIONS_MAX_IMAGE_TILES} "
                f"tiles of 32x32 pixels (about {side}x{side}), got {width}x{height}"
            )
        data = base64.b64encode(image.data).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{image.content_type};base64,{data}"},
        }

    def _to_perplexity(self, question: ClassifierQuestion) -> dict[str, Any]:
        """A question in the Decisions API's own format."""
        instructions = question.instructions
        if isinstance(question, YesNoQuestion):
            noul: dict[str, Any] = {"type": "noul", "instructions": instructions}
            if question.yes is not None or question.no is not None:
                noul["criteria"] = {
                    "true": "" if question.yes is None else question.yes,
                    "false": "" if question.no is None else question.no,
                }
            return noul
        if isinstance(question, ChoiceQuestion):
            if len(question.options) > PERPLEXITY_DECISIONS_MAX_CHOICE_OPTIONS:
                raise ClassifierError(
                    "Perplexity Decisions takes at most "
                    f"{PERPLEXITY_DECISIONS_MAX_CHOICE_OPTIONS} options, "
                    f"got {len(question.options)}"
                )
            return {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(question.options),
            }
        if len(question.levels) > PERPLEXITY_DECISIONS_MAX_SCORE_LEVELS:
            raise ClassifierError(
                f"Perplexity Decisions takes at most {PERPLEXITY_DECISIONS_MAX_SCORE_LEVELS} "
                f"levels, got {len(question.levels)}"
            )
        return {
            "type": "score",
            "instructions": instructions,
            "criteria": list(question.levels),
        }

    def _from_perplexity(
        self, question: ClassifierQuestion, answer: dict[str, Any]
    ) -> ClassifierResult:
        """A result built from the Decisions API's answer to the question."""
        if isinstance(question, YesNoQuestion):
            return YesNoResult(probability=self._number(answer, "noul"))
        if isinstance(question, ChoiceQuestion):
            choice = str(answer.get("choice", ""))
            if choice not in question.options:
                raise ClassifierError(
                    f"Perplexity Decisions chose {choice!r}, which is not an option"
                )
            return ChoiceResult(
                choice=choice,
                probabilities=self._probabilities(answer, list(question.options)),
                confidence=self._number(answer, "confidence"),
            )
        # The Decisions API keys level probabilities by position.
        positions = [str(index) for index in range(len(question.levels))]
        probabilities = self._probabilities(answer, positions)
        return ScoreResult(
            score=self._number(answer, "score"),
            levels=[
                ScoreLevel(level=level, probability=probabilities[position])
                for level, position in zip(question.levels, positions)
            ],
            confidence=self._number(answer, "confidence"),
        )

    def _number(self, answer: dict[str, Any], key: str) -> float:
        """The number the API wrote under ``key``, or a ClassifierError if it did not."""
        try:
            return float(answer[key])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifierError(f"Perplexity Decisions reply has no usable '{key}'") from e

    def _probabilities(self, answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
        """The probability the API gave for each key, 0 for the ones it left out."""
        given = answer.get("probabilities")
        given = given if isinstance(given, dict) else {}
        try:
            return {key: float(given.get(key, 0.0)) for key in keys}
        except (TypeError, ValueError) as e:
            raise ClassifierError(
                f"Perplexity Decisions reply has an unusable probability: {e}"
            ) from e

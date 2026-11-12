- Added `PerplexityDecisionsClassifier` in `pipecat.classifiers.perplexity.decisions.classifier`, a classifier backed by Perplexity's Decisions API (`pplx-decider-flash-v1.2` by default) that can also see images, and can be pointed at Perplexity's local Decider with `base_url`. Install it with `pipecat-ai[perplexity]`.

  ```python
  classifier = PerplexityDecisionsClassifier(api_key=os.getenv("PERPLEXITY_API_KEY"))
  strategies = ClassifierUserTurnStrategies(classifier=classifier)
  ```

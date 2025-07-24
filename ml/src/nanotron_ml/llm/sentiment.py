"""FinBERT-style sentiment over headlines.

We stay framework-light: the actual model loading is lazy and the
``FinBERTSentiment`` class falls back to a simple rule-based scorer
when ``transformers`` isn't available.  The contract is the same in
both cases: a list of strings → a list of {neg, neu, pos} dicts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinBERTSentiment:
    """Wraps a HuggingFace FinBERT pipeline (lazy)."""

    model_name: str = "ProsusAI/finbert"
    device: str = "cpu"
    _pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline  # type: ignore[import-not-found]

        self._pipe = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            device=-1 if self.device == "cpu" else int(self.device),
        )

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        if not texts:
            return []
        try:
            self._ensure_loaded()
        except ImportError:
            return [_rule_based(t) for t in texts]
        out = self._pipe(texts, truncation=True, max_length=256)
        normalised = []
        for r in out:
            label = r["label"].lower()
            score = float(r["score"])
            normalised.append(
                {
                    "negative": score if label == "negative" else (1 - score) / 2,
                    "neutral": score if label == "neutral" else (1 - score) / 2,
                    "positive": score if label == "positive" else (1 - score) / 2,
                }
            )
        return normalised


_POS_LEXICON = {"beat", "beats", "raises", "upgrade", "buy", "growth", "strong", "surge", "rally", "outperform"}
_NEG_LEXICON = {"miss", "misses", "downgrade", "sell", "weak", "plunge", "drop", "underperform", "lawsuit", "fraud", "halts"}


def _rule_based(text: str) -> dict[str, float]:
    tokens = {t.strip(".,!?;").lower() for t in text.split()}
    pos = len(tokens & _POS_LEXICON)
    neg = len(tokens & _NEG_LEXICON)
    if pos == 0 and neg == 0:
        return {"negative": 0.1, "neutral": 0.8, "positive": 0.1}
    total = pos + neg
    return {
        "negative": neg / total * 0.9 + 0.05,
        "neutral": 0.05,
        "positive": pos / total * 0.9 + 0.05,
    }


def batch_sentiment(model: FinBERTSentiment, texts: list[str], batch_size: int = 32) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(texts), batch_size):
        out.extend(model.predict(texts[i : i + batch_size]))
    return out

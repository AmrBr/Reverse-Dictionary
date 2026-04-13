import re
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

db = MorphologyDB.builtin_db() # calima-msa-r13
analyzer = Analyzer(db)

def compute_metrics(results: list[dict]) -> dict:
    """Compute Top-1, Top-5, and MRR from a list of result dicts."""
    top1 = top5 = mrr_sum = 0
    n = len(results)

    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "total": 0}

    for r in results:
        label = normalise(r["label"])
        preds = [normalise(p.lower()) for p in r["predictions"]]

        if preds and preds[0] == label:
            top1 += 1
        if label in preds:
            top5 += 1
            rank = preds.index(label) + 1
            mrr_sum += 1 / rank

    return {
        "top1":  round(top1  / n, 4),
        "top5":  round(top5  / n, 4),
        "mrr":   round(mrr_sum / n, 4),
        "total": n,
    }

def normalise(word: str) -> str:
    """Normalise Arabic words for more forgiving matching."""
    # Strip diacritics (tashkeel)
    word = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", word)
    # Strip tatweel
    word = re.sub(r"\u0640", "", word)
    # Normalise alef variants → ا
    word = re.sub(r"[أإآ]", "ا", word)
    # Strip definite article
    word = re.sub(r"^ال", "", word)
    # Lemmatise using CAMeL Tools (fallback to original word if no analysis)
    analyses = analyzer.analyze(word)
    if analyses:
        return analyses[0]['lemma']
    return word.strip()

def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n── Results ──────────────────────")
    print(f"  Top-1 : {metrics['top1']:.2%}")
    print(f"  Top-5 : {metrics['top5']:.2%}")
    print(f"  MRR   : {metrics['mrr']:.4f}")
    print(f"  Total : {metrics['total']} records")
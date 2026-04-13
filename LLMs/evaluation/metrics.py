import re
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar

db = MorphologyDB.builtin_db() # calima-msa-r13
analyzer = Analyzer(db)

def compute_metrics(results: list[dict]) -> dict:
    """Compute Top-1, Top-5, and MRR from a list of result dicts."""
    top1 = top5 = mrr_sum = 0
    n = len(results)

    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "total": 0}

    for r in results:
        label = r["label"]
        preds = [normalise(p) for p in r["predictions"]]

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

def compute_metrics_normalized(results: list[dict]) -> dict:
    """Compute 'Soft' Top-1, Top-5, and MRR using morphological weights."""
    top1_score = 0
    top5_score = 0
    mrr_sum = 0
    n = len(results)

    if n == 0:
        return {"top1": 0.0, "top5": 0.0, "mrr": 0.0, "total": 0}

    for r in results:
        label = r["label"]
        preds = r["predictions"]
        
        # --- Top-1 Calculation ---
        if preds:
            top1_score += get_similarity_weight(preds[0], label)

        # --- Top-5 & MRR Calculation ---
        best_record_mrr = 0
        found_in_top5 = False
        
        for i, p in enumerate(preds[:5]):
            weight = get_similarity_weight(p, label)
            
            if weight > 0:
                found_in_top5 = True
                rank = i + 1
                current_mrr = 1 / rank
                
                # If multiple matches exist in Top-5, we take the highest MRR contribution
                if current_mrr > best_record_mrr:
                    best_record_mrr = current_mrr
        
        if found_in_top5:
            top5_score += 1 
            mrr_sum += best_record_mrr

    return {
        "top1":  round(top1_score / n, 4),
        "top5":  round(top5_score / n, 4), 
        "mrr":   round(mrr_sum / n, 4),
        "total": n,
    }
    
def get_similarity_weight(pred: str, label: str) -> float:
    """Returns a score between 0.0 and 1.0 based on Arabic similarity."""
    # 1. Exact Match (Cleaned)
    p_norm = normalise(pred)
    l_norm = normalise(label)
    p_norm = re.sub(r"^ال", "", p_norm)
    l_norm = re.sub(r"^ال", "", l_norm)
    
    if p_norm == l_norm:
        return 1.0

    # 2. Morphological Match (Lemma/Root)
    p_analyses = analyzer.analyze(p_norm)
    l_analyses = analyzer.analyze(l_norm)

    if not p_analyses or not l_analyses:
        return 0.0

    # Get primary lemma and root
    p_lemma = p_analyses[0].get('lex')
    l_lemma = l_analyses[0].get('lex')
    p_root = p_analyses[0].get('root')
    l_root = l_analyses[0].get('root')

    # Lemma Match (Handles Singular/Plural/Clitics)
    if p_lemma == l_lemma:
        return 0.8
    
    # Root Match (Handles Verb vs Noun of the same concept)
    if p_root == l_root and p_root is not None:
        return 0.5
    
    return 0.0
    
def normalise(word: str) -> str:
    """Normalise Arabic words for more forgiving matching."""
    # Strip diacritics (tashkeel)
    word = dediac_ar(word)
    # Strip tatweel
    word = re.sub(r"\u0640", "", word)
    word = normalize_alef_maksura_ar(word)
    word = normalize_alef_ar(word)
    word = normalize_teh_marbuta_ar(word)
    return word.strip()

def print_metrics(metrics: dict, stage: str) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n── {stage} Results──────────────────────" )
    print(f"  Top-1 : {metrics['top1']:.2%}")
    print(f"  Top-5 : {metrics['top5']:.2%}")
    print(f"  MRR   : {metrics['mrr']:.4f}")
    print(f"  Total : {metrics['total']} records")
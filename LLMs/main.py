import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

from config.settings import Config
from data.loader import load_hf_dataset, load_already_done
from models import load_model
from evaluation.metrics import compute_metrics, print_metrics, compute_metrics_normalized, generate_report
from evaluation.parser import parse_response
from retrieval.retriever import Retriever
from retrieval.index import build_index

load_dotenv() 

def main():
    cfg = Config()

    # Load dataset
    print("Loading dataset...")
    ds = load_hf_dataset(cfg)

    if cfg.use_rag:
        print("Building index (if not already built)...")
        build_index(cfg)
        print("Initializing retriever...")
        retriever = Retriever(cfg)
    # Load model
    model = load_model(cfg)

    # Resume support
    done_indices = load_already_done(cfg.results_file)
    print(f"Already done: {len(done_indices)} records. Resuming...")

    all_results: list[dict] = []
    if os.path.exists(cfg.results_file):
        with open(cfg.results_file) as f:
            for line in f:
                all_results.append(json.loads(line))

    # Inference loop
    os.makedirs(os.path.dirname(cfg.results_file), exist_ok=True)

    with open(cfg.results_file, "a") as out_file:
        for i, record in enumerate(tqdm(ds, total=len(ds))):
            if i in done_indices:
                continue

            definition = record[cfg.definition_col]
            label      = record[cfg.label_col]

            try:
                examples = retriever.augment(definition) if cfg.use_rag else ""
                prompt = model.build_prompt(definition, examples)
                raw_output  = model.query(prompt)
                predictions = parse_response(raw_output)
            except Exception as e:
                print(f"\nError at index {i}: {e}")
                predictions = []

            result = {
                "index":       i,
                "definition":  definition,
                "label":       label,
                "predictions": predictions,
            }

            all_results.append(result)
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            if i == 999:
                break
    # Metrics
    metrics = compute_metrics(all_results)
    print_metrics(metrics, stage="Raw Matching")

    metrics = compute_metrics_normalized(all_results)
    print_metrics(metrics, stage="Morphological Matching")
    
    generate_report(all_results)

if __name__ == "__main__":
    main()
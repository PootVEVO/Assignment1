"""
Step 1 & 2 (and 6): Sentiment classification of Amazon Gift Card reviews.
Uses an OpenAI-compatible endpoint. Outputs to JSONL.

Usage:
    python sentiment.py [--first N] [--seed SEED] [--balanced] [--classes 3]
    --first N : score only the first N reviews (Step 2, 100 rows)
    --balanced : pull balanced sample (Step 6, ~50 per class)
    --classes 3 : use 3-class labels (POSITIVE/NEUTRAL/NEGATIVE)
"""

import argparse
import gzip
import json
import random
import sys
import time
from openai import OpenAI

DATA_FILE = "Gift_Cards.jsonl.gz"
BASE_URL = "http://dobolyi.com:9001/v1"
API_KEY = "6418"
MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

# --- Prompt ---
PROMPT_2CLASS = """Analyze the following Amazon review and classify its overall sentiment as exactly one of: POSITIVE, NEUTRAL, or NEGATIVE.

Review Title: {title}
Review Text: {text}

Rules:
- Consider the overall opinion expressed in both title and text.
- If title and text conflict, give more weight to the text body.
- Be decisive: choose one label even for brief or ambiguous reviews.
- Output ONLY the single word: POSITIVE, NEUTRAL, or NEGATIVE (no explanation)."""

PROMPT_2CLASS_NO_NEUTRAL = """Analyze the following Amazon review and classify its overall sentiment as exactly one of: POSITIVE or NEGATIVE.

Review Title: {title}
Review Text: {text}

Rules:
- Consider the overall opinion expressed in both title and text.
- If title and text conflict, give more weight to the text body.
- Be decisive: choose one label even for brief or ambiguous reviews.
- Output ONLY the single word: POSITIVE or NEGATIVE (no explanation)."""

# --- Model helper ---
def call_llm(messages, max_tokens=2000):
    """Call the LLM endpoint. The model writes to 'reasoning' field, not 'content'."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    text = (msg.reasoning or msg.content or "").strip()
    # The model outputs a chain-of-thought reasoning, then the actual answer at the end.
    # Scan for the sentiment label starting from the END of the reasoning text.
    words = text.upper().split()
    for w in reversed(words):
        # Strip trailing punctuation like ✅ or .
        clean = w.rstrip(".,;:!'\"")
        if clean in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
            return clean
    # Fallback: if nothing found, return first part of reasoning
    return text[:100]

def load_reviews():
    reviews = []
    with gzip.open(DATA_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    return reviews

def get_label_3class(rating):
    """Map star rating to 3-class label."""
    if rating >= 4:
        return "POSITIVE"
    elif rating == 3:
        return "NEUTRAL"
    else:
        return "NEGATIVE"

def get_label_2class(rating):
    """Map star rating to 2-class label (for Step 2)."""
    if rating >= 4:
        return "POSITIVE"
    else:
        return "NEGATIVE"

def select_reviews(reviews, first_n=None, balanced=False, seed=42):
    """Select reviews based on mode."""
    if first_n is not None:
        return reviews[:first_n]

    if balanced:
        # Assign classes
        labeled = []
        for r in reviews:
            label = get_label_3class(r['rating'])
            labeled.append((r, label))

        random.seed(seed)
        random.shuffle(labeled)

        # Group by class
        by_class = {}
        for r, label in labeled:
            by_class.setdefault(label, []).append(r)

        # Take ~50 from each, up to available
        samples = []
        per_class = 50
        for label in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            pool = by_class.get(label, [])
            samples.extend(random.sample(pool, min(per_class, len(pool))))

        random.shuffle(samples)
        return samples

    return reviews

def run_classification(reviews, use_neutral=False):
    """Classify each review with the LLM."""
    prompt = PROMPT_2CLASS if use_neutral else PROMPT_2CLASS_NO_NEUTRAL
    results = []

    for i, review in enumerate(reviews):
        try:
            prompt_text = prompt.format(title=review['title'], text=review['text'])
            prediction = call_llm([{"role": "user", "content": prompt_text}])
        except Exception as e:
            print(f"Error on review {i}: {e}", file=sys.stderr)
            prediction = "ERROR"
            time.sleep(1)

        review['predicted_sentiment'] = prediction
        results.append(review)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(reviews)}", flush=True)

    return results

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Sentiment classification of Amazon reviews")
    parser.add_argument('--first', type=int, default=None, help='Score only first N reviews')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for balanced sampling')
    parser.add_argument('--balanced', action='store_true', help='Use balanced 3-class sampling')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: auto)')
    args = parser.parse_args()

    print("Loading reviews...")
    reviews = load_reviews()
    print(f"Loaded {len(reviews)} reviews")

    # Select reviews
    reviews = select_reviews(reviews, first_n=args.first, balanced=args.balanced, seed=args.seed)
    print(f"Selected {len(reviews)} reviews for scoring")

    # Determine mode
    use_neutral = args.balanced
    mode = "balanced_3class" if args.balanced else ("first_n" if args.first else "full")

    # Set output file
    if args.output:
        output_file = args.output
    elif args.balanced:
        output_file = f"balanced_3class_results_{args.seed}.jsonl"
    elif args.first:
        output_file = f"first_{args.first}_results.jsonl"
    else:
        output_file = "all_results.jsonl"

    # Run classification
    print(f"\nRunning sentiment classification ({mode})...")
    results = run_classification(reviews, use_neutral=use_neutral)

    # Save
    save_results(results, output_file)
    print(f"\nResults saved to {output_file}")

    # Print summary
    preds = [r['predicted_sentiment'] for r in results]
    from collections import Counter
    pred_counts = Counter(preds)
    print(f"\nPrediction distribution:")
    for label in ["POSITIVE", "NEUTRAL", "NEGATIVE", "ERROR"]:
        if label in pred_counts:
            print(f"  {label}: {pred_counts[label]}")

if __name__ == "__main__":
    main()

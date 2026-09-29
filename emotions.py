"""
Step 5: Primary-emotion detection for Amazon reviews.

Two independent approaches:
1. LLM-based emotion prediction (extends sentiment.py prompt)
2. NRC Word-Emotion Lexicon word-list approach

Usage:
    python emotions.py [--balanced] [--first N]
    Reads from the corresponding results JSONL file.
    Outputs enriched results with emotion fields to <input>_emotions.jsonl
"""

import argparse
import gzip
import json
import re
import string
from collections import Counter
from pathlib import Path

import sys

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

PROMPT_EMOTION = """Analyze the following Amazon review and identify its PRIMARY emotion. Choose exactly one from:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust

Review Title: {title}
Review Text: {text}

Rules:
- Consider what emotion the reviewer is expressing most strongly.
- Output ONLY the single emotion word: anger, anticipation, disgust, fear, joy, sadness, surprise, or trust.
- Do not include any explanation."""


def call_llm_emotion(title, text):
    """Call the LLM for emotion prediction."""
    from openai import OpenAI
    client = OpenAI(base_url="http://dobolyi.com:9001/v1", api_key="6418")
    model = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

    prompt = PROMPT_EMOTION.format(title=title, text=text)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        msg = resp.choices[0].message
        text_output = (msg.reasoning or msg.content or "").strip()

        # Parse from end of reasoning
        words = text_output.upper().split()
        for w in reversed(words):
            clean = w.rstrip(".,;:!'\"")
            if clean in [e.upper() for e in EMOTIONS]:
                return clean.lower()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

    return "unknown"


def load_nrc_lexicon(lexicon_path):
    """Load NRC Word-Emotion Lexicon. Returns dict: word -> {emotion: 0/1}."""
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                word, emotion, score = parts[0].lower(), parts[1].lower(), int(parts[2])
                if word not in lexicon:
                    lexicon[word] = {}
                lexicon[word][emotion] = score
    return lexicon


def extract_words(text):
    """Simple tokenization: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


def compute_emotion_scores(words, lexicon):
    """Score words against NRC lexicon. Returns dict: emotion -> count."""
    scores = {e: 0 for e in EMOTIONS}
    for word in words:
        if word in lexicon:
            for emotion, score in lexicon[word].items():
                if emotion in scores and score == 1:
                    scores[emotion] += 1
                # Skip sentiment columns (positive/negative) — they're not emotions
    return scores


def get_primary_emotion(scores):
    """Get highest-scoring emotion. Ties broken alphabetically."""
    max_score = max(scores.values())
    if max_score == 0:
        return "unknown"
    candidates = [e for e, s in scores.items() if s == max_score]
    candidates.sort()
    return candidates[0]


def load_reviews(input_file):
    """Load reviews from JSONL file."""
    reviews = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    return reviews


def run_llm_emotions(reviews, output_file, batch_size=10):
    """Add LLM-based emotion predictions."""
    print(f"Running LLM emotion detection on {len(reviews)} reviews...")
    for i, review in enumerate(reviews):
        title = review.get('title', '')
        text = review.get('text', '')

        llm_emotion = call_llm_emotion(title, text)
        review['llm_emotion'] = llm_emotion

        # Save progress
        if (i + 1) % batch_size == 0 or i == len(reviews) - 1:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in reviews:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"  Processed {i + 1}/{len(reviews)}", flush=True)


def run_nrc_emotions(reviews, lexicon_path, output_file, batch_size=10):
    """Add NRC lexicon-based emotion predictions."""
    print(f"Loading NRC lexicon...")
    lexicon = load_nrc_lexicon(lexicon_path)
    print(f"Loaded {len(lexicon)} words from lexicon")

    print(f"Running NRC emotion detection on {len(reviews)} reviews...")
    for i, review in enumerate(reviews):
        title = review.get('title', '')
        text = review.get('text', '')
        words = extract_words(title + ' ' + text)
        scores = compute_emotion_scores(words, lexicon)
        primary = get_primary_emotion(scores)

        review['nrc_emotion'] = primary
        review['nrc_emotion_scores'] = scores

        if (i + 1) % batch_size == 0 or i == len(reviews) - 1:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in reviews:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"  Processed {i + 1}/{len(reviews)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Emotion detection for Amazon reviews")
    parser.add_argument('--input', type=str, default=None, help='Input JSONL file')
    parser.add_argument('--method', type=str, default='both', choices=['llm', 'nrc', 'both'],
                        help='Detection method')
    parser.add_argument('--lexicon', type=str, default=None, help='NRC lexicon file')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    args = parser.parse_args()

    # Auto-detect input
    if args.input is None:
        # Try balanced first, then first_100
        if Path("balanced_3class_results.jsonl").exists():
            args.input = "balanced_3class_results.jsonl"
        elif Path("first_100_results.jsonl").exists():
            args.input = "first_100_results.jsonl"
        else:
            print("No input file found. Use --input to specify.")
            sys.exit(1)

    if args.output is None:
        base = Path(args.input).stem
        if args.method == 'both':
            args.output = f"{base}_emotions.jsonl"
        elif args.method == 'llm':
            args.output = f"{base}_llm_emotions.jsonl"
        else:
            args.output = f"{base}_nrc_emotions.jsonl"

    if args.lexicon is None:
        lexicon_path = "NRC-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    else:
        lexicon_path = args.lexicon

    reviews = load_reviews(args.input)
    print(f"Loaded {len(reviews)} reviews")

    if args.method in ('llm', 'both'):
        if args.method == 'both':
            # Run LLM first
            run_llm_emotions(reviews, args.output, batch_size=10)
        else:
            run_llm_emotions(reviews, args.output, batch_size=10)

    if args.method in ('nrc', 'both'):
        # Run NRC
        run_nrc_emotions(reviews, lexicon_path, args.output)

    print(f"\nResults saved to {args.output}")

    # Summary
    if Path(args.output).exists():
        enriched = load_reviews(args.output)
        llm_dist = Counter(r.get('llm_emotion', 'unknown') for r in enriched if 'llm_emotion' in r)
        nrc_dist = Counter(r.get('nrc_emotion', 'unknown') for r in enriched if 'nrc_emotion' in r)
        print("\nLLM emotion distribution:")
        for e in EMOTIONS:
            if e in llm_dist:
                print(f"  {e:12s}: {llm_dist[e]}")
        print("\nNRC emotion distribution:")
        for e in EMOTIONS:
            if e in nrc_dist:
                print(f"  {e:12s}: {nrc_dist[e]}")

        # Agreement
        if 'llm_emotion' in enriched[0] and 'nrc_emotion' in enriched[0]:
            agree = sum(1 for r in enriched if r['llm_emotion'] == r['nrc_emotion'])
            print(f"\nLLM/NRC agreement: {agree}/{len(enriched)} = {agree/len(enriched)*100:.1f}%")


if __name__ == "__main__":
    main()

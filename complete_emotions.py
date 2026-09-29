"""Complete LLM emotion detection on reviews 90-149."""
import json
import sys
from openai import OpenAI

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

client = OpenAI(base_url="http://dobolyi.com:9001/v1", api_key="6418")
model = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

PROMPT = """Analyze the following Amazon review and identify its PRIMARY emotion. Choose exactly one from:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust

Review Title: {title}
Review Text: {text}

Rules:
- Consider what emotion the reviewer is expressing most strongly.
- Output ONLY the single emotion word: anger, anticipation, disgust, fear, joy, sadness, surprise, or trust.
- Do not include any explanation."""

# Load existing results
reviews = []
with open("balanced_3class_emotions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        reviews.append(json.loads(line))

# Find first missing
start_idx = None
for i, r in enumerate(reviews):
    if 'llm_emotion' not in r:
        start_idx = i
        break

if start_idx is None:
    print("All reviews already have LLM emotions.")
    sys.exit(0)

print(f"Starting from review {start_idx} out of {len(reviews)}")

for i in range(start_idx, len(reviews)):
    review = reviews[i]
    title = review.get('title', '')
    text = review.get('text', '')
    
    prompt_text = PROMPT.format(title=title, text=text)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=2000,
        )
        msg = resp.choices[0].message
        text_out = (msg.reasoning or msg.content or "").strip()
        
        words = text_out.upper().split()
        for w in reversed(words):
            clean = w.rstrip(".,;:!'\"")
            if clean in [e.upper() for e in EMOTIONS]:
                review['llm_emotion'] = clean.lower()
                break
        else:
            review['llm_emotion'] = 'unknown'
    except Exception as e:
        print(f"Error on review {i}: {e}")
        review['llm_emotion'] = 'ERROR'

    # Save every 10 reviews
    if (i + 1 - start_idx) % 10 == 0:
        with open("balanced_3class_emotions.jsonl", "w", encoding="utf-8") as f:
            for r in reviews:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"  Processed {i + 1}/{len(reviews)}", flush=True)

# Save final
with open("balanced_3class_emotions.jsonl", "w", encoding="utf-8") as f:
    for r in reviews:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\nDone! Saved to balanced_3class_emotions.jsonl")

# Summary
from collections import Counter
llm_dist = Counter(r.get('llm_emotion', 'unknown') for r in reviews)
nrc_dist = Counter(r.get('nrc_emotion', 'unknown') for r in reviews)
print("\nLLM emotion distribution:")
for e in EMOTIONS:
    if e in llm_dist:
        print(f"  {e:12s}: {llm_dist[e]}")

agree = sum(1 for r in reviews if r.get('llm_emotion') == r.get('nrc_emotion'))
print(f"\nLLM/NRC agreement: {agree}/{len(reviews)} = {agree/len(reviews)*100:.1f}%")

import json

# Load all data files to extract dashboard data
step2 = []
with open("first_100_results.jsonl") as f:
    for line in f:
        step2.append(json.loads(line))

step6 = []
with open("balanced_3class_results.jsonl") as f:
    for line in f:
        step6.append(json.loads(line))

step6_emotions = []
with open("balanced_3class_emotions.jsonl") as f:
    for line in f:
        step6_emotions.append(json.loads(line))

import gzip
rating_dist = {}
with gzip.open("Gift_Cards.jsonl.gz", 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        rat = r["rating"]
        rating_dist[rat] = rating_dist.get(rat, 0) + 1

def get_label_2class(rating):
    return "POSITIVE" if rating >= 4 else "NEGATIVE"

def get_label_3class(rating):
    if rating >= 4: return "POSITIVE"
    if rating == 3: return "NEUTRAL"
    return "NEGATIVE"

# Step 2 confusion
tp = sum(1 for r in step2 if r["predicted_sentiment"] == "POSITIVE" and get_label_2class(r["rating"]) == "POSITIVE")
tn = sum(1 for r in step2 if r["predicted_sentiment"] == "NEGATIVE" and get_label_2class(r["rating"]) == "NEGATIVE")
fp = sum(1 for r in step2 if r["predicted_sentiment"] == "POSITIVE" and get_label_2class(r["rating"]) == "NEGATIVE")
fn = sum(1 for r in step2 if r["predicted_sentiment"] == "NEGATIVE" and get_label_2class(r["rating"]) == "POSITIVE")
total2 = len(step2)
acc2 = (tp + tn) / total2 * 100

pos_total2 = sum(1 for r in step2 if get_label_2class(r["rating"]) == "POSITIVE")
neg_total2 = sum(1 for r in step2 if get_label_2class(r["rating"]) == "NEGATIVE")
pos_correct2 = sum(1 for r in step2 if get_label_2class(r["rating"]) == "POSITIVE" and r["predicted_sentiment"] == "POSITIVE")
neg_correct2 = sum(1 for r in step2 if get_label_2class(r["rating"]) == "NEGATIVE" and r["predicted_sentiment"] == "NEGATIVE")

# Step 6 confusion
from collections import defaultdict
conf6 = defaultdict(int)
tp3 = sum(1 for r in step6 if r["predicted_sentiment"] == "POSITIVE" and get_label_3class(r["rating"]) == "POSITIVE")
tn3 = sum(1 for r in step6 if get_label_3class(r["rating"]) == "NEUTRAL" and r["predicted_sentiment"] == "NEUTRAL")
tf3 = sum(1 for r in step6 if r["predicted_sentiment"] == "NEGATIVE" and get_label_3class(r["rating"]) == "NEGATIVE")
total3 = len(step6)
acc3 = (tp3 + tn3 + tf3) / total3 * 100

for r in step6:
    true = get_label_3class(r["rating"])
    pred = r["predicted_sentiment"]
    conf6[(true, pred)] += 1

pos3_total = sum(1 for r in step6 if get_label_3class(r["rating"]) == "POSITIVE")
neg3_total = sum(1 for r in step6 if get_label_3class(r["rating"]) == "NEGATIVE")
neu3_total = sum(1 for r in step6 if get_label_3class(r["rating"]) == "NEUTRAL")

# Step 2 error reviews
step2_errors = [r for r in step2 if r["predicted_sentiment"] != get_label_2class(r["rating"])]

# Step 6 error reviews
step6_errors = [r for r in step6 if r["predicted_sentiment"] != get_label_3class(r["rating"])]

# Emotion stats
llm_emotions = [r["llm_emotion"] for r in step6_emotions if "llm_emotion" in r]
nrc_emotions = [r["nrc_emotion"] for r in step6_emotions if "nrc_emotion" in r]
agree = sum(1 for i in range(len(step6_emotions)) 
            if "llm_emotion" in step6_emotions[i] and "nrc_emotion" in step6_emotions[i] 
            and step6_emotions[i]["llm_emotion"] == step6_emotions[i]["nrc_emotion"])

from collections import Counter
llm_dist = Counter(llm_emotions)
nrc_dist = Counter(nrc_emotions)

# Emotion by sentiment
emotion_by_sentiment = defaultdict(lambda: Counter())
for r in step6_emotions:
    true_sent = get_label_3class(r["rating"])
    if "llm_emotion" in r:
        emotion_by_sentiment[true_sent][r["llm_emotion"]] += 1

# Star rating distribution (raw counts and percentages)
stars = [(f"★{'★'*int(s-1)}{'☆'*(5-int(s))} ({s:.1f})", rating_dist.get(s, 0)) for s in [1.0, 2.0, 3.0, 4.0, 5.0]]
total_reviews = sum(rating_dist.values())
stars_data = [(label, count, count/total_reviews*100) for label, count in stars]

# Build the per-review detail rows for Step 6 (balanced sample)
review_rows = []
for i, r in enumerate(step6_emotions):
    true_label = get_label_3class(r["rating"])
    pred = r["predicted_sentiment"]
    match = pred == true_label
    review_rows.append({
        "index": i,
        "rating": r["rating"],
        "true_sentiment": true_label,
        "predicted_sentiment": pred,
        "correct": match,
        "llm_emotion": r.get("llm_emotion", "N/A"),
        "nrc_emotion": r.get("nrc_emotion", "N/A"),
        "emotions_agree": r.get("llm_emotion", "") == r.get("nrc_emotion", ""),
        "title": r["title"],
        "text": r["text"],
    })

# Save all data as a single JSON file for the dashboard
data = {
    "step2": {
        "total": total2,
        "accuracy": round(acc2, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pos_total": pos_total2, "neg_total": neg_total2,
        "pos_correct": pos_correct2, "neg_correct": neg_correct2,
    },
    "step6": {
        "total": total3,
        "accuracy": round(acc3, 1),
        "tp": tp3, "tn": tn3, "tf": tf3,
        "pos_total": pos3_total, "neg_total": neg3_total, "neu_total": neu3_total,
        "confusion": {
            "PP": conf6.get(("POSITIVE", "POSITIVE"), 0),
            "PN": conf6.get(("POSITIVE", "NEGATIVE"), 0),
            "PT": conf6.get(("POSITIVE", "NEUTRAL"), 0),
            "NP": conf6.get(("NEGATIVE", "POSITIVE"), 0),
            "NN": conf6.get(("NEGATIVE", "NEGATIVE"), 0),
            "NT": conf6.get(("NEGATIVE", "NEUTRAL"), 0),
            "TP": conf6.get(("NEUTRAL", "POSITIVE"), 0),
            "TN": conf6.get(("NEUTRAL", "NEGATIVE"), 0),
            "TT": conf6.get(("NEUTRAL", "NEUTRAL"), 0),
        },
        "per_class_accuracy": {
            "POSITIVE": round(tp3 / pos3_total * 100, 1) if pos3_total else 0,
            "NEUTRAL": round(tn3 / neu3_total * 100, 1) if neu3_total else 0,
            "NEGATIVE": round(tf3 / neg3_total * 100, 1) if neg3_total else 0,
        }
    },
    "stars": stars_data,
    "emotion_llm": {e: llm_dist[e] for e in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]},
    "emotion_nrc": {e: nrc_dist[e] for e in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]},
    "emotion_agreement": {
        "agree": agree,
        "total": len(step6_emotions),
        "pct": round(agree / len(step6_emotions) * 100, 1),
    },
    "emotion_by_sentiment": {
        "POSITIVE": {e: emotion_by_sentiment["POSITIVE"].get(e, 0) for e in llm_dist},
        "NEGATIVE": {e: emotion_by_sentiment["NEGATIVE"].get(e, 0) for e in llm_dist},
        "NEUTRAL": {e: emotion_by_sentiment["NEUTRAL"].get(e, 0) for e in llm_dist},
    },
    "review_rows": review_rows,
}

# Write data
with open("dashboard_data.json", "w") as f:
    json.dump(data, f)

print("Dashboard data saved: dashboard_data.json")
print(f"Step 2: {total2} reviews, {acc2:.1f}% accuracy (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
print(f"Step 6: {total3} reviews, {acc3:.1f}% accuracy")
print(f"Emotions: {agree}/{len(step6_emotions)} agreement ({agree/len(step6_emotions)*100:.1f}%)")

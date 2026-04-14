import re

import numpy as np
from atproto import Client
from flask import Flask, render_template, request
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

NUM_WORDS = 10000
MAX_LEN = 200

# Load model and word index once at startup
model = load_model("model.keras")
word_index = imdb.get_word_index()


def encode(texts: list[str]) -> np.ndarray:
    sequences = [
        [
            idx if (idx := word_index.get(w.lower(), 2) + 3) < NUM_WORDS else 2
            for w in text.split()
        ]
        for text in texts
    ]
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="pre")


def fetch_posts(url: str) -> list[str]:
    client = Client(base_url="https://public.api.bsky.app")

    post_match = re.match(r"https://bsky\.app/profile/([^/]+)/post/([^/]+)", url)
    profile_match = re.match(r"https://bsky\.app/profile/([^/?]+)$", url)

    if post_match:
        handle, rkey = post_match.groups()
        did = client.resolve_handle(handle).did
        at_uri = f"at://{did}/app.bsky.feed.post/{rkey}"
        thread = client.app.bsky.feed.get_post_thread({"uri": at_uri})
        return [thread.thread.post.record.text]
    elif profile_match:
        handle = profile_match.group(1)
        feed = client.app.bsky.feed.get_author_feed({"actor": handle, "limit": 10})
        return [item.post.record.text for item in feed.feed]
    else:
        raise ValueError("URL must be a Bluesky post or profile URL")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    url = request.form.get("url", "").strip()
    if not url:
        return render_template("index.html", error="Please enter a URL.")

    try:
        posts = fetch_posts(url)
    except ValueError as e:
        return render_template("index.html", url=url, error=str(e))
    except Exception:
        return render_template("index.html", url=url, error="Could not fetch posts. Check the URL and try again.")

    scores = model.predict(encode(posts), verbose=0)[:, 0]
    results = [
        {"text": text, "score": float(score), "label": "Positive" if score >= 0.5 else "Negative"}
        for text, score in zip(posts, scores)
    ]

    return render_template("index.html", url=url, results=results)


if __name__ == "__main__":
    app.run(debug=True)

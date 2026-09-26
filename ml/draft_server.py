"""Local server behind the web app's #/review page. Keeps SAM loaded so fixes are instant.

  GET  /api/queue            drafts still waiting for a decision + progress counts
  GET  /img/<sha>            cached photo (long side 1024)
  GET  /draft/<sha>          current draft label PNG (0 bg, 1 head, 2 thorax, 3 abdomen)
  POST /api/approve {sha}    draft → ml/data/masks/<sha>.png   (this is what becomes ground truth)
  POST /api/reject  {sha}    appended to ml/data/rejected.txt
  POST /api/refine  {sha, part, points: [[x, y, 1|0], ...]}   re-prompt SAM for one part
  POST /api/flip    {sha}    re-draft with every fly's head/abdomen ends swapped (toggles)

Binds to 127.0.0.1 only. usage: ml/.venv/bin/python ml/draft_server.py
"""
import csv
import shutil
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from sam_draft import DATA, Sam, draft

app = Flask(__name__)
sam = Sam()
MASKS, DRAFTS, REJECTED = DATA / "masks", DATA / "drafts", DATA / "rejected.txt"
MASKS.mkdir(exist_ok=True)
FLIPPED: set[str] = set()  # images whose auto head/abdomen guess was reversed by the reviewer
SESSION = {r["sha1"]: r["session"] for r in csv.DictReader(open(DATA / "manifest.csv"))}


def valid(sha: str) -> str:
    if sha not in SESSION:  # also stops path tricks like "../../"
        abort(400, "unknown image")
    return sha


def rejected() -> set[str]:
    return set(REJECTED.read_text().split()) if REJECTED.exists() else set()


@lru_cache(maxsize=4)
def embedding(sha: str):
    return sam.embed(cv2.cvtColor(cv2.imread(str(DATA / "img" / f"{sha}.jpg")), cv2.COLOR_BGR2RGB))


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/api/queue")
def queue():
    done = {p.stem for p in MASKS.glob("*.png")} | rejected()
    todo = sorted(p.stem for p in DRAFTS.glob("*.png") if p.stem not in done)
    return jsonify(queue=[{"sha": s, "session": SESSION.get(s, "")} for s in todo],
                   approved=len(list(MASKS.glob("*.png"))), rejected=len(rejected()))


@app.get("/img/<sha>")
def img(sha):
    return send_file(DATA / "img" / f"{valid(sha)}.jpg", mimetype="image/jpeg")


@app.get("/draft/<sha>")
def get_draft(sha):
    return send_file(DRAFTS / f"{valid(sha)}.png", mimetype="image/png", max_age=0)


@app.post("/api/approve")
def approve():
    sha = valid(request.json["sha"])
    shutil.copy(DRAFTS / f"{sha}.png", MASKS / f"{sha}.png")
    return jsonify(ok=True)


@app.post("/api/reject")
def reject():
    sha = valid(request.json["sha"])
    with open(REJECTED, "a") as f:
        f.write(sha + "\n")
    return jsonify(ok=True)


@app.post("/api/refine")
def refine():
    body = request.json
    sha, part, pts = valid(body["sha"]), int(body["part"]), body["points"]
    if part not in (1, 2, 3) or not pts:
        return jsonify(error="need part 1-3 and at least one point"), 400
    emb, inp = embedding(sha)
    masks, scores = sam.masks(emb, inp, [[(p[0], p[1]) for p in pts]], [[int(p[2]) for p in pts]])
    new = masks[0, int(np.argmax(scores[0]))]
    labels = cv2.imread(str(DRAFTS / f"{sha}.png"), cv2.IMREAD_GRAYSCALE)
    labels[labels == part] = 0
    labels[new] = part
    cv2.imwrite(str(DRAFTS / f"{sha}.png"), labels)
    return jsonify(ok=True)


@app.post("/api/flip")
def flip():
    sha = valid(request.json["sha"])
    FLIPPED.symmetric_difference_update({sha})  # toggle, so pressing X twice undoes it
    cv2.imwrite(str(DRAFTS / f"{sha}.png"), draft(sam, cv2.imread(str(DATA / "img" / f"{sha}.jpg")), flip=sha in FLIPPED))
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055)

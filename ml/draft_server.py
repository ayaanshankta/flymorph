"""Local server behind the web app's #/review page. Keeps SAM loaded so fixes are instant.

Review happens ONE FLY AT A TIME: each photo is split into fly boxes (from the same detector the drafter
uses), and every edit only changes pixels inside the box it was made in, so fixing one fly can never touch
its neighbour. A photo becomes ground truth (ml/data/masks) once every fly in it is approved; rejecting
any fly rejects the photo (a half-labelled photo would teach the model that the other fly is background).

  GET  /api/queue             flies still waiting: {sha, fly, flies, box:[x0,y0,x1,y1], session} + counts
  GET  /img/<sha>             cached photo (long side 1024)
  GET  /draft/<sha>           current draft label PNG (0 bg, 1 head, 2 thorax, 3 abdomen)
  POST /api/approve {sha, fly}
  POST /api/reject  {sha}
  POST /api/refine  {sha, box, part, points: [[x, y, 1|0], ...], pick}   re-prompt SAM for one part;
                    pick 0/1/2 = SAM's smallest / middle / largest candidate outline
  POST /api/paint   {sha, box, part, polygon: [[x, y], ...]}   fill a drawn outline with a part (0 = erase)
  POST /api/flip    {sha, box}   re-draft this fly with its head/abdomen ends swapped (toggles)
  POST /api/reset   {sha, box}   re-draft this fly from scratch
  POST /api/undo    {sha}        undo the last edit on this photo

Binds to 127.0.0.1 and only answers the FlyMorph web app. usage: ml/.venv/bin/python ml/draft_server.py
"""
import csv
import json
import shutil
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from sam_draft import DATA, Sam, draft, fly_blobs

app = Flask(__name__)
sam = Sam()
MASKS, DRAFTS, REJECTED, FLY_OK = DATA / "masks", DATA / "drafts", DATA / "rejected.txt", DATA / "fly_ok.json"
MASKS.mkdir(exist_ok=True)
APP_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
HISTORY: dict[str, list[tuple[bytes, frozenset]]] = {}  # sha → earlier (draft PNG, flipped boxes), newest last
FLIPPED: dict[str, set[tuple]] = {}  # sha → boxes whose head/abdomen guess the reviewer reversed
SESSION = {r["sha1"]: r["session"] for r in csv.DictReader(open(DATA / "manifest.csv"))}


def valid(sha: str) -> str:
    if sha not in SESSION:  # also stops path tricks like "../../"
        abort(400, "unknown image")
    return sha


@lru_cache(maxsize=None)
def fly_boxes(sha: str) -> list[tuple[int, int, int, int]]:
    """One padded box per fly, left to right. Photos with no detected fly get one full-image box."""
    bgr = cv2.imread(str(DATA / "img" / f"{sha}.jpg"))
    h, w = bgr.shape[:2]
    boxes = []
    for blob in fly_blobs(bgr):
        ys, xs = np.nonzero(blob)
        pad = int(0.25 * max(np.ptp(xs), np.ptp(ys)))
        boxes.append((int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)),
                      int(min(w, xs.max() + pad + 1)), int(min(h, ys.max() + pad + 1))))
    return sorted(boxes) or [(0, 0, w, h)]


def box_of(body) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in body["box"])
    return x0, y0, x1, y1


def labels_of(sha: str) -> np.ndarray:
    return cv2.imread(str(DRAFTS / f"{sha}.png"), cv2.IMREAD_GRAYSCALE)


def save(sha: str, labels: np.ndarray) -> None:
    cv2.imwrite(str(DRAFTS / f"{sha}.png"), labels)


def remember(sha: str) -> None:
    """Save the current draft before changing it, so Undo can bring it back. Keeps the last 20."""
    HISTORY.setdefault(sha, []).append(((DRAFTS / f"{sha}.png").read_bytes(), frozenset(FLIPPED.get(sha, set()))))
    del HISTORY[sha][:-20]


def rejected() -> set[str]:
    return set(REJECTED.read_text().split()) if REJECTED.exists() else set()


def fly_ok() -> dict[str, list[int]]:
    return json.loads(FLY_OK.read_text()) if FLY_OK.exists() else {}


@lru_cache(maxsize=4)
def embedding(sha: str):
    return sam.embed(cv2.cvtColor(cv2.imread(str(DATA / "img" / f"{sha}.jpg")), cv2.COLOR_BGR2RGB))


def redraft_box(sha: str, box, flip: bool) -> None:
    """Re-run the automatic draft, but only copy the result inside this fly's box."""
    x0, y0, x1, y1 = box
    fresh = draft(sam, cv2.imread(str(DATA / "img" / f"{sha}.jpg")), flip=flip)
    labels = labels_of(sha)
    labels[y0:y1, x0:x1] = fresh[y0:y1, x0:x1]
    save(sha, labels)


@app.after_request
def cors(resp):
    if request.headers.get("Origin") in APP_ORIGINS:  # other websites open in the browser get no access
        resp.headers["Access-Control-Allow-Origin"] = request.headers["Origin"]
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/api/queue")
def queue():
    done, ok = {p.stem for p in MASKS.glob("*.png")} | rejected(), fly_ok()
    items = []
    for sha in sorted(p.stem for p in DRAFTS.glob("*.png") if p.stem not in done):
        boxes = fly_boxes(sha)
        items += [{"sha": sha, "fly": i, "flies": len(boxes), "box": list(b), "session": SESSION.get(sha, "")}
                  for i, b in enumerate(boxes) if i not in ok.get(sha, [])]
    return jsonify(queue=items, approved=len(list(MASKS.glob("*.png"))), rejected=len(rejected()))


@app.get("/img/<sha>")
def img(sha):
    return send_file(DATA / "img" / f"{valid(sha)}.jpg", mimetype="image/jpeg")


@app.get("/draft/<sha>")
def get_draft(sha):
    return send_file(DRAFTS / f"{valid(sha)}.png", mimetype="image/png", max_age=0)


@app.post("/api/approve")
def approve():
    sha, fly = valid(request.json["sha"]), int(request.json["fly"])
    ok = fly_ok()
    ok[sha] = sorted(set(ok.get(sha, [])) | {fly})
    FLY_OK.write_text(json.dumps(ok))
    photo_done = len(ok[sha]) >= len(fly_boxes(sha))
    if photo_done:
        shutil.copy(DRAFTS / f"{sha}.png", MASKS / f"{sha}.png")
    return jsonify(ok=True, photo_done=photo_done)


@app.post("/api/reject")
def reject():
    sha = valid(request.json["sha"])
    with open(REJECTED, "a") as f:
        f.write(sha + "\n")
    return jsonify(ok=True)


@app.post("/api/refine")
def refine():
    body = request.json
    sha, (x0, y0, x1, y1), part, pts = valid(body["sha"]), box_of(body), int(body["part"]), body["points"]
    if part not in (1, 2, 3) or not pts:
        return jsonify(error="need part 1-3 and at least one point"), 400
    labels = labels_of(sha)
    inside = np.zeros_like(labels, bool)
    inside[y0:y1, x0:x1] = True
    # SAM's favourite answer to one click is often the whole fly. Tell it where this fly's other parts are
    # (their centres as negative points). SAM returns three nested outlines (e.g. eye / head / fly); the
    # reviewer steps through them with Smaller / Bigger.
    points, labs = [(p[0], p[1]) for p in pts], [int(p[2]) for p in pts]
    for other in {1, 2, 3} - {part}:
        ys, xs = np.nonzero((labels == other) & inside)
        if len(xs):
            points.append((float(xs.mean()), float(ys.mean())))
            labs.append(0)
    emb, inp = embedding(sha)
    masks, _ = sam.masks(emb, inp, [points], [labs])
    by_size = sorted(range(3), key=lambda j: masks[0, j].sum())
    new = masks[0, by_size[min(max(int(body.get("pick", 0)), 0), 2)]] & inside
    remember(sha)
    labels[(labels == part) & inside] = 0
    labels[new] = part
    save(sha, labels)
    return jsonify(ok=True)


@app.post("/api/paint")
def paint():
    body = request.json
    sha, (x0, y0, x1, y1), part, poly = valid(body["sha"]), box_of(body), int(body["part"]), body["polygon"]
    if part not in (0, 1, 2, 3) or len(poly) < 3:
        return jsonify(error="need part 0-3 and at least 3 corners"), 400
    labels = labels_of(sha)
    region = np.zeros_like(labels)
    cv2.fillPoly(region, [np.round(np.array(poly)).astype(np.int32)], 1)
    keep = np.zeros_like(region)
    keep[y0:y1, x0:x1] = region[y0:y1, x0:x1]  # never paint outside this fly's box
    remember(sha)
    labels[keep.astype(bool)] = part
    save(sha, labels)
    return jsonify(ok=True)


@app.post("/api/flip")
def flip():
    sha, box = valid(request.json["sha"]), box_of(request.json)
    remember(sha)
    FLIPPED.setdefault(sha, set()).symmetric_difference_update({box})  # toggle, so pressing X twice undoes it
    redraft_box(sha, box, flip=box in FLIPPED[sha])
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    sha, box = valid(request.json["sha"]), box_of(request.json)
    remember(sha)
    redraft_box(sha, box, flip=box in FLIPPED.get(sha, set()))
    return jsonify(ok=True)


@app.post("/api/undo")
def undo():
    sha = valid(request.json["sha"])
    if not HISTORY.get(sha):
        return jsonify(error="nothing to undo"), 400
    png, flipped = HISTORY[sha].pop()
    (DRAFTS / f"{sha}.png").write_bytes(png)
    FLIPPED[sha] = set(flipped)
    return jsonify(ok=True, left=len(HISTORY[sha]))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055)

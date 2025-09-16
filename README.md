# Face Match (FaceNet + MTCNN) — PySide6 Desktop Demo
A small desktop app that detects a face, extracts a FaceNet embedding, and matches it against a local gallery using cosine similarity. Built as a clean, interactive portfolio piece.

---

## Features
- Face detection & crop using **MTCNN**
- Embeddings via **keras-facenet (FaceNet)** with **L2 normalization**
- **Top-K** matches with a **similarity threshold** slider
- **First sample**: enter a name → “Add as New”
- **After that**: double-click a known match or press **Confirm match**
- Stores samples per person in `faces_db/<name>/` and metadata in `embeddings.json`
- Handles PNG/JPEG (alpha-safe), EXIF rotation, and correct Qt image buffering

---

## Repo Structure

```bash
.
├─ app_faces.py          # PySide6 app (UI + logic)
├─ requirements.txt
├─ README.md
├─ faces_db/             # created at runtime (per-person samples)
└─ test_images/          # your 9 actress images (example set, optional)
```

--- 

## Requirements

Tested on Python 3.10 (Windows, CPU).

```bash
PySide6>=6.6
numpy<2.0
Pillow>=10.0
scikit-learn>=1.2
keras-facenet==0.3.2
mtcnn==0.1.1
tensorflow==2.10.1
protobuf<4.0
```

If you prefer TF 2.15 (or GPU): use tensorflow==2.15.* and remove the numpy<2.0 pin.
Apple Silicon: tensorflow-macos==2.15.* (+ optional tensorflow-metal).

---

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

--- 

## Run

```bash
python app_faces.py
```

### Worklfow
1. Click Open Image and pick a photo.
2. If the database is empty: tick Add New, type the person’s name, click Add as New.
3. Later images of known people: double-click a match in Top-K Matches or press Confirm match.
4. Adjust the threshold to trade off recall vs. false positives (default: 0.70).

--- 

## How It Works
1. Detect & crop the single face with MTCNN (0 or >1 faces ⇒ warning and skip).
2. Resize to 160×160 and compute FaceNet embedding.
3. L2-normalize the embedding to stabilize cosine similarity.
4. Match against all stored embeddings; sort by similarity; show Top-K.
5. Decision: highlight scores ≥ threshold (✅). You can still override below threshold.

---

## Privacy & Ethics
- Only use images you have rights/consent to use.
- Store the minimum necessary data; provide a simple “clear DB” (you can delete faces_db/ and embeddings.json).
- Be aware of bias/limitations of face models; document observed failure modes.

---

## Acknowledgements
- FaceNet via keras-facenet
- MTCNN for detection
- PySide6 / Qt for the UI
- Pillow, NumPy, scikit-learn for image/ML utilities

---

## Future Work

### Accuracy & Robustness
- Better detection: switch MTCNN → RetinaFace (InsightFace) for sharper crops & hard cases.
- Quality filter: reject blurry/low-light faces (variance/Laplacian score, occlusion checks) before embedding.
- Hard negative mining: deliberately add visually similar impostors to stress the threshold.
- Stronger embeddings: try ArcFace/Glint360K models (InsightFace) vs. keras-facenet; compare ROC/PR.

### Evaluation & Calibration
- Eval script: log genuine vs. impostor similarities; plot ROC/PR, compute EER, pick threshold by target precision.
- k-fold holdout per identity (once you have more samples) to avoid optimistic scores.

### Speed & Scale
- Indexing: switch cosine search to FAISS (IndexFlatIP) or Annoy for >10k samples.
- Batch embedder warm-up and lazy model loading at app start (smoother UX).

### UX & Workflow
- Drag & drop images onto the window.
- Batch add: process a folder; auto-group by top match, then review in a queue.
- Confirm match improvements: side-by-side (current crop vs. gallery sample).
- Manage identities: rename/delete person, move samples between identities.
- Keyboard-first flow: Enter = confirm top-1; N = toggle “Add New”; ↑/↓ navigate matches.

### Data & Storage
- Switch JSON → SQLite (embeddings, persons, images tables). Add migrations and backup.
- Export/Import gallery (zip of faces_db + DB dump) for portability.
- Deduping: near-duplicate detection (cosine > 0.98 within same identity).

### Privacy, Security, Compliance
- Optional encryption at rest for embeddings (SQLite + keyring).
- Clear DB button; per-identity purge.

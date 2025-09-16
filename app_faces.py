import os, json, uuid, sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QListWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QLineEdit, QMessageBox,
    QGroupBox, QCheckBox
)
from PySide6.QtGui import QPixmap, QImage

# ML imports (lazy singletons inside ModelHub)
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

# -------------------- Config --------------------
EMBEDDINGS_PATH = "embeddings.json"
IMG_SIZE = (160, 160)
TOP_K = 10
DEFAULT_THRESHOLD = 0.70  # cosine similarity; calibrate on your own data

# -------------------- Utils --------------------
def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def pil_to_qpix(img: Image.Image) -> QPixmap:
    img = img.convert("RGB")
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# -------------------- Data classes --------------------
@dataclass
class Entry:
    id: str
    name: str
    image_path: str
    embedding: List[float]
    model: str = "keras_facenet"
    version: str = "1.0"

# -------------------- FaceDB --------------------
class FaceDB:
    """Simple JSON-backed DB with in-RAM NumPy matrix cache."""
    def __init__(self, path: str):
        self.path = path
        self.entries: List[Entry] = self._load()
        self._X_cache: Optional[np.ndarray] = None

    def _load(self) -> List[Entry]:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                raw = json.load(f)
            return [Entry(**item) for item in raw]
        return []

    def save(self):
        with open(self.path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def add(self, name: str, image_path: str, embedding: np.ndarray):
        self.entries.append(Entry(
            id=str(uuid.uuid4()),
            name=name,
            image_path=image_path,
            embedding=embedding.astype(np.float32).tolist()
        ))
        self._X_cache = None
        self.save()

    def as_matrix(self) -> Tuple[np.ndarray, List[Entry]]:
        if self._X_cache is None:
            if not self.entries:
                self._X_cache = np.empty((0, 512), dtype=np.float32)
            else:
                self._X_cache = np.array([e.embedding for e in self.entries], dtype=np.float32)
        return self._X_cache, self.entries

# -------------------- ModelHub (singletons) --------------------
class ModelHub:
    _facenet = None
    _mtcnn = None

    @classmethod
    def facenet(cls):
        if cls._facenet is None:
            cls._facenet = FaceNet()
        return cls._facenet

    @classmethod
    def mtcnn(cls):
        if cls._mtcnn is None:
            cls._mtcnn = MTCNN()
        return cls._mtcnn

# -------------------- Worker (detect + embed) --------------------
class FaceWorker(QThread):
    result_ready = Signal(object)  # dict: {ok, error?, pixmap?, face_pix?, embedding?}

    def __init__(self, img_path: str):
        super().__init__()
        self.img_path = img_path

    def detect_and_align(self, img: Image.Image) -> Optional[Image.Image]:
        detector = ModelHub.mtcnn()
        arr = np.array(img.convert("RGB"))
        dets = detector.detect_faces(arr)
        if len(dets) != 1:
            return None
        x, y, w, h = dets[0]["box"]
        x, y = max(0, x), max(0, y)
        face = arr[y:y+h, x:x+w]
        face_img = Image.fromarray(face).resize(IMG_SIZE)
        return face_img

    def embed(self, face_img: Image.Image) -> np.ndarray:
        emb = ModelHub.facenet().embeddings([np.array(face_img)])[0]
        return l2_normalize(emb)

    def run(self):
        try:
            img = Image.open(self.img_path)
            face_img = self.detect_and_align(img)
            if face_img is None:
                self.result_ready.emit({"ok": False, "error": "No or multiple faces detected."})
                return
            emb = self.embed(face_img)
            self.result_ready.emit({
                "ok": True,
                "pixmap": pil_to_qpix(img),
                "face_pix": pil_to_qpix(face_img),
                "embedding": emb
            })
        except Exception as e:
            self.result_ready.emit({"ok": False, "error": str(e)})

# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Match (FaceNet + MTCNN)")
        self.resize(1000, 650)

        # State
        self.db = FaceDB(EMBEDDINGS_PATH)
        self.current_img_path: Optional[str] = None
        self.current_embedding: Optional[np.ndarray] = None
        self.current_matches: List[Tuple[Entry, float]] = []

        # --- Left panel (images) ---
        self.img_label = QLabel("Image")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(QSize(400, 300))

        self.face_label = QLabel("Detected face")
        self.face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_label.setFixedHeight(180)

        self.btn_open = QPushButton("Open Image")
        self.btn_open.clicked.connect(self.open_image)

        left = QVBoxLayout()
        left.addWidget(self.img_label)
        face_group = QGroupBox("Detected face")
        face_l = QVBoxLayout()
        face_l.addWidget(self.face_label)
        face_group.setLayout(face_l)
        left.addWidget(face_group)
        left.addWidget(self.btn_open)

        # --- Right panel (controls) ---
        self.matches_list = QListWidget()
        self.matches_list.itemSelectionChanged.connect(self._on_match_selection)
        self.matches_list.itemDoubleClicked.connect(lambda _: self.confirm_match())

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(50)   # 0.50
        self.slider.setMaximum(95)   # 0.95
        self.slider.setValue(int(DEFAULT_THRESHOLD * 100))
        self.slider.valueChanged.connect(self.refresh_matches)

        self.threshold_label = QLabel(f"Threshold: {DEFAULT_THRESHOLD:.2f}")

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Similarity threshold"))
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.threshold_label)

        top_k_box = QGroupBox("Top-K Matches")
        top_k_l = QVBoxLayout()
        top_k_l.addWidget(self.matches_list)
        top_k_box.setLayout(top_k_l)

        # Add-new controls
        self.chk_new = QCheckBox("Add New")
        self.chk_new.stateChanged.connect(self._toggle_new_mode)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name for new person...")
        self.name_input.setEnabled(False)

        self.btn_add = QPushButton("Add as New")
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self.add_as_new)

        self.btn_confirm = QPushButton("Confirm match")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self.confirm_match)

        new_box = QHBoxLayout()
        new_box.addWidget(self.chk_new)
        new_box.addWidget(self.name_input, 1)
        new_box.addWidget(self.btn_add)
        new_box.addWidget(self.btn_confirm)

        right = QVBoxLayout()
        right.addLayout(ctrl)
        right.addWidget(top_k_box, 1)
        right.addLayout(new_box)

        # --- Root layout ---
        root = QHBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.worker: Optional[FaceWorker] = None
        self._update_threshold_label()

    # -------------------- Modes --------------------
    def _set_mode_unknown(self):
        """DB empty → require new name."""
        self.chk_new.setChecked(True)
        self.chk_new.setEnabled(False)  # force 'new' on first sample
        self.name_input.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_confirm.setEnabled(False)

    def _set_mode_known(self):
        """DB has entries → default to matching mode."""
        self.chk_new.setEnabled(True)
        self.chk_new.setChecked(False)
        self.name_input.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_confirm.setEnabled(False)

    def _toggle_new_mode(self):
        is_new = self.chk_new.isChecked()
        self.name_input.setEnabled(is_new)
        self.btn_add.setEnabled(is_new)
        self.btn_confirm.setEnabled(not is_new)

    # -------------------- Actions --------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.current_img_path = path
        self.current_embedding = None
        self.current_matches = []
        self.matches_list.clear()
        self.name_input.clear()
        self.img_label.setText("Processing...")
        self.face_label.setText("Processing...")

        self.worker = FaceWorker(path)
        self.worker.result_ready.connect(self.on_worker_result)
        self.worker.start()

    def on_worker_result(self, payload: dict):
        if not payload.get("ok", False):
            self.current_embedding = None
            self.img_label.setText("Image")
            self.face_label.setText("Detected face")
            QMessageBox.warning(self, "Detection", payload.get("error", "Unknown error"))
            return

        # Show images
        self.img_label.setPixmap(payload["pixmap"].scaled(self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.face_label.setPixmap(payload["face_pix"].scaled(self.face_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Save embedding
        self.current_embedding = payload["embedding"]

        # Choose mode: unknown vs known DB
        X, _ = self.db.as_matrix()
        if X.shape[0] == 0:
            self._set_mode_unknown()
        else:
            self._set_mode_known()

        # Compute matches
        self.refresh_matches()

    def refresh_matches(self):
        self._update_threshold_label()
        self.matches_list.clear()
        self.current_matches = []

        if self.current_embedding is None:
            return

        X, entries = self.db.as_matrix()
        if X.shape[0] == 0:
            self.matches_list.addItem("No embeddings in DB yet.")
            return

        sims = cosine_similarity([self.current_embedding], X)[0]
        idx = np.argsort(-sims)[:TOP_K]
        threshold = self.slider.value() / 100.0

        for i in idx:
            e = entries[i]
            s = float(sims[i])
            self.current_matches.append((e, s))
            tag = "✅" if s >= threshold else "—"
            self.matches_list.addItem(f"{tag} {e.name}   (cos={s:.3f})")

        # auto-select top-1 so Enter/double-click works immediately
        if self.current_matches:
            self.matches_list.setCurrentRow(0)
            self._on_match_selection()

    def _on_match_selection(self):
        has_sel = len(self.matches_list.selectedIndexes()) > 0
        if not self.chk_new.isChecked():
            self.btn_confirm.setEnabled(has_sel and self.current_embedding is not None)

    def _save_image_copy(self, name: str) -> str:
        person_dir = os.path.join("faces_db", name)
        os.makedirs(person_dir, exist_ok=True)

        # keep original ext if sensible; fallback to .png
        orig_ext = os.path.splitext(self.current_img_path)[1].lower()
        ext = orig_ext if orig_ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp") else ".png"

        # unique filename
        i = 0
        out_path = os.path.join(person_dir, f"{name}_{i}{ext}")
        while os.path.exists(out_path):
            i += 1
            out_path = os.path.join(person_dir, f"{name}_{i}{ext}")

        img = Image.open(self.current_img_path)

        # convert only when needed
        if ext in (".jpg", ".jpeg"):
            # JPEG heeft geen alpha; converteer naar RGB
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
        else:
            # voor PNG/WebP e.d. mag alpha blijven; palette -> RGBA om veilig te zijn
            if img.mode == "P":
                img = img.convert("RGBA")

        img.save(out_path)
        return out_path

    def add_as_new(self):
        if self.current_embedding is None or not self.current_img_path:
            return
        name = self.name_input.text().strip().replace(" ", "_")
        if not name:
            QMessageBox.information(self, "Name", "Please enter a name.")
            return
        out_path = self._save_image_copy(name)
        self.db.add(name=name, image_path=out_path, embedding=self.current_embedding)
        self.statusBar().showMessage(f"Added new person {name}", 2000)
        # After first add, switch to known mode
        self._set_mode_known()
        self.refresh_matches()

    def confirm_match(self):
        if self.current_embedding is None or not self.current_img_path:
            return
        sel = self.matches_list.currentRow()
        if sel < 0 or sel >= len(self.current_matches):
            return
        entry, sim = self.current_matches[sel]
        threshold = self.slider.value()/100.0
        if sim < threshold:
            resp = QMessageBox.question(
                self, "Below threshold",
                f"Similarity {sim:.3f} < {threshold:.2f}. Add to {entry.name} anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        out_path = self._save_image_copy(entry.name)
        self.db.add(name=entry.name, image_path=out_path, embedding=self.current_embedding)
        self.statusBar().showMessage(f"Added to {entry.name}", 2000)
        self.refresh_matches()

    def _update_threshold_label(self):
        self.threshold_label.setText(f"Threshold: {self.slider.value()/100:.2f}")

# -------------------- main --------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
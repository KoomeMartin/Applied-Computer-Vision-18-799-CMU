"""
STARTER CODE for Binary Image Classification using FAST + BRIEF

Run this file after:
  1. Setting TARGET_SIZE and augmentation factors
  2. Verifying the dataset exists in data/classify/face/ and data/classify/no_face/
  3. Ensuring helper.py is in the SAME directory as this file
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import corner_detection, computeBrief


# ═══════════════════════════════════════════════════════════════════
#  WHAT THE HELPER FUNCTIONS GIVE YOU
# ═══════════════════════════════════════════════════════════════════
#
#  corner_detection(img, sigma=0.15)
#  ─────────────────────────────────
#    img    : 2D numpy array, uint8 grayscale (NOT float, NOT 3-channel)
#    sigma  : FAST corner threshold — higher means fewer corners detected
#    returns: locs  shape (K, 2)  dtype int64
#             Each row is [ROW, COL]  ← skimage convention, NOT OpenCV [x, y]
#             Returns shape (0, 2) when no corners are found — never None
#
#  computeBrief(img, locs)
#  ───────────────────────
#    img    : 2D uint8 grayscale (same image you passed to corner_detection)
#    locs   : (K, 2) int64 [row, col]  — pass directly from corner_detection
#             DO NOT swap to [col, row] or [x, y]
#    Filters keypoints within 4 pixels of any image border.
#    returns: desc  shape (K', 256)  dtype float64  values {0.0, 1.0}
#             locs  shape (K', 2)    border-filtered subset of input locs
#             K' may be 0 even when K > 0  (all points were near border)
#             D is ALWAYS 256 — hardcoded inside helper, never changes
#
#  CRITICAL: np.mean on a (0, 256) array returns a NaN vector.
#            Always check  if len(desc) == 0  BEFORE calling np.mean.
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
#  DATASET STRUCTURE
# ═══════════════════════════════════════════════════════════════════
#
#  The Face vs No-Face dataset is provided in the assignment assets.
#  All images have been preprocessed to the same dimensions.
#
#  data/
#  └── classify/
#      ├── face/          ← images containing faces
#      │   ├── img_01.jpg
#      │   └── ...
#      └── no_face/       ← images without faces
#          ├── img_01.jpg
#          └── ...
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← fill in BEFORE running anything
# ═══════════════════════════════════════════════════════════════════

# Every image is resized to this (height, width) before any processing.
# All images must share the same size — required by corner_detection.
#
# Q: Why must all images be resized to the same TARGET_SIZE before
#    extracting FAST keypoints?  (answer in extract_features below)
#
TARGET_SIZE = (128, 128)  # (height, width)

# ── Augmentation variation factors ──────────────────────────────────
# Choose exactly 2 factors of variation to simulate through augmentation.
# These should be relevant to real-world variation in the Face vs No-Face
# classification task (e.g., 'Brightness', 'Rotation').
#
# See the factors table in the handout for options.
#
AUGMENTATION_FACTOR_1 = "Brightness"  # e.g. 'Brightness'
AUGMENTATION_FACTOR_2 = "Rotation"  # e.g. 'Rotation'

DATA_DIR = "./data/classify"
RESULTS_DIR = "./results"

# D is hardcoded inside computeBrief as nbits=256 — do not change
BRIEF_D = 256


# Q3.1 Task D1: Design choice for TARGET SIZE
# Reasoning: We pick TARGET_SIZE to balance descriptor stability and computation
# efficiency. The dataset images (from data/classify) are expected to be resized
# to a common resolution before feature extraction, and 128x128 is a practical
# size that preserves facial structure while keeping the BRIEF descriptor computation
# lightweight. This choice will also simplify reporting in your write-up.
# ═══════════════════════════════════════════════════════════════════
#  1. LOAD DATASET  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════


def load_dataset(data_dir, target_size=TARGET_SIZE):
    """
    Loads, resizes, and converts every image to uint8 grayscale.

    Why uint8?
        corner_detection and computeBrief both call skimage functions that
        require a 2D uint8 array. Passing float images causes incorrect
        descriptor values. cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) after
        cv2.imread guarantees uint8.

    Returns
    -------
    images     : list of (H, W) ndarray, dtype uint8
    labels     : list of int
    class_names: list of str
    file_paths : list of str
    """
    images, labels, file_paths = [], [], []
    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    for idx, cls in enumerate(class_names):
        folder = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full_path = os.path.join(folder, fname)
            img = cv2.imread(full_path)
            if img is None:
                print(f"  [WARN] unreadable: {fname}")
                continue
            # resize → uint8 BGR, then convert to 2D uint8 grayscale
            resized = cv2.resize(img, (target_size[1], target_size[0]))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # gray.dtype == uint8, gray.ndim == 2  ✓
            images.append(gray)
            labels.append(idx)
            file_paths.append(os.path.abspath(full_path))

    print(
        f"Loaded {len(images)} images  size={target_size}  "
        f"dtype={images[0].dtype if images else 'N/A'}  classes={class_names}"
    )
    return images, labels, class_names, file_paths


# ═══════════════════════════════════════════════════════════════════
#  2. DATASET GRID  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════


def show_dataset_grid(
    images, labels, class_names, file_paths, n_per_class=4, save_dir=RESULTS_DIR
):
    """
    Saves results/dataset_grid.png and prints each image's file path.
    Include both the grid image and the printed paths in your write-up.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_cls = len(class_names)
    fig, axes = plt.subplots(
        n_cls, n_per_class, figsize=(n_per_class * 2.8, n_cls * 3.0)
    )
    print("\n========== DATASET IMAGE LINKS ==========")
    for ci, cls in enumerate(class_names):
        samples = [
            (im, fp) for im, lbl, fp in zip(images, labels, file_paths) if lbl == ci
        ][:n_per_class]
        print(f"\n  Class {ci} — '{cls}':")
        for j, (im, fp) in enumerate(samples):
            ax = axes[ci][j] if n_cls > 1 else axes[j]
            ax.imshow(im, cmap="gray")
            ax.axis("off")
            ax.set_title(os.path.basename(fp), fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(
                    cls,
                    fontsize=11,
                    fontweight="bold",
                    rotation=0,
                    labelpad=55,
                    va="center",
                )
            print(f"    [{j + 1}] {fp}")
    print("=========================================\n")
    plt.suptitle(
        f"Dataset Samples  {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px\n"
        f"Dataset: Face vs No-Face",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "dataset_grid.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Dataset grid saved → {path}")


# ═══════════════════════════════════════════════════════════════════
#  3. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────
# FACTOR SELECTION  ← complete before writing any code
#
# Factor 1 chosen: Brightness
#   Why relevant to Face vs No-Face classification (2–3 sentences):
#   Faces can appear under different lighting conditions such as indoor
#   lighting, shadows, or sunlight. Changing brightness helps the model
#   learn features that remain stable even when illumination varies.
#   OpenCV function you will use:
#   np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
#   Parameter range (e.g., ±30 pixels, ±15°):
#   Brightness shift of approximately +30 to +60 pixel intensity.
#
# Factor 2 chosen: Rotation
#   Why relevant to Face vs No-Face classification (2–3 sentences):
#   Faces in images are not always perfectly aligned and may appear
#   slightly tilted due to camera angle or head pose. Small rotations
#   help the classifier recognize faces even when orientation varies.
#   OpenCV function you will use:
#   cv2.getRotationMatrix2D() with cv2.warpAffine()
#   Parameter range (e.g., ±30 pixels, ±15°):
#   Rotation of about 30° around the image center.
#
# Q: Name one factor that would HURT BRIEF descriptor matching
#    if used as augmentation and explain why.  (2–3 sentences)
#   Large rotations would hurt BRIEF descriptor matching because BRIEF
#   compares fixed pixel pairs within a patch. When the image rotates,
#   those pixel relationships change, causing the binary descriptor to
#   change significantly even for the same feature.
# ───────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────

def augment_images(images, labels, seed=42):
    """
    Produces 2 augmented copies of every image — one per factor.
    Dataset grows from N to 3N.
    """
    np.random.seed(seed)
    aug_images = list(images)
    aug_labels = list(labels)

    h, w = images[0].shape
    centre = (w // 2, h // 2)

    for img, lbl in zip(images, labels):

        # ─────────────────────────────────────────────────────────
        # AUGMENTATION_FACTOR_1 — Brightness
        # ─────────────────────────────────────────────────────────
        shift = np.random.randint(30, 60)
        aug1 = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        aug_images.append(aug1)
        aug_labels.append(lbl)

        # ─────────────────────────────────────────────────────────
        # AUGMENTATION_FACTOR_2 — Rotation
        # ─────────────────────────────────────────────────────────
        h, w = img.shape
        centre = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(centre, 15, 1.0)
        aug2 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        aug_images.append(aug2)
        aug_labels.append(lbl)

    n_orig = len(images)
    n_aug = len(aug_images)

    print(
        f"Augmentation: {n_orig} → {n_aug}  (×{n_aug // n_orig})  "
        f"factors: {AUGMENTATION_FACTOR_1} + {AUGMENTATION_FACTOR_2}"
    )

    return aug_images, aug_labels




def show_augmentation_examples(
    images, labels, class_names, aug_images, aug_labels, save_dir=RESULTS_DIR
):
    """Saves results/augmentation_examples.png. Include in your write-up."""
    os.makedirs(save_dir, exist_ok=True)
    n_cls = len(class_names)
    n_orig = len(images)
    titles = ["Original", AUGMENTATION_FACTOR_1, AUGMENTATION_FACTOR_2]
    fig, axes = plt.subplots(n_cls, 3, figsize=(9, n_cls * 3.2))
    for ci, cls in enumerate(class_names):
        orig = next(im for im, lbl in zip(images, labels) if lbl == ci)
        aug_imgs = [
            im for im, lbl in zip(aug_images[n_orig:], aug_labels[n_orig:]) if lbl == ci
        ][:2]
        for j, (var, title) in enumerate(zip([orig] + aug_imgs, titles)):
            ax = axes[ci][j] if n_cls > 1 else axes[j]
            ax.imshow(var, cmap="gray")
            ax.axis("off")
            ax.set_title(title, fontsize=10, fontweight="bold" if j == 0 else "normal")
            if j == 0:
                ax.set_ylabel(
                    cls,
                    fontsize=11,
                    fontweight="bold",
                    rotation=0,
                    labelpad=55,
                    va="center",
                )
    plt.suptitle(
        f"Augmentation  Factor 1: {AUGMENTATION_FACTOR_1}  "
        f"Factor 2: {AUGMENTATION_FACTOR_2}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "augmentation_examples.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Augmentation grid saved → {path}")


# ═══════════════════════════════════════════════════════════════════
#  4. EXTRACT FEATURES
# ═══════════════════════════════════════════════════════════════════


def extract_features(images, sigma=0.15):
    """
    FAST keypoints  →  BRIEF descriptors  →  mean pooling  →  (N, 256) matrix

    Parameters
    ----------
    images : list of (H, W) uint8 ndarray — must be 2D uint8 grayscale
    sigma  : FAST threshold passed to corner_detection (default 0.15)

    Returns
    -------
    feature_matrix : (N, 256) float64 ndarray — one row per image, no NaN
    kp_counts      : list of int — raw keypoint count per image
    """

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 1 — answer before coding
    # Q: Why must all images be resized to TARGET_SIZE before calling
    #    corner_detection?  What would go wrong with different sizes?
    #    (2–3 sentences)
    #   Answer: Resizing ensures a consistent scale for FAST keypoint detection
    #   and BRIEF descriptor extraction. If images have varying sizes, patches
    #   used by BRIEF will have different relative scales, leading to
    #   incomparable feature descriptors across images and making learning
    #   unreliable.
    # ___________________________________________________________
    #    ___________________________________________________________
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 2 — answer before coding
    # Q: computeBrief returns a (K', 256) matrix — one row per keypoint.
    #    Why can't you pass this directly to a classifier?
    #    How does mean pooling fix this?  (2–3 sentences)
    #   Answer: The number of keypoints K' varies per image, so the descriptor
    #   matrix has variable rows. Classifiers in scikit-learn require fixed-size
    #   feature vectors for every sample. Mean pooling collapses all BRIEF
    #   descriptors for an image into a single 256-d vector, producing a fixed-
    #   length representation suitable for learning.
    # ___________________________________________________________
    #    ___________________________________________________________
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 3 — answer before coding
    # Q: What information is lost when you mean-pool K' descriptors
    #    into one vector?  How might this hurt accuracy?  (2–3 sentences)
    #   Answer: Mean pooling loses information about the distribution and
    #   arrangement of individual descriptors, such as which keypoints contributed
    #   most and how descriptors cluster spatially. This can reduce discriminative
    #   power and hurt accuracy when images differ in descriptor patterns that are
    #   not captured by the mean alone.
    # ___________________________________________________________
    #    ___________________________________________________________
    # ─────────────────────────────────────────────────────────────

    feature_matrix = []
    kp_counts = []

    for img in images:
        # Step 1 — detect keypoints
        locs = corner_detection(img, sigma)
        # Step 2 — compute BRIEF descriptors
        desc, locs = computeBrief(img, locs)
        # Step 3 — mean pool (K', 256) → (256,)
        if len(desc) == 0:
            feature_vector = np.zeros(BRIEF_D)
            kp_counts.append(0)
        else:
            feature_vector = np.mean(desc, axis=0)
            kp_counts.append(locs.shape[0])
        feature_matrix.append(feature_vector)

    feature_matrix = np.vstack(feature_matrix)  # (N, 256)
    print(
        f"Feature matrix: {feature_matrix.shape}  "
        f"NaN present: {np.isnan(feature_matrix).any()}  "
        f"avg kp/img: {np.mean(kp_counts):.1f}"
    )
    return feature_matrix, kp_counts


# ═══════════════════════════════════════════════════════════════════
#  5. TRAIN / TEST SPLIT  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════


def train_test_split(feature_matrix, labels, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    labels = np.array(labels)
    feature_matrix = np.array(feature_matrix)
    X_train, X_test, y_train, y_test = [], [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        X_train.append(feature_matrix[idx[:n_train]])
        X_test.append(feature_matrix[idx[n_train:]])
        y_train.extend(labels[idx[:n_train]].tolist())
        y_test.extend(labels[idx[n_train:]].tolist())
    X_train, X_test = np.vstack(X_train), np.vstack(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    print(f"Split: train={len(y_train)}  test={len(y_test)}  ratio={train_ratio}")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════
#  6. TRAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════


def train_classifier(X_train, y_train):
    """
    Trains one classifier on the (N, 256) feature matrix.

    ┌────────────────────────────────────────────────────────────┐
    │  CONSTRAINT: SVM / SVC is NOT permitted.                   │
    │  Choose ONE of:                                            │
    │    (a) sklearn.neighbors.KNeighborsClassifier              │
    │    (b) sklearn.ensemble.RandomForestClassifier             │
    └────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    X_train : (N_train, 256) float64 ndarray
    y_train : (N_train,)     int ndarray

    Returns
    -------
    classifier : trained sklearn classifier
    """

    # ─────────────────────────────────────────────────────────────
    # CONCEPTUAL QUESTION 4 — answer before coding
    # Q: Which classifier did you choose?
    #    How does it produce a prediction from a (256,) feature vector?
    #    Why is it appropriate for a small dataset?  (3–4 sentences)
    #   Answer: We chose K-Nearest Neighbors (KNN). Given a new 256-d feature
    #   vector, it computes distances to training vectors and assigns the label of
    #   the majority among the k nearest neighbors (k=3). This is simple, fast,
    #   and effective for small datasets where a complex model may overfit.
    # ___________________________________________________________
    #    ___________________________________________________________
    # ─────────────────────────────────────────────────────────────

    # Use a simple, robust classifier suitable for small datasets: KNN
    from sklearn.neighbors import KNeighborsClassifier

    # You may tweak n_neighbors if needed
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    return classifier


# ═══════════════════════════════════════════════════════════════════
#  7. EVALUATE  (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════


def evaluate(classifier, X_test, y_test, class_names, save_dir=RESULTS_DIR):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )

    os.makedirs(save_dir, exist_ok=True)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  Image size  : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}  D={BRIEF_D}")
    print(f"  Classes     : {class_names}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'=' * 50}\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        cmap=plt.cm.Blues, ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.show()
    print(f"Confusion matrix saved → {path}")

    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
    )


# ═══════════════════════════════════════════════════════════════════
#  8. TEST ON YOUR OWN IMAGE
# ═══════════════════════════════════════════════════════════════════


def predict_single_image(image_path, classifier):
    """
    Loads an image, extracts features, and predicts its class.

    Parameters
    ----------
    image_path : str  — path to your test image
    classifier : trained sklearn classifier

    Returns
    -------
    prediction : int (0 or 1)
    class_name : str ('face' or 'no_face')
    confidence : float (probability of predicted class, if available)
    """

    # ─────────────────────────────────────────────────────────────
    # Implement inference pipeline
    # 1. Load image with cv2.imread
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 2. Resize to TARGET_SIZE (height, width)
    h, w = TARGET_SIZE
    img_resized = cv2.resize(img, (w, h))

    # 3. Convert to grayscale (2D uint8)
    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized

    # 4. Extract features using extract_features([img_gray])
    feats, _ = extract_features([img_gray], sigma=0.15)

    if feats is None or feats.size == 0:
        # No features, cannot predict
        return -1, "unknown", None

    # 5. Predict using classifier.predict
    pred = classifier.predict(feats)
    pred_idx = int(pred[0])

    # 6. Get confidence using classifier.predict_proba if available
    confidence = None
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(feats)
        confidence = float(proba[0, pred_idx])

    # 7. Determine class name from classifier's classes_ if available
    class_name = f"class_{pred_idx}"
    if hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
        if len(classes) == 2:
            if all(isinstance(c, str) for c in classes):
                class_name = classes[pred_idx] if 0 <= pred_idx < 2 else class_name
            else:
                class_name = str(classes[pred_idx])
        else:
            class_name = f"class_{pred_idx}"

    return pred_idx, class_name, confidence


# ═══════════════════════════════════════════════════════════════════
#  MAIN  —  full pipeline
# ═══════════════════════════════════════════════════════════════════


def main():
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. load — all images are 2D uint8 grayscale after this call
    images, labels, class_names, file_paths = load_dataset(DATA_DIR, TARGET_SIZE)
    print(f"Image: shape={images[0].shape}  dtype={images[0].dtype}")

    # 2. dataset grid with file links
    show_dataset_grid(
        images, labels, class_names, file_paths, n_per_class=4, save_dir=RESULTS_DIR
    )

    # 3. augment — images must remain 2D uint8 after augmentation
    aug_images, aug_labels = augment_images(images, labels)
    show_augmentation_examples(
        images, labels, class_names, aug_images, aug_labels, RESULTS_DIR
    )

    # 4. extract — uses corner_detection + computeBrief from helper.py
    print("\nExtracting features ...")
    feature_matrix, kp_counts = extract_features(aug_images, sigma=0.15)
    print(
        f"Feature matrix shape: {feature_matrix.shape}  "
        f"(should be (N, {BRIEF_D})  NaN={np.isnan(feature_matrix).any()})"
    )

    # 5. split
    # TODO: justify your train_ratio choice in your write-up
    train_ratio = 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, aug_labels, train_ratio=train_ratio
    )

    # 6. train
    print("\nTraining classifier ...")
    classifier = train_classifier(X_train, y_train)

    # 7. evaluate
    results = evaluate(classifier, X_test, y_test, class_names)

    # ─────────────────────────────────────────────────────────────
    # PIPELINE DOCUMENTATION TASK  ← fill in after running
    #
    # Report these values in your write-up under Part D:
    #
    # TARGET SIZE (H × W)             : 128 × 128
    # Augmentation Factor 1           : Brightness
    # Augmentation Factor 2           : Rotation
    # Total images (before aug)       : 30
    # Total images (after  aug)       : 90
    # Descriptor dim D                : 256  (always 256)
    # Feature matrix shape (N, D)     : (90, 256)
    # Train samples                   : 72
    # Test samples                    : 18
    # Classifier used                 : k-NN
    # Key hyperparameter              : 3
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # TEST ON YOUR OWN IMAGE TASK  ← complete for Task D5
    #
    # Uncomment and fill in after implementing predict_single_image:
    #
    test_image_path = "./data/my_photo.jpg"
    prediction, class_name, confidence = predict_single_image(
        test_image_path, classifier
    )
    print(f"\nTest Image Prediction:")
    print(f"  Predicted class: {class_name}")
    print(f"  Confidence: {confidence:.2%}")
    # ─────────────────────────────────────────────────────────────

    return results


if __name__ == "__main__":
    main()

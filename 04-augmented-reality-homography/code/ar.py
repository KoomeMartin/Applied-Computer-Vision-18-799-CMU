import numpy as np
import cv2
import os

from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH


def center_crop_to(img, target_h, target_w):
    h, w = img.shape[:2]
    # If needed, scale up so we can crop exactly target size
    scale = 1.0
    if h < target_h or w < target_w:
        scale = max(target_h / float(h), target_w / float(w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    # Center crop
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return img[top : top + target_h, left : left + target_w]


def main():
    # Load reference book cover and the two videos
    cover_path = os.path.join("data", "cv_cover.jpg")
    book_video_path = os.path.join("data", "book.mov")
    ar_video_path = os.path.join("data", "ar_source.mov")

    cover = cv2.imread(cover_path)
    book_frames = loadVid(book_video_path)
    ar_frames = loadVid(ar_video_path)

    if cover is None:
        raise FileNotFoundError("Cover image not found")
    if book_frames is None or book_frames.size == 0:
        raise FileNotFoundError("Book video could not be loaded")
    if ar_frames is None or ar_frames.size == 0:
        raise FileNotFoundError("AR video could not be loaded")

    h_cover, w_cover = cover.shape[:2]
    # Prepare output video writer (same size as book frames)
    h_book, w_book = book_frames[0].shape[:2]
    # Try to determine FPS from the input book video for a faithful output
    cap_for_fps = cv2.VideoCapture(book_video_path)
    fps = cap_for_fps.get(cv2.CAP_PROP_FPS)
    cap_for_fps.release()
    if fps == 0:
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ar.avi")
    fps = 30  # a reasonable default; book video fps may differ
    out = cv2.VideoWriter(out_path, fourcc, fps, (w_book, h_book))

    # Process frames
    N = (
        min(book_frames.shape[0], ar_frames.shape[0])
        if isinstance(book_frames, np.ndarray)
        else 0
    )
    if N == 0:
        print("No frames to process")
        return

    for i in range(N):
        frame = book_frames[i]
        ar_src = ar_frames[i]

        # 1) Detect/match features between cover and current book frame
        matches, locs1, locs2 = matchPics(cover, frame)

        overlay = frame.copy()
        if matches is not None and len(matches) >= 4:
            # Coordinates are in (row, col) -> (y, x). Convert to (x, y)
            pts1 = locs1[matches[:, 0], :]
            pts2 = locs2[matches[:, 1], :]
            x1 = pts1[:, [1, 0]].astype(float)
            x2 = pts2[:, [1, 0]].astype(float)

            try:
                H2to1, inliers = computeH_ransac(x1, x2)
            except Exception:
                H2to1, inliers = None, None

            if H2to1 is not None and inliers is not None and int(inliers.sum()) >= 4:
                # Crop AR frame to central region to fit the cover size
                ar_template = center_crop_to(ar_src, h_cover, w_cover)
                # Warp AR template onto the book frame using the computed H
                overlay = compositeH(H2to1, ar_template, frame)
        # Write frame
        out.write(overlay)

    out.release()
    print(f"AR video written to {out_path}")


if __name__ == "__main__":
    main()

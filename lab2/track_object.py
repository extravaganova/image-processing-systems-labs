"""Track an object in a video using ORB keypoints and a homography."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    """Runtime parameters for the keypoint tracker."""

    max_features: int = 1200
    ratio_test: float = 0.75
    min_matches: int = 8


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse an ROI written as ``x,y,w,h``."""
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("ROI must have the form x,y,w,h") from error
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise argparse.ArgumentTypeError("ROI dimensions must be positive")
    return x, y, width, height


def build_tracker(
    first_frame: np.ndarray,
    roi: tuple[int, int, int, int],
    config: TrackerConfig,
) -> tuple[cv2.ORB, list[cv2.KeyPoint], np.ndarray, np.ndarray]:
    """Create ORB features and the reference rectangle for the first frame."""
    x, y, width, height = roi
    reference = first_frame[y : y + height, x : x + width]
    if reference.size == 0:
        raise ValueError("ROI lies outside the first video frame")
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=config.max_features)
    keypoints, descriptors = detector.detectAndCompute(gray_reference, None)
    if descriptors is None or len(keypoints) < config.min_matches:
        raise ValueError("The selected object has too few keypoints")
    corners = np.float32(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    ).reshape(-1, 1, 2)
    return detector, keypoints, descriptors, corners


def locate_object(
    frame: np.ndarray,
    detector: cv2.ORB,
    reference_keypoints: list[cv2.KeyPoint],
    reference_descriptors: np.ndarray,
    reference_corners: np.ndarray,
    matcher: cv2.BFMatcher,
    config: TrackerConfig,
) -> np.ndarray | None:
    """Return the projected object corners, or None when tracking is unreliable."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_keypoints, frame_descriptors = detector.detectAndCompute(gray_frame, None)
    if frame_descriptors is None:
        return None
    matches = matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)
    good_matches = [
        first
        for first, second in matches
        if first.distance < config.ratio_test * second.distance
    ]
    if len(good_matches) < config.min_matches:
        return None
    source = np.float32(
        [reference_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    destination = np.float32(
        [frame_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < config.min_matches:
        return None
    return cv2.perspectiveTransform(reference_corners, homography)


def draw_tracking_result(
    frame: np.ndarray, corners: np.ndarray | None, frame_number: int
) -> np.ndarray:
    """Draw the tracking polygon and status label on a copy of the frame."""
    result = frame.copy()
    if corners is None:
        label = "Object: not found"
        color = (0, 0, 255)
    else:
        polygon = np.int32(corners)
        cv2.polylines(result, [polygon], True, (0, 255, 0), 3)
        label = "Object: tracked"
        color = (0, 180, 0)
    cv2.rectangle(result, (12, 12), (310, 55), (255, 255, 255), -1)
    cv2.putText(result, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(result, f"Frame: {frame_number}", (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return result


def track_video(
    input_path: Path,
    output_path: Path,
    roi: tuple[int, int, int, int] | None = None,
    config: TrackerConfig | None = None,
) -> int:
    """Track the object from the first frame and save an annotated video."""
    active_config = config or TrackerConfig()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open input video: {input_path}")
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise ValueError("Input video has no readable first frame")
    height, width = first_frame.shape[:2]
    selected_roi = roi or (0, 0, width, height)
    detector, keypoints, descriptors, corners = build_tracker(
        first_frame, selected_roi, active_config
    )
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Cannot open output video: {output_path}")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            projected = locate_object(
                frame, detector, keypoints, descriptors, corners, matcher, active_config
            )
            writer.write(draw_tracking_result(frame, projected, frame_number))
            frame_number += 1
    finally:
        capture.release()
        writer.release()
    if frame_number == 0:
        raise ValueError("Input video has no frames after initialization")
    return frame_number


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input video")
    parser.add_argument("output", type=Path, help="annotated output video")
    parser.add_argument("--roi", type=parse_roi, help="object ROI on first frame: x,y,w,h")
    return parser


def main() -> None:
    """Run the tracker from command-line arguments."""
    args = build_parser().parse_args()
    frames = track_video(args.input, args.output, args.roi)
    print(f"Processed {frames} frames: {args.output}")


if __name__ == "__main__":
    main()

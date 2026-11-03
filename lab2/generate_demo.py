"""Generate a deterministic textured-card video for testing the tracker."""

from pathlib import Path

import cv2
import numpy as np


def make_demo(output: Path, frame_count: int = 90) -> None:
    """Write a video whose textured card moves and changes perspective."""
    width, height = 960, 540
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Cannot create demo video: {output}")
    card = np.full((260, 420, 3), (235, 235, 235), dtype=np.uint8)
    for row in range(0, 260, 26):
        cv2.line(card, (0, row), (420, row), (70, 70, 70), 1)
    for column in range(0, 420, 26):
        cv2.line(card, (column, 0), (column, 260), (70, 70, 70), 1)
    cv2.putText(card, "SKOI LAB 2", (48, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (20, 20, 20), 3)
    source = np.float32([[0, 0], [420, 0], [420, 260], [0, 260]])
    destination = np.float32([[270, 120], [690, 100], [715, 380], [245, 400]])
    for index in range(frame_count):
        canvas = np.full((height, width, 3), (45, 55, 70), dtype=np.uint8)
        shift = index / max(frame_count - 1, 1)
        current = destination + np.float32([[80 * shift, 35 * shift], [50 * shift, 55 * shift], [-55 * shift, -25 * shift], [-80 * shift, -35 * shift]])
        matrix = cv2.getPerspectiveTransform(source, current)
        warped = cv2.warpPerspective(card, matrix, (width, height))
        mask = cv2.warpPerspective(np.full(card.shape[:2], 255, dtype=np.uint8), matrix, (width, height))
        canvas[mask > 0] = warped[mask > 0]
        writer.write(canvas)
    writer.release()


if __name__ == "__main__":
    make_demo(Path("demo_input.mp4"))

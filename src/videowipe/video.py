"""Sequential source-frame positioning shared by detection and execution."""


def skip_frames(reader, count: int, check_cancelled=None) -> None:
    """Consume exactly ``count`` frames from the reader's current position.

    Callers own the reader and track its next ordinal. OpenCV's successful
    frame seek can still land on another image for timestamp-irregular video;
    reading forward preserves the same ordinals as a full run from frame zero.
    """
    for decoded in range(count):
        if check_cancelled is not None:
            check_cancelled()
        if not reader.read()[0]:
            raise ValueError(
                f"Cannot advance source reader by {count} frames; decoded {decoded}"
            )

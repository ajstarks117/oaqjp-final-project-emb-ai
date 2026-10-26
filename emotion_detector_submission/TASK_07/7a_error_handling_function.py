"""Error-aware emotion detector implementation."""

from EmotionDetection.emotion_detection import emotion_detector


def safe_emotion_detector(text_to_analyze):
    """Validate blank input and call the main detector."""
    if text_to_analyze is None or not str(text_to_analyze).strip():
        raise ValueError("Text to analyze is empty")
    return emotion_detector(text_to_analyze)

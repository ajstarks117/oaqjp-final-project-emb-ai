"""Unit tests for emotion detection."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from EmotionDetection.emotion_detection import emotion_detector


@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_detector_returns_scores_and_dominant_emotion(mock_post):
    """The API response should be normalized into the expected schema."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "emotion": {
            "document": {
                "emotion": {
                    "anger": 0.1,
                    "disgust": 0.1,
                    "fear": 0.1,
                    "joy": 0.8,
                    "sadness": 0.2,
                }
            }
        }
    }
    mock_post.return_value.raise_for_status.return_value = None

    result = emotion_detector("I am really happy today")

    assert set(result) == {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"}
    assert result["joy"] == 0.8
    assert result["dominant_emotion"] == "joy"


@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_detector_handles_empty_response(mock_post):
    """Empty response payloads should still return a valid structure."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"emotion": {"document": {"emotion": {}}}}
    mock_post.return_value.raise_for_status.return_value = None

    result = emotion_detector("I am very angry")
    assert "dominant_emotion" in result
    assert result["dominant_emotion"] in {"anger", "joy", "fear", "sadness", "disgust"}


def test_emotion_detector_rejects_blank_input():
    """Blank strings should be rejected before any API call is made."""
    with pytest.raises(ValueError, match="Text to analyze is empty"):
        emotion_detector("   ")


@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_detector_handles_network_error(mock_post):
    """Network failures should degrade to a safe fallback without crashing."""
    mock_post.side_effect = requests.exceptions.RequestException("offline")

    result = emotion_detector("I feel nervous")
    assert "dominant_emotion" in result
    assert result["dominant_emotion"] in {"fear", "joy", "anger", "sadness", "disgust"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I am very angry", "anger"),
        ("This is wonderful", "joy"),
        ("I am terribly scared", "fear"),
    ],
)
@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_detector_keyword_fallback(mock_post, text, expected):
    """Keyword matching should still choose the strongest emotional signal."""
    mock_post.side_effect = requests.exceptions.RequestException("offline")

    result = emotion_detector(text)
    assert result["dominant_emotion"] == expected

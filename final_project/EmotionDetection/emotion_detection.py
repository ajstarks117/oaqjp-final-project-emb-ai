"""Watson NLP emotion detection wrapper with local fallback."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

import requests

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def _fallback_emotion_scores(text_to_analyze: str) -> Dict[str, Any]:
    """Return deterministic keyword-based scores when the Watson service is unavailable."""
    text = str(text_to_analyze or "").lower()
    scores = {emotion: 0.0 for emotion in EMOTIONS}
    keyword_map = {
        "anger": ("angry", "mad", "furious", "rage", "hate", "annoyed", "upset"),
        "disgust": ("disgusted", "gross", "repulsive", "nasty", "awful", "revolting"),
        "fear": ("afraid", "scared", "fear", "terrified", "panic", "nervous", "unsafe"),
        "joy": ("happy", "joy", "excited", "love", "great", "wonderful", "delighted"),
        "sadness": ("sad", "crying", "depressed", "down", "lonely", "grief", "heartbroken"),
    }

    for emotion, triggers in keyword_map.items():
        for trigger in triggers:
            if trigger in text:
                scores[emotion] += 0.9

    if all(score == 0 for score in scores.values()):
        scores["joy"] = 0.25

    dominant = max(scores, key=scores.get)
    return {**scores, "dominant_emotion": dominant}


def _extract_emotion_scores(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the Watson API response into the expected dictionary."""
    if not isinstance(raw_response, dict):
        return _fallback_emotion_scores("")

    emotion_data = raw_response.get("emotion", {}).get("document", {}).get("emotion", {})
    if not isinstance(emotion_data, dict) or not emotion_data:
        return _fallback_emotion_scores("")

    cleaned = {emotion: float(emotion_data.get(emotion, 0.0) or 0.0) for emotion in EMOTIONS}
    cleaned["dominant_emotion"] = max(cleaned, key=cleaned.get)
    return cleaned


def emotion_detector(text_to_analyze: str) -> Dict[str, Any]:
    """Call the Watson Emotion API and return the five scores plus the dominant emotion."""
    if text_to_analyze is None or not str(text_to_analyze).strip():
        raise ValueError("Text to analyze is empty")

    api_key = os.getenv("WATSON_APIKEY", "")
    url = os.getenv("WATSON_URL", "https://sn-watson-emotion.labs.skills.network")
    version = os.getenv("WATSON_VERSION", "2022-04-07")

    try:
        auth = base64.b64encode(f"apikey:{api_key}".encode("utf-8")).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth}",
        }
        payload = {"text": str(text_to_analyze), "features": {"emotion": {}}}
        endpoint = f"{url.rstrip('/')}/v1/analyze?version={version}"
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=15)
        if response.status_code == 400:
            raise ValueError("Text to analyze is empty")
        response.raise_for_status()
        return _extract_emotion_scores(response.json())
    except requests.RequestException:
        return _fallback_emotion_scores(str(text_to_analyze))

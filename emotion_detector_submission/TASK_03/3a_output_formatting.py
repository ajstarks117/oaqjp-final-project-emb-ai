from EmotionDetection.emotion_detection import emotion_detector

result = emotion_detector("I am very angry")
print("anger:", result["anger"])
print("disgust:", result["disgust"])
print("fear:", result["fear"])
print("joy:", result["joy"])
print("sadness:", result["sadness"])
print("dominant_emotion:", result["dominant_emotion"])

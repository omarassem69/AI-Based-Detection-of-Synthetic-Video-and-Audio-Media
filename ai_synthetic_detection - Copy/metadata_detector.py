import json
import subprocess
import os

FFPROBE_EXE = r"C:\Users\omara\Downloads\ffmpeg-8.0.1-essentials_build (1)\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"

AI_KEYWORDS = [
    "chatgpt", "openai", "dall-e", "dalle",
    "midjourney", "stable diffusion", "runway",
    "sora", "pika", "leonardo", "firefly",
    "synthetic", "generated"
]

REAL_CAMERA_KEYWORDS = [
    "iphone", "samsung", "huawei", "canon",
    "nikon", "sony", "xiaomi", "oppo",
    "vivo", "pixel", "camera",
    "dell"
]


def extract_metadata(file_path):
    if not os.path.exists(FFPROBE_EXE):
        return {
            "score": 40,
            "label": "Suspicious",
            "reason": "ffprobe.exe path not found"
        }

    command = [
        FFPROBE_EXE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        if result.stdout.strip() == "":
            return {
                "score": 50,
                "label": "Suspicious",
                "reason": "No metadata output from ffprobe"
            }

        metadata_json = json.loads(result.stdout)
        metadata_text = json.dumps(metadata_json).lower()

        return analyze_metadata(metadata_text)

    except Exception as e:
        return {
            "score": 40,
            "label": "Suspicious",
            "reason": "Metadata extraction failed: " + str(e)
        }


def analyze_metadata(text):
    if len(text.strip()) < 20:
        return {
            "score": 50,
            "label": "Suspicious",
            "reason": "Metadata missing"
        }

    for word in AI_KEYWORDS:
        if word in text:
            return {
                "score": 100,
                "label": "Fake",
                "reason": f"AI keyword found: {word}"
            }

    for word in REAL_CAMERA_KEYWORDS:
        if word in text:
            return {
                "score": 10,
                "label": "Normal",
                "reason": f"Real device found: {word}"
            }

    return {
        "score": 30,
        "label": "Weak Suspicious",
        "reason": "No clear device or AI keyword"
    }
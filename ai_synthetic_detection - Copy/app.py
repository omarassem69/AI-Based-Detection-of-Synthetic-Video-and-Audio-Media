import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
import subprocess
from metadata_detector import extract_metadata
from text_detector import predict_text

# =========================
# CONFIG
# =========================
VIDEO_MODEL_PATH = "models/video_model_balanced.keras"
AUDIO_MODEL_PATH = "models/audio_model_final.keras"

IMG_SIZE = (160, 160)
MAX_FRAMES = 10

# Video thresholds (keep as-is)
FAKE_THRESHOLD = 0.75
SUSPICIOUS_THRESHOLD = 0.55

# Audio thresholds
AUDIO_FAKE_THRESHOLD = 0.50
AUDIO_SUSPICIOUS_THRESHOLD = 0.35

AUDIO_SR = 16000

# Grad-CAM last conv layer name
LAST_CONV_LAYER_NAME = "Conv_1"

# ✅ IMPORTANT: If audio looks flipped, change this ONE line
INVERT_AUDIO = False  # try True first; if still all REAL, set to False

# Hybrid (late fusion) weights
HYBRID_W_VIDEO = 0.5
HYBRID_W_AUDIO = 0.5

# Try extracting audio from video automatically if no audio file uploaded
AUTO_EXTRACT_AUDIO_FROM_VIDEO = True

# 🔧 Your ffmpeg.exe path (no PATH needed)
FFMPEG_EXE = r"C:\Users\omara\Downloads\ffmpeg-8.0.1-essentials_build (1)\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

# Show extraction debug info
SHOW_FFMPEG_DEBUG = True

# ✅ NEW: Speech gate (skip audio scoring if no speech)
ENABLE_SPEECH_GATE = True
# Gate sensitivity (lower = more likely to accept audio as "speech-like")
SPEECH_RMS_MEAN_TH = 0.005
SPEECH_RMS_STD_TH = 0.002


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    video_model = None
    audio_model = None
    video_error = None
    audio_error = None

    try:
        if os.path.exists(VIDEO_MODEL_PATH):
            video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
        else:
            video_error = f"File not found: {VIDEO_MODEL_PATH}"
    except Exception as e:
        video_model = None
        video_error = str(e)

    try:
        if os.path.exists(AUDIO_MODEL_PATH):
            audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        else:
            audio_error = f"File not found: {AUDIO_MODEL_PATH}"
    except Exception as e:
        audio_model = None
        audio_error = str(e)

    return video_model, audio_model, video_error, audio_error


# =========================
# VIDEO UTILS
# =========================
def extract_frames(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // max_frames, 1)

    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, IMG_SIZE)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        idx += 1

    cap.release()

    if len(frames) == 0:
        return np.zeros((max_frames, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.stack(frames, axis=0).astype(np.float32)


def predict_video(model, frames):
    preds = model.predict(frames, verbose=0)

    if preds.ndim == 2 and preds.shape[1] == 1:
        probs = preds[:, 0]
    elif preds.ndim == 2 and preds.shape[1] == 2:
        probs = preds[:, 1]
    else:
        probs = preds.reshape(-1)

    return float(np.mean(probs))


# =========================
# GRAD-CAM (VIDEO)
# =========================
def make_gradcam_heatmap(model, image, last_conv_layer_name=LAST_CONV_LAYER_NAME):
    img_tensor = tf.expand_dims(image, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)

        if predictions.shape[-1] == 2:
            loss = predictions[:, 1]  # fake
        else:
            loss = predictions[:, 0]  # sigmoid

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(frame, heatmap, alpha=0.4):
    frame_uint8 = np.uint8(frame * 255)
    return cv2.addWeighted(frame_uint8, 1 - alpha, heatmap, alpha, 0)


# =========================
# AUDIO EXTRACTION (VIDEO -> WAV)
# =========================
def extract_audio_from_video_ffmpeg(video_path: str, sr: int = AUDIO_SR):
    """
    Extract mono 16k wav from a video using ffmpeg.
    Returns (wav_path, debug_dict) where wav_path can be None.
    """
    debug = {
        "ffmpeg_exe_exists": os.path.exists(FFMPEG_EXE),
        "video_path_exists": os.path.exists(video_path),
        "cmd": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "wav_path": None,
        "wav_exists": False,
        "wav_size": 0,
    }

    if not debug["ffmpeg_exe_exists"]:
        debug["stderr"] = f"FFmpeg not found at: {FFMPEG_EXE}"
        return None, debug

    try:
        wav_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sr),
            "-f", "wav",
            wav_path
        ]
        debug["cmd"] = cmd
        debug["wav_path"] = wav_path

        result = subprocess.run(cmd, capture_output=True, text=True)
        debug["returncode"] = result.returncode
        debug["stdout"] = result.stdout[-2000:] if result.stdout else ""
        debug["stderr"] = result.stderr[-2000:] if result.stderr else ""

        if os.path.exists(wav_path):
            debug["wav_exists"] = True
            debug["wav_size"] = os.path.getsize(wav_path)

        if debug["wav_exists"] and debug["wav_size"] > 1000 and result.returncode == 0:
            return wav_path, debug

        return None, debug

    except Exception as e:
        debug["stderr"] = f"Exception: {e}"
        return None, debug


# =========================
# AUDIO UTILS (MFCC -> (40,400,1))
# =========================
def build_audio_features_mfcc(audio_path, sr=AUDIO_SR, n_mfcc=40, target_T=400):
    try:
        import librosa
    except Exception:
        raise RuntimeError("Install librosa: pip install librosa audioread soundfile")

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    n_fft = 512
    hop_length = 160
    win_length = 400

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    ).astype(np.float32)

    if mfcc.shape[1] < target_T:
        pad = target_T - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :target_T]

    return mfcc


def predict_audio(audio_model, audio_path):
    feat = build_audio_features_mfcc(audio_path, n_mfcc=40, target_T=400)
    x = feat[..., np.newaxis][np.newaxis, ...].astype(np.float32)  # (1,40,400,1)

    preds = audio_model.predict(x, verbose=0)

    if preds.ndim == 2 and preds.shape[1] == 1:
        raw = float(preds[0, 0])          # sigmoid
    elif preds.ndim == 2 and preds.shape[1] == 2:
        raw = float(preds[0, 1])          # softmax fake index
    else:
        raw = float(np.mean(preds))

    score = (1.0 - raw) if INVERT_AUDIO else raw
    return score, raw


def speech_gate(audio_path, sr=AUDIO_SR):
    """
    Quick activity/speech-like gate to avoid OOD cases (sea/wind/music/no voice).
    Returns (ok_to_use_audio, info_dict)
    """
    info = {"enabled": ENABLE_SPEECH_GATE, "rms_mean": None, "rms_std": None, "decision": None}

    if not ENABLE_SPEECH_GATE:
        info["decision"] = "SKIPPED (gate disabled)"
        return True, info

    try:
        import librosa
    except Exception:
        # If librosa missing, we cannot gate -> allow audio
        info["decision"] = "ALLOWED (librosa missing, gate bypass)"
        return True, info

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) < sr:  # <1 sec
        info["decision"] = "BLOCKED (too short)"
        info["rms_mean"], info["rms_std"] = 0.0, 0.0
        return False, info

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    info["rms_mean"] = rms_mean
    info["rms_std"] = rms_std

    # Block near-silence or steady ambient
    if rms_mean < SPEECH_RMS_MEAN_TH or rms_std < SPEECH_RMS_STD_TH:
        info["decision"] = "BLOCKED (no speech / ambient)"
        return False, info

    info["decision"] = "ALLOWED (speech/activity detected)"
    return True, info


# =========================
# LABEL + EXPLANATION
# =========================
def score_to_label(score, use_audio_thresholds=False):
    if use_audio_thresholds:
        if score >= AUDIO_FAKE_THRESHOLD:
            return "FAKE ❌"
        elif score >= AUDIO_SUSPICIOUS_THRESHOLD:
            return "SUSPICIOUS ⚠️"
        else:
            return "REAL ✅"

    if score >= FAKE_THRESHOLD:
        return "FAKE ❌"
    elif score >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS ⚠️"
    else:
        return "REAL ✅"


def generate_explanation(video_score, audio_used, hybrid_used):
    reasons = []

    if video_score is None:
        reasons.append("No video was provided, so visual analysis was skipped.")
    else:
        if video_score > FAKE_THRESHOLD:
            reasons.append("Strong visual inconsistencies detected (mouth/eye artifacts, skin texture issues, face blending).")
        elif video_score > SUSPICIOUS_THRESHOLD:
            reasons.append("Moderate visual irregularities detected.")
        else:
            reasons.append("Visual patterns are mostly consistent with real videos.")

    if audio_used:
        reasons.append("Audio analysis was included.")
    else:
        reasons.append("Audio analysis was not used (or skipped by speech gate).")

    if hybrid_used:
        reasons.append("Final decision used a hybrid (late-fusion) score combining audio and video probabilities.")
    else:
        reasons.append("Final decision used a single-modality score.")

    return " ".join(reasons)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="AI Synthetic Media Detector", layout="wide")
st.title("🎥 AI Synthetic Media Detector")
st.caption("Tip: Upload BOTH video and audio for Hybrid score. If you only upload a video, the app will try to extract audio automatically.")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mpeg4"])
audio_file = st.file_uploader("Upload audio (optional)", type=["wav", "mp3"])
text_input = st.text_area("Enter text for AI-generated text detection (optional)")

video_model, audio_model, video_err, audio_err = load_models()

st.write("✅ Video model loaded" if video_model else f"❌ Video model error: {video_err}")
st.write("✅ Audio model loaded" if audio_model else f"❌ Audio model error: {audio_err}")

video_score = None
audio_score = None
audio_raw = None
audio_used = False
audio_source = None  # "uploaded" or "extracted"
frames = None

video_path = None
audio_path = None
ffmpeg_debug = None
metadata_result = None
text_result = None
gate_info = None  # speech gate info
# TEXT
if text_input.strip():
    text_result = predict_text(text_input)
# VIDEO
if video_file and video_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_path)
    frames = extract_frames(video_path)
    video_score = predict_video(video_model, frames)
    metadata_result = extract_metadata(video_path)

# AUDIO (uploaded)
if audio_file:
    suffix = os.path.splitext(audio_file.name)[1].lower()
    if suffix not in [".wav", ".mp3"]:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name
        metadata_result = extract_metadata(audio_path)

    st.audio(audio_path)

    if audio_model:
        st.write("Audio model input shape:", audio_model.input_shape)
        try:
            ok, gate_info = speech_gate(audio_path, sr=AUDIO_SR)
            if not ok:
                st.warning("Speech gate: No speech detected (ambient/no talking). Audio score skipped to avoid unreliable prediction.")
            else:
                audio_score, audio_raw = predict_audio(audio_model, audio_path)
                audio_used = True
                audio_source = "uploaded"
        except Exception as e:
            st.error(f"Audio prediction failed: {e}")

# AUDIO (auto-extract from video if no uploaded audio)
if (audio_score is None) and (video_path is not None) and (audio_model is not None) and AUTO_EXTRACT_AUDIO_FROM_VIDEO:
    extracted, ffmpeg_debug = extract_audio_from_video_ffmpeg(video_path, sr=AUDIO_SR)
    if extracted is not None:
        try:
            ok, gate_info = speech_gate(extracted, sr=AUDIO_SR)
            if not ok:
                st.warning("Speech gate: No speech detected in extracted audio (ambient/no talking). Audio score skipped.")
            else:
                extracted_score, extracted_raw = predict_audio(audio_model, extracted)
                audio_score = extracted_score
                audio_raw = extracted_raw
                audio_used = True
                audio_source = "extracted_from_video"

                with st.expander("🎧 Audio extracted from video (for hybrid)", expanded=False):
                    st.audio(extracted)
                    st.write("Audio model input shape:", audio_model.input_shape)
        except Exception as e:
            st.warning(f"Audio extracted but prediction failed: {e}")

# =========================
# FINAL DECISION LOGIC
# =========================
hybrid_used = False
final_score = None
use_audio_thresholds = False
# =========================
# METADATA HARD RULE
# =========================
if metadata_result is not None:
    if metadata_result["score"] >= 70:
        final_score = 1.0
        label = "FAKE ❌"
        st.error("🚨 Metadata detected AI generation → 100% FAKE")
        
        st.subheader("📊 Metadata Analysis")
        st.write(metadata_result)
        # Show metadata always
        if metadata_result is not None:
            st.subheader("🧾 Metadata Analysis")
            st.write("Label:", metadata_result["label"])
            st.write("Fake Score:", str(metadata_result["score"]) + "%")
            st.write("Reason:", metadata_result["reason"])
        st.subheader("📊 Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Final Probability:** {final_score:.2f}")

        st.stop()
# =========================
# TEXT HARD RULE
# =========================
    if text_result["score"] >= 70:
        final_score = 1.0
        label = "FAKE ❌"
        st.error("🚨 Text detected AI-generated content → 100% FAKE")

        st.subheader("📝 Text Analysis")
        st.write("Label:", text_result["label"])
        st.write("Fake Score:", str(text_result["score"]) + "%")
        st.write("Reason:", text_result["reason"])

        st.subheader("📊 Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Final Probability:** {final_score:.2f}")

        st.stop()
TEXT_WEIGHT = 0.2

text_score_norm = 0
if text_result is not None:
    text_score_norm = text_result["score"] / 100.0

if (video_score is not None) and (audio_score is not None):
    hybrid_used = True
    final_score = (
        (0.4 * video_score) +
        (0.4 * audio_score) +
        (TEXT_WEIGHT * text_score_norm)
    )
    use_audio_thresholds = False

elif video_score is not None:
    final_score = (
        (0.8 * video_score) +
        (TEXT_WEIGHT * text_score_norm)
    )
    use_audio_thresholds = False

elif audio_score is not None:
    final_score = (
        (0.8 * audio_score) +
        (TEXT_WEIGHT * text_score_norm)
    )
    use_audio_thresholds = True

elif text_result is not None:
    final_score = text_score_norm
    use_audio_thresholds = False

# =========================
# RESULTS
# =========================
if final_score is not None:
        # =========================
    # TEXT UI
    # =========================
    if text_result is not None:
        st.subheader("📝 Text Analysis")

        t_score = text_result["score"]
        t_label = text_result["label"]
        t_reason = text_result["reason"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Text Label", t_label)

        with col2:
            st.metric("Text Fake Score", f"{t_score}%")

        with col3:
            if t_score >= 70:
                st.error("🔴 High Risk")
            elif t_score >= 40:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

        st.progress(t_score / 100)
        st.info(f"Reason: {t_reason}")
    label = score_to_label(final_score, use_audio_thresholds=use_audio_thresholds)

    # =========================
    # POLISHED METADATA UI
    # =========================
    if metadata_result is not None:
        st.subheader("🧾 Metadata Analysis")

        m_score = metadata_result["score"]
        m_label = metadata_result["label"]
        m_reason = metadata_result["reason"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Metadata Label", m_label)

        with col2:
            st.metric("Metadata Fake Score", f"{m_score}%")

        with col3:
            if m_score >= 70:
                st.error("High Risk")
            elif m_score >= 30:
                st.warning("Medium Risk")
            else:
                st.success("Low Risk")

        st.progress(m_score / 100)
        st.info(f"Reason: {m_reason}")
        st.caption("Metadata is used as supporting evidence only, because it can be edited or removed.")

    # =========================
    # POLISHED RESULT UI
    # =========================
    st.subheader("📊 Final Result")

    if "FAKE" in label:
        st.error(f"Prediction: {label}")
    elif "SUSPICIOUS" in label:
        st.warning(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    st.metric("Final Fake Probability", f"{final_score * 100:.1f}%")
    st.progress(float(final_score))

    col1, col2 = st.columns(2)

    with col1:
        if video_score is not None:
            st.metric("Video Fake Score", f"{video_score * 100:.1f}%")
        else:
            st.metric("Video Fake Score", "No video")

    with col2:
        if audio_score is not None:
            st.metric("Audio Fake Score", f"{audio_score * 100:.1f}%")
        else:
            st.metric("Audio Fake Score", "No audio / skipped")
    label = score_to_label(final_score, use_audio_thresholds=use_audio_thresholds)
    # Show metadata always
    if metadata_result is not None:
        st.subheader("🧾 Metadata Analysis")
        st.write("Label:", metadata_result["label"])
        st.write("Fake Score:", str(metadata_result["score"]) + "%")
        st.write("Reason:", metadata_result["reason"])
    st.subheader("📊 Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Final Probability:** {final_score:.2f}")

    st.write(f"**Video Score:** {video_score:.2f}" if video_score is not None else "**Video Score:** (no video)")

    if audio_score is not None:
        if audio_source == "uploaded":
            st.write(f"**Audio Score:** {audio_score:.2f} (uploaded)")
        elif audio_source == "extracted_from_video":
            st.write(f"**Audio Score:** {audio_score:.2f} (extracted from video)")
        else:
            st.write(f"**Audio Score:** {audio_score:.2f}")
    else:
        st.write("**Audio Score:** (no audio / skipped)")

    if hybrid_used:
        st.success(f"Hybrid mode ON ✅  (w_video={HYBRID_W_VIDEO}, w_audio={HYBRID_W_AUDIO})")
    else:
        st.info("Hybrid mode OFF (single modality).")

    if audio_used:
        st.success("Audio model used.")
        with st.expander("Audio debug (raw output)"):
            st.write("INVERT_AUDIO =", INVERT_AUDIO)
            st.write("Raw model output =", audio_raw)
            st.write("Final audio score used =", audio_score)
            st.write("Audio source =", audio_source)
    else:
        st.info("Audio model not used (or skipped by speech gate).")

    # Speech gate debug
    if gate_info is not None:
        with st.expander("🗣️ Speech gate debug", expanded=False):
            st.write(gate_info)
            st.write("Thresholds:", {"rms_mean_th": SPEECH_RMS_MEAN_TH, "rms_std_th": SPEECH_RMS_STD_TH})

    # ffmpeg debug (shows why extraction failed)
    if SHOW_FFMPEG_DEBUG and (video_path is not None) and (audio_file is None) and AUTO_EXTRACT_AUDIO_FROM_VIDEO:
        with st.expander("🧪 FFmpeg extraction debug (if audio is missing)", expanded=False):
            if ffmpeg_debug is None:
                st.write("No ffmpeg debug info available.")
            else:
                st.write("FFmpeg exe exists:", ffmpeg_debug.get("ffmpeg_exe_exists"))
                st.write("Video exists:", ffmpeg_debug.get("video_path_exists"))
                st.write("Return code:", ffmpeg_debug.get("returncode"))
                st.write("WAV exists:", ffmpeg_debug.get("wav_exists"))
                st.write("WAV size:", ffmpeg_debug.get("wav_size"))
                st.write("WAV path:", ffmpeg_debug.get("wav_path"))
                st.code(" ".join(ffmpeg_debug.get("cmd") or []))
                if ffmpeg_debug.get("stderr"):
                    st.text("STDERR (last part):")
                    st.code(ffmpeg_debug["stderr"])
                if ffmpeg_debug.get("stdout"):
                    st.text("STDOUT (last part):")
                    st.code(ffmpeg_debug["stdout"])

    # Heatmap
    if frames is not None and video_model is not None:
        st.subheader("🔥 Model Attention (Video)")
        try:
            heatmap = make_gradcam_heatmap(video_model, frames[0], LAST_CONV_LAYER_NAME)
            overlay = overlay_heatmap(frames[0], heatmap)
            st.image(overlay, caption="Model attention regions", use_container_width=False)
        except Exception as e:
            st.warning(f"Grad-CAM failed. Layer name might be wrong ({LAST_CONV_LAYER_NAME}). Error: {e}")

    st.subheader("🧠 Why?")
    st.write(generate_explanation(video_score, audio_used, hybrid_used))

else:
    st.info("Upload a video and/or audio to get a prediction.")

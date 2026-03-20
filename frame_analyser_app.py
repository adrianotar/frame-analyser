#!/usr/bin/env python3
"""
Frame Analyser — Self-contained Streamlit app
Extracts best frames from any video for illustration inspiration.

Requirements:
    pip3 install streamlit yt-dlp --break-system-packages
    brew install ffmpeg
    ollama pull qwen3-vl:8b

Run:
    streamlit run frame_analyser_app.py
"""

import base64
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"

# Permanent output folder
OUTPUT_BASE = Path.home() / "Documents" / "FrameAnalyser"

MODELS = {
    "Ollama — qwen3-vl:8b (local, free)": {
        "type": "ollama",
        "id": "qwen3-vl:8b"
    },
    "Gemini 2.0 Flash (requires API key)": {
        "type": "gemini",
        "id": "gemini-2.0-flash"
    },
    "Claude Haiku (requires API key)": {
        "type": "anthropic",
        "id": "claude-haiku-4-5-20251001"
    },
}

JSON_FORMAT = """

Respond in this exact JSON format only, no other text:
{
  "composition": <score>,
  "light_and_colour": <score>,
  "emotional_resonance": <score>,
  "narrative": <score>,
  "total": <sum of four scores>,
  "rationale": "Composition: [explain compositional choices — rule of thirds, leading lines, negative space, symmetry, depth layers, foreground/background relationship]. Lens and technique: [infer the likely lens choice and camera technique — wide angle for vastness, telephoto compression, low angle for power, overhead for vulnerability, shallow depth of field for intimacy, long exposure, handheld for tension]. Light: [explain the quality, direction and colour of light — golden hour, backlight, hard shadows, diffused light, practical lights, colour temperature and what emotional effect each creates]. Shadow and contrast: [explain how shadows are used — silhouettes, chiaroscuro, high contrast vs flat light, what the shadows hide or reveal]. Mood: [explain what emotion this frame evokes and how the specific technical choices create that feeling]. Illustration value: [explain why this frame works as a reference — graphic quality, colour palette, contrast ratio, how it would translate into illustration]."
}"""

DEFAULT_PROMPT = """You are a cinematographer and photography director evaluating frames for use as illustration inspiration.

Analyse this frame on four criteria:

1. COMPOSITION (1-10)
   Rule of thirds, leading lines, negative space, visual balance. Does this frame feel intentionally composed?

2. LIGHT AND COLOUR (1-10)
   Quality of light, palette harmony, contrast, shadow. The kind of light a photographer waits hours for.

3. EMOTIONAL RESONANCE (1-10)
   Does this frame make you feel something? Tension, stillness, movement, intimacy, vastness.

4. NARRATIVE (1-10)
   Does this frame imply a story? Something happened before this moment. Something will happen after. The viewer wants to know what.

AUTOMATIC ZERO — score total 0 if:
- Excessive motion blur making the frame unusable as reference
- Transitional frame (mid-cut, fade, black frame)"""


# ── Helper functions ────────────────────────────────────────────────────────────

def make_output_dir(video_title: str) -> Path:
    """Create timestamped output folder in Documents/FrameAnalyser."""
    date = datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_title = "".join(c for c in video_title if c.isalnum() or c in " -_")[:40].strip()
    folder = OUTPUT_BASE / f"{safe_title}_{date}"
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    (folder / "top_frames").mkdir(parents=True, exist_ok=True)
    return folder


def download_video(url: str, output_dir: Path) -> Path:
    """Download video from URL using yt-dlp."""
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(output_dir / "video.%(ext)s"),
        "--no-playlist",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Download failed: {result.stderr}")

    video_files = list(output_dir.glob("video.*"))
    if not video_files:
        raise Exception("No video file found after download")
    return video_files[0]


def get_video_title(url: str) -> str:
    """Get video title from URL without downloading."""
    cmd = ["yt-dlp", "--get-title", "--no-playlist", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return "video"


def extract_frames(video_path: Path, output_dir: Path, interval: int) -> list[Path]:
    """Extract frames every N seconds using ffmpeg."""
    pattern = output_dir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        str(pattern),
        "-y", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffmpeg error: {result.stderr}")
    return sorted(output_dir.glob("frame_*.jpg"))


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyse_frame_ollama(image_path: Path, full_prompt: str, model_id: str, timeout: int) -> dict:
    payload = {
        "model": model_id,
        "prompt": full_prompt,
        "images": [encode_image(image_path)],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return parse_scores(result.get("response", ""))


def analyse_frame_gemini(image_path: Path, full_prompt: str, model_id: str, api_key: str, timeout: int) -> dict:
    image_b64 = encode_image(image_path)
    payload = {
        "contents": [{"parts": [
            {"text": full_prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
        ]}]
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return parse_scores(text)


def analyse_frame_anthropic(image_path: Path, full_prompt: str, model_id: str, api_key: str, timeout: int) -> dict:
    image_b64 = encode_image(image_path)
    payload = {
        "model": model_id,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
            {"type": "text", "text": full_prompt}
        ]}]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return parse_scores(result["content"][0]["text"])


def parse_scores(raw: str) -> dict:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        scores = json.loads(raw[start:end])
        return {
            "composition": scores.get("composition", 0),
            "light_and_colour": scores.get("light_and_colour", 0),
            "emotional_resonance": scores.get("emotional_resonance", 0),
            "narrative": scores.get("narrative", 0),
            "total": scores.get("total", 0),
            "rationale": scores.get("rationale", "No rationale provided")
        }
    raise ValueError("No JSON found in response")


def generate_report(top: list[dict], video_title: str, interval: int, model_label: str) -> str:
    """Generate markdown report."""
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟",
              "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

    lines = [
        f"# Frame Analysis Report",
        f"",
        f"**Video:** {video_title}  ",
        f"**Date:** {date}  ",
        f"**Model:** {model_label}  ",
        f"**Interval:** every {interval} seconds  ",
        f"**Purpose:** Illustration inspiration — cinematographic and photography eye  ",
        f"",
        f"---",
        f"",
        f"## Top {len(top)} Frames",
        f"",
    ]

    for rank, result in enumerate(top):
        lines += [
            f"### {medals[rank]} Rank {rank + 1} — Score {result['total']}/40 — {result['timestamp']}",
            f"",
            f"| Criterion | Score |",
            f"|---|---|",
            f"| Composition | {result['composition']}/10 |",
            f"| Light and colour | {result['light_and_colour']}/10 |",
            f"| Emotional resonance | {result['emotional_resonance']}/10 |",
            f"| Narrative | {result['narrative']}/10 |",
            f"| **Total** | **{result['total']}/40** |",
            f"",
            f"> {result['rationale']}",
            f"",
            f"**File:** `{Path(result['saved_path']).name}`",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"*Generated by Frame Analyser 🎬*",
    ]
    return "\n".join(lines)


# ── UI ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Frame Analyser",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Frame Analyser")
st.caption("Extract the best frames from any video for illustration inspiration")

st.divider()

video_url = st.text_input(
    "Video URL",
    placeholder="Paste a YouTube link — e.g. https://youtu.be/xxxxx"
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    interval = st.slider("Frame interval (seconds)", 1, 10, 2)
    top_n = st.slider("Top frames to extract", 3, 20, 5)

with col2:
    model_label = st.selectbox("Model", list(MODELS.keys()))
    timeout = st.slider("Timeout per frame (seconds)", 60, 300, 120)

model_config = MODELS[model_label]
api_key = None
if model_config["type"] != "ollama":
    api_key = st.text_input(
        f"API key for {model_label}",
        type="password",
        placeholder="Paste your API key here"
    )

st.divider()

prompt = st.text_area(
    "Analysis prompt",
    height=300,
    value=DEFAULT_PROMPT,
    help="Customise the creative criteria. The technical output format is handled automatically."
)

full_prompt = prompt + JSON_FORMAT

st.divider()

if st.button("🎬 Analyse", type="primary", use_container_width=True):

    if not video_url:
        st.error("Please paste a video URL first")
        st.stop()

    if model_config["type"] != "ollama" and not api_key:
        st.error(f"Please enter an API key for {model_label}")
        st.stop()

    try:
        # Step 1: Get video title and create output folder
        with st.status("Getting video info...") as status:
            video_title = get_video_title(video_url)
            output_dir = make_output_dir(video_title)
            status.update(label=f"Output folder: {output_dir.name}", state="complete")

        # Step 2: Download video
        with st.status("Downloading video...") as status:
            video_path = download_video(video_url, output_dir)
            status.update(label=f"Downloaded: {video_path.name}", state="complete")

        # Step 3: Extract frames
        with st.status("Extracting frames...") as status:
            frames = extract_frames(video_path, output_dir / "frames", interval)
            status.update(label=f"Extracted {len(frames)} frames", state="complete")

        if not frames:
            st.error("No frames extracted. Check the video.")
            st.stop()

        # Step 4: Analyse frames
        st.write(f"**Analysing {len(frames)} frames with {model_label}...**")
        progress = st.progress(0)
        results = []

        for i, frame_path in enumerate(frames):
            timestamp = i * interval
            minutes = timestamp // 60
            seconds = timestamp % 60
            time_label = f"{minutes:02d}:{seconds:02d}"

            try:
                if model_config["type"] == "ollama":
                    scores = analyse_frame_ollama(frame_path, full_prompt, model_config["id"], timeout)
                elif model_config["type"] == "gemini":
                    scores = analyse_frame_gemini(frame_path, full_prompt, model_config["id"], api_key, timeout)
                else:
                    scores = analyse_frame_anthropic(frame_path, full_prompt, model_config["id"], api_key, timeout)

                scores["timestamp"] = time_label
                scores["frame_path"] = str(frame_path)
                results.append(scores)

            except Exception as e:
                results.append({
                    "timestamp": time_label,
                    "frame_path": str(frame_path),
                    "composition": 0, "light_and_colour": 0,
                    "emotional_resonance": 0, "narrative": 0,
                    "total": 0, "rationale": f"Error: {e}"
                })

            progress.progress((i + 1) / len(frames))

        # Step 5: Save top frames
        top = sorted(results, key=lambda x: x["total"], reverse=True)[:top_n]
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟",
                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

        for rank, result in enumerate(top):
            src = Path(result["frame_path"])
            dst = output_dir / "top_frames" / f"rank_{rank+1:02d}_score{result['total']}_at{result['timestamp'].replace(':', 'm')}s.jpg"
            shutil.copy2(src, dst)
            result["saved_path"] = str(dst)

        # Step 6: Save report
        report_text = generate_report(top, video_title, interval, model_label)
        report_path = output_dir / "report.md"
        report_path.write_text(report_text)

        # Step 7: Show results
        st.divider()
        st.success(f"✅ Analysis complete — saved to Documents/FrameAnalyser/{output_dir.name}")
        st.caption(f"📁 {output_dir}")

        st.subheader(f"🏆 Top {top_n} frames")

        for rank, result in enumerate(top):
            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.image(result["frame_path"], use_container_width=True)
                # Download button for each frame
                with open(result["saved_path"], "rb") as f:
                    st.download_button(
                        label="⬇️ Download frame",
                        data=f.read(),
                        file_name=Path(result["saved_path"]).name,
                        mime="image/jpeg",
                        key=f"dl_{rank}"
                    )

            with col_info:
                st.markdown(f"### {medals[rank]} {result['timestamp']} — {result['total']}/40")
                st.markdown(f"*{result['rationale']}*")
                st.markdown(f"""
| Criterion | Score |
|---|---|
| Composition | {result['composition']}/10 |
| Light & colour | {result['light_and_colour']}/10 |
| Emotional resonance | {result['emotional_resonance']}/10 |
| Narrative | {result['narrative']}/10 |
""")
            st.divider()

        # Download full report
        st.download_button(
            label="📄 Download full report",
            data=report_text,
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

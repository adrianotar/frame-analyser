# 🎬 Frame Analyser

Extract the most visually compelling frames from any video 
for inspiration and visual learning.

Built with a cinematographer and photography eye — evaluating 
composition, light, emotional resonance, and narrative.

---

## What it does

- Paste a YouTube URL or local video file
- Extracts frames at a configurable interval
- Analyses each frame using AI vision models
- Returns the top N frames ranked by visual quality
- Saves frames + a full report to your Documents folder
- Customizable Prompt

---

## Requirements

- Python 3.10+
- [Streamlit](https://streamlit.io)
- [ffmpeg](https://ffmpeg.org)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Ollama](https://ollama.ai) with `qwen3-vl:8b` for local analysis
- Or a Gemini / Anthropic API key for cloud analysis

## Install
```bash
pip3 install streamlit yt-dlp
brew install ffmpeg
ollama pull qwen3-vl:8b
```

## Run
```bash
streamlit run frame_analyser_app.py
```

---

## Models supported

- **Ollama qwen3-vl:8b** — local, free, private
- **Gemini 2.0 Flash** — fast, cheap, requires API key
- **Claude Haiku** — high quality, requires API key

---

## Made by

Adriano — UX/Service Designer exploring AI tools for 
creative practice.

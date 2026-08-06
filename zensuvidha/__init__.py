"""ZenSuvidha OSS — a free, offline, pluggable voice-AI receptionist engine.

Pipeline:  telephony/mic → STT → LLM (+ Industry Pack) → TTS → caller
Everything runs locally: faster-whisper + Ollama + pyttsx3/Piper/Kokoro.
"""
__version__ = "0.1.0"

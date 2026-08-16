#!/usr/bin/env bash
# whisper.cpp — the same Whisper model, decoded on the GPU that is already in the box.
#
# WHY. STT is the dominant cost on the audio path (1070ms at 3s of speech, against
# 66ms for isolation and 34ms for the speaker gate) and every CPU-side lever has been
# measured and rejected: more CTranslate2 threads is WORSE on an M1 (auto 1691ms,
# 8 threads 2212ms, 10 threads 3798ms), and greedy decoding was not reliably faster.
#
# MEASURED on this repo, M1 Pro, small model, best of 2 warm runs:
#
#     clip     faster-whisper   whisper.cpp   speedup
#     3.5s            1562ms         833ms      1.87x
#     7.0s            1488ms         797ms      1.87x
#    10.5s            1519ms         876ms      1.73x
#
#   …and it is not paid for in accuracy: mean WER 0.25 vs 0.29 over the same four
#   clips, the difference being the Hindi one (0.29 vs 0.43).
#
# Still OPT-IN, because it needs a system package and a 490MB model that `git clone`
# does not bring, and this project's premise is that cloning and running works. Turn
# it on with `stt.provider: whisper_cpp`.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=models/whispercpp
MODEL=${1:-small}      # tiny | base | small | medium | large-v3

if ! command -v whisper-cli >/dev/null 2>&1; then
  echo "whisper-cli not found. Install it first:"
  echo "    brew install whisper-cpp          # macOS"
  echo "    # or build from https://github.com/ggml-org/whisper.cpp"
  exit 1
fi

mkdir -p "$DIR"
BASE=https://huggingface.co/ggerganov/whisper.cpp/resolve/main

if [ ! -f "$DIR/ggml-$MODEL.bin" ]; then
  echo "Downloading ggml-$MODEL.bin …"
  # …to .part and move on success. An interrupted 490MB download otherwise leaves a
  # truncated file that every later run SKIPS, and WhisperCppSTT only checks that the
  # path exists — so the broken model loads and whisper-cli aborts on every turn.
  curl -L --progress-bar -o "$DIR/ggml-$MODEL.bin.part" "$BASE/ggml-$MODEL.bin"
  mv "$DIR/ggml-$MODEL.bin.part" "$DIR/ggml-$MODEL.bin"
fi

# The VAD model must be whisper.cpp's own ggml build. Pointing --vad at one of the
# THREE Silero .onnx files this repo already ships does not error — the process
# ABORTS (SIGABRT), which from the server looks like a broken recogniser rather than
# a wrong file. Hence a separate download rather than reusing what is on disk.
if [ ! -f "$DIR/ggml-silero-v5.1.2.bin" ]; then
  echo "Downloading the ggml Silero VAD (0.9MB) …"
  curl -L --progress-bar -o "$DIR/ggml-silero-v5.1.2.bin.part" \
    https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
  mv "$DIR/ggml-silero-v5.1.2.bin.part" "$DIR/ggml-silero-v5.1.2.bin"
fi

echo
echo "Done. To use it, set in config.yaml:"
echo "    stt:"
echo "      provider: whisper_cpp"
echo "      whispercpp_model: $DIR/ggml-$MODEL.bin"
echo
echo "Compare the two on your own recordings before switching:"
echo "    python scripts/bench_stt.py --dir <folder of .wav + .txt>"

PACK ?= clinic
PY := .venv/bin/python

.PHONY: install run web test voice docker clean

install:            ## create venv + install core deps
	python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt
	@echo "Next:  ollama pull qwen2.5:3b"

run:                ## text CLI  (make run PACK=restaurant)
	$(PY) -m zensuvidha.cli --pack $(PACK)

web:                ## browser voice demo on :8000
	.venv/bin/uvicorn zensuvidha.server:app --host 0.0.0.0 --port 8000

test:               ## run the engine tests
	$(PY) -m pip install -q pytest && $(PY) -m pytest -q

voice:              ## install optional neural voices (piper/kokoro)
	$(PY) -m pip install -q -r requirements-voice.txt

clone:              ## install the voice-cloning stack (heavy ~4GB)
	$(PY) -m pip install -r requirements-clone.txt

docker:             ## run everything (Ollama + app) in containers
	docker compose up --build

clean:
	rm -rf .venv data __pycache__ zensuvidha/__pycache__ tests/__pycache__

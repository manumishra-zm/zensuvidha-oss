"""Local LLM via Ollama.

Robustness: retries with backoff, explicit timeouts, and `keep_alive` so the
model stays resident in RAM between turns (a big first-token latency win).
JSON-mode gives a small model a parseable {say, action} envelope.
Both blocking (`chat`) and streaming (`astream`) paths are provided.
"""
import json
import logging
import time

import httpx

log = logging.getLogger("zensuvidha.llm")


class OllamaError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, cfg: dict):
        self.model = cfg.get("model", "qwen3:4b")
        self.base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = cfg.get("temperature", 0.4)
        self.keep_alive = cfg.get("keep_alive", "30m")
        self.timeout = float(cfg.get("timeout", 120))
        self.retries = int(cfg.get("retries", 2))
        self.num_predict = int(cfg.get("num_predict", 200))
        self.num_ctx = int(cfg.get("num_ctx", 4096))   # small context = smaller KV cache = faster
        self.num_thread = cfg.get("num_thread")         # None = Ollama auto (all cores)
        # think: None = auto, True/False forces it. Left OFF for voice — Qwen3 (and
        # other reasoning models) default to a chain-of-thought that adds SECONDS of
        # dead air per turn. See _resolved_think().
        self.think = cfg.get("think", None)

    # ---- thinking-mode resolver --------------------------------------------
    def _resolved_think(self, model=None):
        """Whether to let the model 'think' (reason) before replying.

        A voice receptionist needs the answer immediately, so reasoning is turned
        off. `think` in config wins; when unset (None) we AUTO-disable it for
        reasoning models (qwen3, deepseek-r1, *-thinking) and send nothing for
        plain models like qwen2.5 / llama — passing `think` to those 400s in
        Ollama, so we must not.
        """
        if self.think is not None:
            return self.think
        m = (model or self.model).lower()
        if "qwen3" in m or "thinking" in m or "deepseek-r1" in m or "-r1" in m:
            return False
        return None

    # ---- payload helper -----------------------------------------------------
    def _payload(self, messages, force_json, stream, model=None, num_predict=None, num_ctx=None):
        p = {
            "model": model or self.model,     # per-session override → faster/smaller models
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            # num_predict is per-CALL: Indic scripts cost 3–4× more tokens per word than
            # English, so a budget tuned for English guillotines Hindi/Telugu replies
            # mid-word. The caller scales it to the language being spoken.
            "options": {"temperature": self.temperature,
                        "num_predict": int(num_predict or self.num_predict),
                        # num_ctx is per-CALL too: when the prompt overflows it, Ollama
                        # silently drops the OLDEST messages — the system prompt or the
                        # conversation — and the agent then loops or forgets the call.
                        "num_ctx": int(num_ctx or self.num_ctx)},
        }
        if self.num_thread:
            p["options"]["num_thread"] = int(self.num_thread)
        think = self._resolved_think(model)
        if think is not None:
            p["think"] = think  # Ollama >= 0.9 toggles reasoning on qwen3 etc.
        if force_json:
            p["format"] = "json"
        return p

    # ---- blocking chat (CLI / fallback) ------------------------------------
    def chat(self, messages, force_json=True, model=None, num_predict=None, meta=None,
             num_ctx=None) -> str:
        """`meta`, if given, is filled with {"finish_reason": "stop"|"length"} — the model's
        own report of whether it finished or ran out of tokens. The guard needs that to
        tell a genuinely cut-off reply from one that merely lacks a full stop."""
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(f"{self.base_url}/api/chat",
                               json=self._payload(messages, force_json, False, model, num_predict, num_ctx),
                               timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if meta is not None:
                    meta["finish_reason"] = data.get("done_reason") or "stop"
                return data["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < self.retries:
                    time.sleep(0.4 * (attempt + 1))
        raise OllamaError(f"Ollama chat failed after {self.retries + 1} tries: {last}")

    # ---- streaming chat (server) -------------------------------------------
    async def astream(self, messages, force_json=True, model=None, num_predict=None, meta=None,
                      num_ctx=None):
        """Yield content deltas as they arrive. Falls through on transient errors
        after retries by raising OllamaError (caller decides how to degrade).
        `meta` is filled with the finish reason when the stream completes."""
        payload = self._payload(messages, force_json, True, model, num_predict, num_ctx)
        last = None
        for attempt in range(self.retries + 1):
            started = False   # true once we've yielded a delta this attempt
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as c:
                    async with c.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            delta = obj.get("message", {}).get("content")
                            if delta:
                                started = True
                                yield delta
                            if obj.get("done"):
                                if meta is not None:
                                    meta["finish_reason"] = obj.get("done_reason") or "stop"
                                return
                return
            except Exception as e:  # noqa: BLE001
                last = e
                # Only retry if nothing was emitted yet. Retrying mid-stream would
                # re-run the generation from scratch and re-yield the beginning,
                # corrupting the consumer's buffer (duplicated/garbled reply).
                if started:
                    raise OllamaError(f"Ollama stream dropped mid-reply: {e}")
                if attempt < self.retries:
                    import asyncio
                    await asyncio.sleep(0.4 * (attempt + 1))
        raise OllamaError(f"Ollama stream failed after {self.retries + 1} tries: {last}")

    # ---- warmup / health ----------------------------------------------------
    def warmup(self, model=None, messages=None):
        """Load a model's weights into RAM so the first real turn isn't cold.

        `messages` should be the REAL system prompt this deployment will serve. Loading
        the weights is only half the cost: Ollama must also EVALUATE the prompt, and the
        clinic pack's is ~6,000 tokens because the whole knowledge base sits in a
        deliberately cache-stable prefix. Measured on this machine, cold vs warm:

            first turn   first_token 23546ms
            second turn  first_token 19782ms
            third turn   first_token   640ms   <- the KV cache finally holds the prefix

        Warming with "hi" loads the weights and caches a two-token prefix that no real
        turn shares, so the first caller paid the whole 23s — and worse, a later "hi"
        warmup EVICTS the real prefix and makes the next caller pay it again.
        """
        m = model or self.model
        try:
            # num_ctx MUST match what real turns send. Ollama sizes the KV cache from it
            # at load time, so warming at the default and then serving at 12288 makes the
            # FIRST real turn reload the whole model — measured 1.99s vs 0.26s. The warmup
            # was not just useless there, it cost an extra load.
            warm = {"model": m,
                    "messages": messages or [{"role": "user", "content": "hi"}],
                    "stream": False, "keep_alive": self.keep_alive,
                    "options": {"num_predict": 1, "num_ctx": self.num_ctx}}
            think = self._resolved_think(m)
            if think is not None:
                warm["think"] = think
            t0 = time.perf_counter()
            httpx.post(f"{self.base_url}/api/chat", json=warm, timeout=180)
            log.info("LLM warmed up (%s%s) in %.1fs", m,
                     ", prompt cached" if messages else "", time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM warmup skipped: %s", e)

    def health(self):
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return True, [m["name"] for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            return False, str(e)


class OpenAICompatLLM:
    """LLM over an OpenAI-compatible /v1 API — for **vLLM** (recommended on a GPU: continuous
    batching → far higher concurrency + lower latency than Ollama), or any OpenAI-style server.
    Same interface as OllamaLLM, so the rest of the app doesn't change."""

    def __init__(self, cfg: dict):
        self.model = cfg.get("model", "Qwen/Qwen2.5-14B-Instruct")
        base = cfg.get("base_url", "http://localhost:8001/v1").rstrip("/")
        self.base_url = base if base.endswith("/v1") else base + "/v1"
        self.api_key = cfg.get("api_key", "EMPTY")     # vLLM accepts any token
        self.temperature = cfg.get("temperature", 0.3)
        self.timeout = float(cfg.get("timeout", 120))
        self.retries = int(cfg.get("retries", 2))
        self.max_tokens = int(cfg.get("num_predict", 200))

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages, force_json, stream, model=None, num_predict=None, num_ctx=None):
        p = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            # per-call budget — see the note in OllamaLLM._payload
            "max_tokens": int(num_predict or self.max_tokens),
        }
        if force_json:
            p["response_format"] = {"type": "json_object"}   # vLLM guided-JSON
        return p

    def chat(self, messages, force_json=True, model=None, num_predict=None, meta=None,
             num_ctx=None) -> str:
        last = None   # num_ctx is fixed at vLLM launch (--max-model-len)
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                               json=self._payload(messages, force_json, False, model, num_predict),
                               timeout=self.timeout)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                if meta is not None:
                    meta["finish_reason"] = choice.get("finish_reason") or "stop"
                return choice["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < self.retries:
                    time.sleep(0.4 * (attempt + 1))
        raise OllamaError(f"vLLM chat failed after {self.retries + 1} tries: {last}")

    async def astream(self, messages, force_json=True, model=None, num_predict=None, meta=None,
                      num_ctx=None):
        payload = self._payload(messages, force_json, True, model, num_predict)
        last = None
        for attempt in range(self.retries + 1):
            started = False
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as c:
                    async with c.stream("POST", f"{self.base_url}/chat/completions",
                                        headers=self._headers(), json=payload) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                return
                            try:
                                choice = json.loads(data)["choices"][0]
                                if choice.get("finish_reason") and meta is not None:
                                    meta["finish_reason"] = choice["finish_reason"]
                                delta = choice["delta"].get("content")
                            except Exception:  # noqa: BLE001
                                continue
                            if delta:
                                started = True
                                yield delta
                return
            except Exception as e:  # noqa: BLE001
                last = e
                if started:
                    raise OllamaError(f"vLLM stream dropped mid-reply: {e}")
                if attempt < self.retries:
                    import asyncio
                    await asyncio.sleep(0.4 * (attempt + 1))
        raise OllamaError(f"vLLM stream failed after {self.retries + 1} tries: {last}")

    def warmup(self, model=None):
        try:
            self.chat([{"role": "user", "content": "hi"}], force_json=False, model=model)
            log.info("LLM warmed up (%s)", model or self.model)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM warmup skipped: %s", e)

    def health(self):
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            r.raise_for_status()
            return True, [m["id"] for m in r.json().get("data", [])]
        except Exception as e:  # noqa: BLE001
            return False, str(e)


def get_llm(cfg: dict):
    """Pick the LLM backend from config: 'ollama' (default, CPU/dev) or 'vllm'/'openai' (GPU)."""
    provider = (cfg or {}).get("provider", "ollama").lower()
    if provider in ("vllm", "openai", "openai_compat"):
        return OpenAICompatLLM(cfg)
    return OllamaLLM(cfg)


def measure_concurrency(llm, *, model=None, timeout_s: float = 25.0) -> tuple[bool, str]:
    """Can this backend actually answer two requests AT ONCE?

    Why this is measured rather than configured. Speculative reply — answering the guess
    while the caller may still be pausing — was built, shipped, and then turned off,
    because on one local Ollama it made turns 2.6x SLOWER (2339ms -> 6044ms): requests
    are serialised per model, so the real generation QUEUED BEHIND the guess instead of
    overlapping it. The feature is only ever a win where the server can genuinely run
    concurrent requests, and whether it can is a property of the machine in front of you,
    not of the config file.

    So: run one short generation, then two of them together, and compare. If the pair
    finishes in appreciably less than twice the single, they overlapped.

    Returns (can_overlap, why). Any failure is (False, reason) — the conservative
    answer, because being wrong here costs every turn on the call.
    """
    import concurrent.futures as _f
    import time
    msgs = [{"role": "user", "content": "Reply with the single word: ok"}]

    def once():
        t0 = time.time()
        try:
            llm.chat(msgs, force_json=False, model=model, num_predict=8)
        except Exception:  # noqa: BLE001
            return None
        return time.time() - t0

    try:
        llm.chat(msgs, force_json=False, model=model, num_predict=8)   # warm
        solo = once()
        if not solo:
            return False, "the model did not answer a probe"
        t0 = time.time()
        with _f.ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(once), ex.submit(once)]
            done = [f.result(timeout=timeout_s) for f in futures]
        pair = time.time() - t0
        if any(d is None for d in done):
            return False, "a concurrent probe failed"
        # 1.5x is the honest cut. Perfect overlap is 1.0x and perfect serialisation is
        # 2.0x; anything under 1.5 means real parallelism rather than scheduler noise.
        ratio = pair / solo if solo else 2.0
        ok = ratio < 1.5
        return ok, ("two requests overlap (%.2fx of one)" % ratio if ok
                    else "requests serialise (%.2fx of one — the guess would QUEUE in "
                         "front of the real turn)" % ratio)
    except Exception as e:  # noqa: BLE001
        return False, "probe failed (%s)" % e

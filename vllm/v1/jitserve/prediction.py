# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original QRF length estimator and request-scoped asynchronous updates.

Adapted from UIUC-MLSys/JITServe at 84584a1 (Apache-2.0). Models are loaded
only from operator-configured local artifacts, never from request payloads.
"""

import asyncio
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.utils.async_utils import cancel_task_threadsafe

logger = init_logger(__name__)


class QRFPredictor:
    def __init__(self, model_path: str, vectorizer_path: str):
        # Optional dependencies: ordinary inference does not import sklearn/QRF.
        from joblib import load

        self.model = load(model_path)
        self.vectorizer = load(vectorizer_path)
        self.metadata = {
            "model_sha256": self._digest(model_path),
            "vectorizer_sha256": self._digest(vectorizer_path),
            "quantile": 0.95,
            "feature_contract": "rendered_text_or_decoded_tokens_v1",
        }

    @staticmethod
    def _digest(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def predict(self, prompt: str) -> int:
        features = self.vectorizer.transform([prompt])
        result = self.model.predict(features.toarray(), quantiles=[0.95])
        value = float(result.reshape(-1)[0])
        if result.size != 1 or not math.isfinite(value) or int(value) <= 0:
            raise ValueError("QRF must produce one finite positive token estimate")
        # Match the original int conversion, not rounding/ceil or max_tokens.
        return int(value)


class PredictionRuntime:
    """Inference proceeds with the ingress estimate while QRF runs on CPU."""

    def __init__(
        self,
        predictor: Any,
        decode: Callable[[list[int]], str],
        update: Callable[[str, int, int], Awaitable[bool]],
        *,
        trace: bool = False,
    ):
        self.predictor = predictor
        self.decode = decode
        self.update = update
        self.trace = trace
        # The reference prediction server processes one request at a time.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qrf")
        self.tasks: dict[str, asyncio.Task] = {}
        self.counts: Counter[str] = Counter()

    @classmethod
    def from_config(cls, config, tokenizer, engine_core):
        additional = config.additional_config
        options = additional.get("jitserve", {}) if isinstance(additional, dict) else {}
        settings = options.get("prediction")
        if settings is None:
            return None
        from vllm.v1.jitserve.scheduler import JITServeScheduler

        if not issubclass(
            config.scheduler_config.get_scheduler_cls(), JITServeScheduler
        ):
            raise ValueError("QRF prediction requires JITServeScheduler")
        if tokenizer is None:
            raise ValueError("QRF prediction requires the generation tokenizer")
        predictor = QRFPredictor(**settings)

        async def update(request_id: str, output_len: int, version: int) -> bool:
            return await engine_core.call_utility_async(
                "jitserve_update_prediction", request_id, output_len, version
            )

        runtime = cls(
            predictor,
            lambda ids: tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            update,
            trace=options.get("trace_requests", False),
        )
        logger.info("JITServe QRF loaded %s", json.dumps(predictor.metadata))
        return runtime

    def start(
        self,
        request_id: str,
        prompt: str | None,
        token_ids: list[int] | None,
    ) -> None:
        if request_id in self.tasks:
            raise ValueError("A QRF task already owns this engine request ID")
        self.counts["submitted"] += 1
        self.tasks[request_id] = asyncio.create_task(
            self._run(request_id, prompt, token_ids)
        )

    async def _run(self, request_id, prompt, token_ids):
        started = time.monotonic()
        basis = "rendered_text" if prompt is not None else "decoded_prompt_tokens"

        def predict():
            text = prompt
            if text is None:
                if token_ids is None:
                    raise ValueError("QRF requires text or prompt token IDs")
                text = self.decode(token_ids)
            return self.predictor.predict(text)

        try:
            output_len = await asyncio.get_running_loop().run_in_executor(
                self.executor, predict
            )
            accepted = await self.update(request_id, output_len, 1)
            self._record(
                request_id,
                "applied" if accepted else "stale_or_finished",
                output_len=output_len,
                version=1,
                feature_basis=basis,
                elapsed_seconds=time.monotonic() - started,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Error text can include private feature text. Record only its type.
            self._record(request_id, "failed", error_type=type(exc).__name__)
            logger.error(
                "JITServe QRF failed for %s (%s); ingress estimate retained",
                request_id,
                type(exc).__name__,
            )
        finally:
            if self.tasks.get(request_id) is asyncio.current_task():
                del self.tasks[request_id]

    def _record(self, request_id: str, outcome: str, **metadata) -> None:
        self.counts[outcome] += 1
        if self.trace:
            logger.info(
                "JITServe prediction %s",
                json.dumps(
                    {"engine_request_id": request_id, "outcome": outcome, **metadata}
                ),
            )

    def finish(self, request_id: str, outcome: str = "finished") -> None:
        task = self.tasks.pop(request_id, None)
        if task is not None:
            self._record(request_id, outcome)
            # Cancels queued work/result delivery. A running sklearn call cannot
            # be interrupted; its eventual result is discarded without an RPC.
            cancel_task_threadsafe(task)

    def shutdown(self) -> None:
        for request_id in list(self.tasks):
            self.finish(request_id, "shutdown")
        self.executor.shutdown(wait=False, cancel_futures=True)

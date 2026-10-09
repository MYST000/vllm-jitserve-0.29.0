# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QRF numerical fidelity and nonblocking request lifecycle contracts."""

import ast
import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from vllm.v1.jitserve.prediction import PredictionRuntime, QRFPredictor


@pytest.fixture(scope="module")
def qrf():
    assets = os.environ.get("JITSERVE_QRF_ASSETS")
    if not assets:
        pytest.skip("Set JITSERVE_QRF_ASSETS for original pretrained QRF evidence")
    assert assets is not None
    name = "0_qrf_lmsys_chat_llama3_8b.pkl"
    return QRFPredictor(
        str(Path(assets) / "qrf_model" / name),
        str(Path(assets) / "qrf_vectorizer" / name),
    )


def test_pretrained_qrf_matches_original_function(qrf):
    root = os.environ.get("JITSERVE_REFERENCE")
    if not root:
        pytest.skip("Set JITSERVE_REFERENCE for original function differential")
    assert root is not None
    path = Path(root) / "jitserve/request_analyzer/prediction.py"
    tree = ast.parse(path.read_text())
    module = ast.Module(
        body=[
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "predict"
        ],
        type_ignores=[],
    )
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    prompts = [
        "What is the capital of France?",
        "Write a Python function to merge two sorted lists.",
        "解释量子回归森林的输出长度预测。",
        "<|im_start|>user\nAdd 12 and 30.<|im_end|>\n<|im_start|>assistant\n",
    ]
    expected = namespace["predict"](qrf.vectorizer, qrf.model, prompts)
    assert [qrf.predict(prompt) for prompt in prompts] == expected


@pytest.mark.parametrize("values", [[0], [np.nan], [np.inf], [1, 2]])
def test_invalid_qrf_result_is_an_error_not_a_substitute_estimate(values):
    qrf = QRFPredictor.__new__(QRFPredictor)
    qrf.model = NS(predict=lambda *a, **kw: np.array(values))
    qrf.vectorizer = NS(transform=lambda _: NS(toarray=lambda: np.zeros((1, 2))))
    with pytest.raises(ValueError):
        qrf.predict("private prompt")


@pytest.mark.parametrize(
    "outcome", ["applied", "stale_or_finished", "finished", "cancelled"]
)
def test_prediction_never_blocks_and_discards_terminal_results(outcome):
    async def scenario():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def predict(prompt):
            assert prompt == "private prompt"
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return 73

        update = AsyncMock(return_value=outcome == "applied")
        runtime = PredictionRuntime(
            NS(predict=predict), lambda _: "private prompt", update
        )
        try:
            runtime.start("actual-engine-id", None, [42])
            task = runtime.tasks["actual-engine-id"]
            await asyncio.wait_for(entered.wait(), 5)
            # The event loop continues while the CPU predictor is held.
            assert not task.done()
            update.assert_not_called()
            if outcome in {"finished", "cancelled"}:
                runtime.finish("actual-engine-id", outcome)
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if outcome in {"finished", "cancelled"}:
                update.assert_not_called()
            else:
                update.assert_awaited_once_with("actual-engine-id", 73, 1)
            assert runtime.counts[outcome] == 1
            assert not runtime.tasks
        finally:
            release.set()
            runtime.shutdown()

    asyncio.run(scenario())


def test_prediction_failure_is_visible_without_logging_private_input(caplog):
    async def scenario():
        def fail(_):
            raise RuntimeError("secret prompt text")

        update = AsyncMock()
        runtime = PredictionRuntime(NS(predict=fail), str, update, trace=True)
        try:
            runtime.start("engine-id", "secret prompt text", None)
            await runtime.tasks["engine-id"]
            assert runtime.counts["failed"] == 1
            assert not runtime.tasks
            update.assert_not_called()
        finally:
            runtime.shutdown()

    asyncio.run(scenario())
    assert "RuntimeError" in caplog.text
    assert "secret prompt text" not in caplog.text


def test_disabled_prediction_does_not_load_assets():
    assert PredictionRuntime.from_config(NS(additional_config={}), None, None) is None


@pytest.mark.parametrize("submission_error", [False, True])
def test_async_llm_starts_prediction_only_after_engine_submission(submission_error):
    from vllm.v1.engine.async_llm import AsyncLLM

    async def scenario():
        llm = AsyncLLM.__new__(AsyncLLM)
        llm.log_requests = False
        llm.output_processor = Mock()
        llm.output_processor.has_request.return_value = True
        llm.jitserve_prediction = Mock()

        async def submit(request):
            llm.jitserve_prediction.start.assert_not_called()
            if submission_error:
                raise RuntimeError("submission failed")

        llm.engine_core = NS(add_request_async=submit, shutdown=Mock())
        request = NS(request_id="randomized-engine-id", prompt_token_ids=[1, 2])
        if submission_error:
            with pytest.raises(RuntimeError, match="submission failed"):
                await llm._add_request(request, "text", None, 0, Mock())
            llm.jitserve_prediction.start.assert_not_called()
        else:
            await llm._add_request(request, "text", None, 0, Mock())
            llm.jitserve_prediction.start.assert_called_once_with(
                "randomized-engine-id", "text", [1, 2]
            )
        llm.shutdown()

    asyncio.run(scenario())

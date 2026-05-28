"""Small API adapters used by agents.

The LLM adapter targets OpenAI-compatible chat completion APIs. Other providers
can be used by pointing base_url at their compatible endpoint.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import ImageConfig, LLMConfig
from .storage.paths import project_dir


class ApiBackendError(RuntimeError):
    """Raised when a configured API backend cannot produce a usable result."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = 4096
    response_format: dict[str, str] = Field(default_factory=lambda: {"type": "json_object"})


class ChatCompletionResponse(BaseModel):
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0

    def json_content(self) -> dict[str, Any]:
        try:
            value = json.loads(self.content)
        except json.JSONDecodeError as exc:
            raise ApiBackendError(f"LLM response is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ApiBackendError("LLM response JSON must be an object")
        return value


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return self.config.is_available

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            raise ApiBackendError("LLM API is not configured")

        request_payload = ChatCompletionRequest(
            model=self.config.model,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(
                    role="user",
                    content=json.dumps(user_payload, ensure_ascii=False),
                ),
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ).model_dump()

        started = time.time()
        raw = self._post_json(self._completion_url(), request_payload)
        choices = raw.get("choices") or []
        if not choices:
            raise ApiBackendError("LLM response does not contain choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        response = ChatCompletionResponse(
            content=content,
            raw=raw,
            latency_ms=int((time.time() - started) * 1000),
        )
        return response.json_content()

    def _completion_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ApiBackendError(f"LLM API request failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise ApiBackendError("LLM API returned a non-object response")
        return raw


class ImageGenerationRequest(BaseModel):
    project_id: str
    kind: str
    prompt: str
    negative_prompt: str = ""
    width: int = 768
    height: int = 1024
    seed: int | None = None
    workflow_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageGenerationResponse(BaseModel):
    image_path: str = ""
    backend: str
    model_version: str = ""
    seed: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class ComfyUIImageClient:
    """Minimal ComfyUI API wrapper.

    It submits a workflow to /prompt, polls /history/{prompt_id}, downloads the
    first generated image through /view, and stores it in the project directory.
    """

    def __init__(self, config: ImageConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return self.config.is_available and self.config.provider == "comfyui"

    def submit(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        if not self.available:
            raise ApiBackendError("Image API is not configured")
        workflow_path = Path(request.workflow_path)
        if not workflow_path.is_absolute():
            workflow_path = Path.cwd() / workflow_path
        if not workflow_path.exists():
            raise ApiBackendError(f"Workflow file not found: {workflow_path}")

        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict):
            raise ApiBackendError(f"Workflow file must contain a JSON object: {workflow_path}")
        workflow = self._apply_common_overrides(workflow, request)
        payload = {
            "prompt": workflow,
            "client_id": f"comicweaver-{request.project_id}",
            "extra_data": {
                "comicweaver": request.model_dump(),
            },
        }
        raw = self._post_json(f"{self.config.server_url.rstrip('/')}/prompt", payload)
        prompt_id = raw.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ApiBackendError(f"ComfyUI did not return a prompt_id: {raw}")
        history = self._wait_for_history(prompt_id)
        image_info = self._first_output_image(history)
        image_path = self._download_image(image_info, request)
        return ImageGenerationResponse(
            image_path=image_path,
            backend=self.config.provider,
            model_version=self.config.model,
            seed=request.seed or 0,
            raw={"prompt": raw, "history": history},
        )

    def _apply_common_overrides(
        self,
        workflow: dict[str, Any],
        request: ImageGenerationRequest,
    ) -> dict[str, Any]:
        text_node_index = 0
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue

            if class_type == "CLIPTextEncode" and "text" in inputs:
                inputs["text"] = request.prompt if text_node_index == 0 else request.negative_prompt
                text_node_index += 1
            elif class_type in {"KSampler", "KSamplerAdvanced"}:
                if request.seed is not None and "seed" in inputs:
                    inputs["seed"] = request.seed
            elif class_type in {"EmptyLatentImage", "EmptySD3LatentImage", "LatentImage"}:
                if "width" in inputs:
                    inputs["width"] = request.width
                if "height" in inputs:
                    inputs["height"] = request.height

            if self.config.model and self.config.model != "placeholder":
                for key in ("ckpt_name", "model_name", "unet_name"):
                    if key in inputs:
                        inputs[key] = self.config.model
        return workflow

    def _wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        url = f"{self.config.server_url.rstrip('/')}/history/{urllib.parse.quote(prompt_id)}"
        deadline = time.time() + self.config.timeout_seconds
        last: dict[str, Any] = {}
        while time.time() < deadline:
            history = self._get_json(url)
            if isinstance(history, dict):
                last = history
                item = history.get(prompt_id)
                if isinstance(item, dict):
                    status = item.get("status")
                    if isinstance(status, dict) and status.get("status_str") == "error":
                        raise ApiBackendError(f"ComfyUI generation failed: {status}")
                    outputs = item.get("outputs")
                    if isinstance(outputs, dict) and self._first_output_image(item):
                        return item
            time.sleep(1.0)
        raise ApiBackendError(f"Timed out waiting for ComfyUI prompt {prompt_id}: {last}")

    def _first_output_image(self, history_item: dict[str, Any]) -> dict[str, Any]:
        outputs = history_item.get("outputs")
        if not isinstance(outputs, dict):
            return {}
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            images = output.get("images")
            if isinstance(images, list) and images:
                image = images[0]
                if isinstance(image, dict) and image.get("filename"):
                    return image
        return {}

    def _download_image(
        self,
        image_info: dict[str, Any],
        request: ImageGenerationRequest,
    ) -> str:
        filename = str(image_info.get("filename", ""))
        if not filename:
            raise ApiBackendError("ComfyUI history did not contain an output image")
        query = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": str(image_info.get("subfolder", "")),
            "type": str(image_info.get("type", "output")),
        })
        url = f"{self.config.server_url.rstrip('/')}/view?{query}"
        data = self._get_bytes(url)

        output_dir = project_dir(request.project_id) / "comfyui" / request.kind
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            request.metadata.get("panel_id")
            or request.metadata.get("char_id")
            or Path(filename).stem
        )
        suffix = Path(filename).suffix or ".png"
        path = output_dir / f"{stem}{suffix}"
        path.write_bytes(data)
        return str(path)

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ApiBackendError(f"Image API request failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise ApiBackendError("Image API returned a non-object response")
        return raw

    def _get_json(self, url: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=self.config.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ApiBackendError(f"Image API request failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise ApiBackendError("Image API returned a non-object response")
        return raw

    def _get_bytes(self, url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=self.config.timeout_seconds) as resp:
                return resp.read()
        except urllib.error.URLError as exc:
            raise ApiBackendError(f"Image download failed: {exc}") from exc

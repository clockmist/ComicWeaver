"""Small API adapters used by agents.

The LLM adapter targets OpenAI-compatible chat completion APIs. Other providers
can be used by pointing base_url at their compatible endpoint.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import ImageConfig, LLMConfig, VLMConfig
from .storage.paths import project_dir

logger = logging.getLogger("comicweaver.comfyui")
if not logger.handlers:
    # 配置 file handler 写入项目根目录下的 comfyui_requests.log
    log_path = Path.cwd() / "comfyui_requests.log"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    # 防止向 root logger 重复传播
    logger.propagate = False


class ApiBackendError(RuntimeError):
    """Raised when a configured API backend cannot produce a usable result."""


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]  # str for text; list for vision content array


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return a base64 data-URI string."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    # Try to detect mime type from extension
    ext = Path(image_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    return f"data:{mime};base64,{b64}"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = 4096
    response_format: dict[str, str] | None = Field(default_factory=lambda: {"type": "json_object"})


class ChatCompletionResponse(BaseModel):
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0

    def json_content(self) -> dict[str, Any]:
        text = self.content.strip()
        # 移除常见 markdown 代码块包装（某些模型即使要求纯 JSON 也会加）
        if text.startswith("```"):
            # 找到第一个换行后的内容
            first_newline = text.find("\n")
            if first_newline > 0:
                text = text[first_newline + 1:]
            # 移除结尾的 ```
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApiBackendError(
                f"LLM response is not valid JSON: {exc}. "
                f"Raw content (first 200 chars): {self.content[:200]}"
            ) from exc
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

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(
                role="user",
                content=json.dumps(user_payload, ensure_ascii=False),
            ),
        ]

        last_error: Exception | None = None
        # 外层：先尝试 json_object，失败后回退到无格式约束
        for outer_attempt in range(2):
            req = ChatCompletionRequest(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            if outer_attempt == 0:
                req.response_format = {"type": "json_object"}
            else:
                req.response_format = None  # type: ignore[assignment]

            # 内层：对 5xx / 网络错误做指数退避重试
            request_payload = req.model_dump(exclude_none=True)
            for inner_attempt in range(3):
                try:
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
                except ApiBackendError as exc:
                    last_error = exc
                    error_str = str(exc).lower()
                    # 仅对服务端/网络错误重试（4xx 客户端错误不重试）
                    is_retryable = (
                        "500" in error_str
                        or "502" in error_str
                        or "503" in error_str
                        or "504" in error_str
                        or "timeout" in error_str
                        or "timed out" in error_str
                        or "connection" in error_str
                    )
                    if is_retryable and inner_attempt < 2:
                        delay = (2 ** inner_attempt) * 1.0
                        time.sleep(delay)
                        continue
                    # 不可重试的错误，跳出内层
                    break

        raise last_error or ApiBackendError("LLM call failed after retries")

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


def _write_generation_log(request: ImageGenerationRequest, workflow_path: Path) -> None:
    """Write the exact seed + prompt to a generation log file in the project dir.

    This makes it easy for the user to copy-paste parameters for manual ComfyUI testing.
    """
    out_dir = project_dir(request.project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generation_log.txt"

    panel_id = request.metadata.get("panel_id") or request.metadata.get("char_id") or "?"
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    entry = (
        f"{'=' * 80}\n"
        f"[{ts}]  kind={request.kind}  id={panel_id}\n"
        f"workflow: {workflow_path}\n"
        f"seed:     {request.seed}\n"
        f"size:     {request.width} x {request.height}\n"
        f"{'─' * 80}\n"
        f"POSITIVE:\n{request.prompt}\n"
        f"{'─' * 80}\n"
        f"NEGATIVE:\n{request.negative_prompt}\n"
        f"{'=' * 80}\n\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


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

        # --- 记录完整 ComfyUI 生图参数（文件日志）---
        logger.info(
            "ComfyUI 生图请求 [%s] seed=%s %s×%s\n"
            "  workflow: %s\n"
            "  positive: %s\n"
            "  negative: %s\n"
            "  metadata: %s",
            request.kind, request.seed, request.width, request.height,
            workflow_path, request.prompt, request.negative_prompt,
            request.metadata,
        )

        # --- 写入项目目录下的生成参数日志（方便手动复现）---
        _write_generation_log(request, workflow_path)

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

        logger.info(
            "ComfyUI 生图完成 [%s] seed=%s path=%s",
            request.kind, request.seed, image_path,
        )

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


# ============================================================================
# Standalone VLM utilities (independent of OpenAICompatibleLLMClient)
# ============================================================================


def _vlm_completion_url(base_url: str) -> str:
    """Build the /chat/completions URL from a base URL."""
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _vlm_post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    """POST JSON to an OpenAI-compatible endpoint, returning the parsed response."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ApiBackendError(f"VLM API request failed: {exc}") from exc
    if not isinstance(raw, dict):
        raise ApiBackendError("VLM API returned a non-object response")
    return raw


def detect_faces_vlm_api(image_path: str, config: VLMConfig) -> list[dict[str, Any]]:
    """Use a vision-capable LLM to find character faces in a panel image.

    Takes a **VLMConfig** (separate from LLMConfig).  Returns a list of dicts:
        [{"char_name": "林染", "bbox": {"x": 0.3, "y": 0.2, "w": 0.15, "h": 0.2}}, ...]

    All bbox values are normalised [0, 1] relative to image dimensions.
    Returns an empty list if VLM is unavailable, fails, or finds nothing.
    """
    if not config.is_available:
        return []

    data_uri = encode_image_base64(image_path)

    system_prompt = (
        "You are a precise visual analysis tool. Your ONLY job is to detect "
        "character faces in a manga/comic panel image. Output valid JSON.\n\n"
        "RULES:\n"
        "1. Identify every visible character face.\n"
        "2. For each face, return a normalised bounding box: x, y, w, h in [0, 1].\n"
        "   x=0 means left edge, x=1 means right edge. y=0 means top, y=1 means bottom.\n"
        "3. If you can guess which character it is, include char_name.\n"
        "4. If NO faces are visible, return an empty list.\n"
        "5. Output ONLY the JSON array, no markdown, no extra text.\n"
        'Example: [{"char_name": "hero", "bbox": {"x": 0.3, "y": 0.2, "w": 0.15, "h": 0.25}}]'
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Find all character faces in this panel. Return JSON array."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        ),
    ]

    req = ChatCompletionRequest(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    req.response_format = None  # no json_object for vision models

    request_payload = req.model_dump(exclude_none=True)
    try:
        started = time.time()
        raw = _vlm_post_json(
            _vlm_completion_url(config.base_url),
            request_payload,
            config.api_key,
            config.timeout_seconds,
        )
        choices = raw.get("choices") or []
        if not choices:
            return []
        content = (choices[0].get("message") or {}).get("content") or ""
        response = ChatCompletionResponse(
            content=content, raw=raw,
            latency_ms=int((time.time() - started) * 1000),
        )
        result = response.json_content()
        if isinstance(result, list):
            return result
        for v in result.values():
            if isinstance(v, list):
                return v
        return []
    except Exception:
        return []  # graceful fallback — VLM failure should never crash

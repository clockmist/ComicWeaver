"""Configuration loading tests."""
from pathlib import Path

from comicweaver.api import ComfyUIImageClient, ImageGenerationRequest
from comicweaver.config import reload_config


def test_default_config_has_local_fallback(monkeypatch):
    monkeypatch.delenv("COMICWEAVER_CONFIG", raising=False)
    monkeypatch.delenv("COMICWEAVER_LLM_ENABLED", raising=False)

    cfg = reload_config("__missing__.yaml")

    assert cfg.runtime.fallback_to_local is True
    assert cfg.llm.enabled is False
    assert cfg.image.fallback_to_placeholder is True


def test_config_env_overrides(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "comicweaver.yaml"
    config_path.write_text(
        "llm:\n  provider: local\n  enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COMICWEAVER_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("COMICWEAVER_LLM_ENABLED", "true")
    monkeypatch.setenv("COMICWEAVER_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("COMICWEAVER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("COMICWEAVER_LLM_MODEL", "json-model")

    cfg = reload_config(str(config_path))

    assert cfg.llm.provider == "openai_compatible"
    assert cfg.llm.enabled is True
    assert cfg.llm.is_available is True


def test_comfyui_common_overrides(monkeypatch):
    monkeypatch.setenv("COMICWEAVER_IMAGE_ENABLED", "true")
    monkeypatch.setenv("COMICWEAVER_IMAGE_PROVIDER", "comfyui")
    monkeypatch.setenv("COMICWEAVER_IMAGE_MODEL", "comic.safetensors")
    cfg = reload_config("__missing__.yaml")
    client = ComfyUIImageClient(cfg.image)
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "old positive"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "old negative"}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 512, "height": 512}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 1}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": "old_unet.safetensors"}},
    }

    patched = client._apply_common_overrides(
        workflow,
        ImageGenerationRequest(
            project_id="test",
            kind="panel",
            prompt="new positive",
            negative_prompt="new negative",
            width=768,
            height=1024,
            seed=42,
        ),
    )

    assert patched["1"]["inputs"]["ckpt_name"] == "comic.safetensors"
    assert patched["2"]["inputs"]["text"] == "new positive"
    assert patched["3"]["inputs"]["text"] == "new negative"
    assert patched["4"]["inputs"]["width"] == 768
    assert patched["4"]["inputs"]["height"] == 1024
    assert patched["5"]["inputs"]["seed"] == 42
    assert patched["6"]["inputs"]["unet_name"] == "comic.safetensors"


def test_comfyui_output_image_download(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = reload_config("__missing__.yaml")
    client = ComfyUIImageClient(cfg.image)
    image_info = {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "type": "output",
    }
    monkeypatch.setattr(client, "_get_bytes", lambda _url: b"png-bytes")

    path = client._download_image(
        image_info,
        ImageGenerationRequest(
            project_id="proj",
            kind="panel",
            prompt="prompt",
            metadata={"panel_id": "page_001_p01"},
        ),
    )

    assert Path(path).read_bytes() == b"png-bytes"
    assert path.endswith("projects/proj/comfyui/panel/page_001_p01.png")

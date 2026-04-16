import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import requests
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OFFLOAD_DIR = os.path.join(ROOT_DIR, "logs", "offload")


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger(service_name: str) -> logging.Logger:
    config = load_config()
    log_file = config["logging"]["file"]
    if not os.path.isabs(log_file):
        log_file = os.path.join(ROOT_DIR, log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=config["logging"]["level"],
        filename=log_file,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(service_name)


def safe_cuda_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "devices": [],
    }
    if not summary["cuda_available"]:
        return summary

    try:
        summary["cuda_device_count"] = torch.cuda.device_count()
        for index in range(summary["cuda_device_count"]):
            device_info: Dict[str, Any] = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
            }
            try:
                props = torch.cuda.get_device_properties(index)
                device_info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
            try:
                device_info["memory_allocated_gb"] = round(torch.cuda.memory_allocated(index) / (1024 ** 3), 3)
                device_info["memory_reserved_gb"] = round(torch.cuda.memory_reserved(index) / (1024 ** 3), 3)
            except Exception:
                pass
            summary["devices"].append(device_info)
    except Exception:
        pass

    return summary


def model_runtime_summary(model: Any, model_name: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "model_name": model_name,
        "cuda": safe_cuda_summary(),
        "has_hf_device_map": hasattr(model, "hf_device_map"),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "model_device": str(getattr(model, "device", "unknown")),
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "offload_folder": OFFLOAD_DIR if os.path.isdir(OFFLOAD_DIR) else None,
    }
    return summary


class MistralRuntime:
    def __init__(self, model_name: str, logger: logging.Logger) -> None:
        self.model_name = model_name
        self.logger = logger
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        self._resolved_openai_model: Optional[str] = None
        self._available_openai_models: Optional[list[str]] = None
        config = load_config()
        inference = config.get("inference", {})
        self.backend = inference.get("backend", "transformers")
        self.base_url = inference.get("base_url", "").rstrip("/")
        self.api_key = inference.get("api_key", "lm-studio")
        self.inference_model_name = inference.get("model_name")
        self.auto_discover_model = bool(inference.get("auto_discover_model", True))
        self.request_timeout = int(inference.get("timeout_seconds", 600))
        self.kv_cache_enabled = bool(inference.get("kv_cache", True))
        self.cache_implementation = inference.get("cache_implementation")

    def _build_model_kwargs(self) -> Dict[str, Any]:
        config = load_config()
        quantization = config.get("quantization", {})
        use_cuda = torch.cuda.is_available()
        os.makedirs(OFFLOAD_DIR, exist_ok=True)
        kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "offload_folder": OFFLOAD_DIR,
            "offload_state_dict": True,
        }

        if use_cuda:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            if (
                quantization.get("enable")
                and int(quantization.get("bits", 0)) == 4
                and BitsAndBytesConfig is not None
            ):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        else:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16

        return kwargs

    def runtime_report(self) -> Dict[str, Any]:
        report = {
            "model_name": self.model_name,
            "resolved_openai_model": self._resolved_openai_model,
            "available_openai_models": self._available_openai_models,
            "load_error": self._load_error,
            "backend": self.backend,
            "kv_cache_enabled": self.kv_cache_enabled,
            "cache_implementation": self.cache_implementation,
            "cuda": safe_cuda_summary(),
            "offload_folder": OFFLOAD_DIR,
            "bitsandbytes_available": BitsAndBytesConfig is not None,
        }
        if self._model is not None:
            report.update(model_runtime_summary(self._model, self.model_name))
        return report

    def _normalize_openai_base_url(self) -> str:
        if not self.base_url:
            return ""
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _list_openai_models(self) -> list[str]:
        base_url = self._normalize_openai_base_url()
        if not base_url:
            return []

        url = f"{base_url}/v1/models"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.get(url, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
        self._available_openai_models = models
        return models

    def _resolve_openai_model_name(self) -> str:
        if self._resolved_openai_model:
            return self._resolved_openai_model

        preferred = (self.inference_model_name or self.model_name or "").strip()
        if not self.auto_discover_model:
            self._resolved_openai_model = preferred
            return self._resolved_openai_model

        try:
            available_models = self._list_openai_models()
        except Exception as exc:
            self.logger.warning(
                "LM Studio model discovery failed for '%s': %s. Falling back to configured model name.",
                preferred,
                exc,
            )
            self._resolved_openai_model = preferred
            return self._resolved_openai_model

        if preferred and preferred in available_models:
            self._resolved_openai_model = preferred
            return self._resolved_openai_model

        if self.inference_model_name and available_models:
            self.logger.warning(
                "Configured LM Studio model '%s' was not found. Using first available model '%s' instead.",
                preferred,
                available_models[0],
            )
            self._resolved_openai_model = available_models[0]
            return self._resolved_openai_model

        if available_models:
            self.logger.info(
                "Configured model '%s' not exposed by LM Studio. Using discovered model '%s'.",
                preferred,
                available_models[0],
            )
            self._resolved_openai_model = available_models[0]
            return self._resolved_openai_model

        self._resolved_openai_model = preferred
        return self._resolved_openai_model

    def load(self) -> bool:
        if self.backend == "openai_compatible":
            return True
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            self.logger.info("Loading model '%s'", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self._build_model_kwargs(),
            )
            if hasattr(self._model, "config"):
                self._model.config.use_cache = self.kv_cache_enabled
            self._model.eval()
            self.logger.info("Model '%s' loaded successfully", self.model_name)
            self.logger.info("Model runtime summary: %s", json.dumps(self.runtime_report(), default=str))
            return True
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            self.logger.exception("Failed to load model '%s': %s", self.model_name, exc)
            return False

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        if self.backend == "openai_compatible":
            return self._generate_via_openai_compatible(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        if not self.load():
            raise RuntimeError(self._load_error or f"Unable to load {self.model_name}")

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        model_device = getattr(self._model, "device", torch.device("cpu"))
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        do_sample = temperature > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 1e-5),
            "do_sample": do_sample,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": self.kv_cache_enabled,
        }
        if self.cache_implementation:
            generation_kwargs["cache_implementation"] = self.cache_implementation

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate_via_openai_compatible(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("Inference backend is openai_compatible but no base_url is configured")

        base_url = self._normalize_openai_base_url()
        model_name = self._resolve_openai_model_name()
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self.logger.exception("OpenAI-compatible inference failed for '%s': %s", model_name, exc)
            raise


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


class TextClassifierRuntime:
    def __init__(self, model_name: str, logger: logging.Logger) -> None:
        self.model_name = model_name
        self.logger = logger
        self._classifier = None
        self._load_error: Optional[str] = None

    def load(self) -> bool:
        if self._classifier is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            device = 0 if torch.cuda.is_available() else -1
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
                device=device,
            )
            self.logger.info("Classifier '%s' loaded successfully", self.model_name)
            self.logger.info(
                "Classifier runtime summary: %s",
                json.dumps(
                    {
                        "model_name": self.model_name,
                        "pipeline_device": device,
                        "cuda": safe_cuda_summary(),
                    },
                    default=str,
                ),
            )
            return True
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            self.logger.exception("Failed to load classifier '%s': %s", self.model_name, exc)
            return False

    def classify(self, text: str):
        if not self.load():
            raise RuntimeError(self._load_error or f"Unable to load {self.model_name}")
        return self._classifier(text)


class LocalTransformersRuntime:
    def __init__(self, model_name: str, logger: logging.Logger) -> None:
        self.model_name = model_name
        self.logger = logger
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        config = load_config()
        inference = config.get("inference", {})
        self.kv_cache_enabled = bool(inference.get("kv_cache", True))
        self.cache_implementation = inference.get("cache_implementation")

    def _build_model_kwargs(self) -> Dict[str, Any]:
        config = load_config()
        quantization = config.get("quantization", {})
        use_cuda = torch.cuda.is_available()
        os.makedirs(OFFLOAD_DIR, exist_ok=True)
        kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "offload_folder": OFFLOAD_DIR,
            "offload_state_dict": True,
        }

        if use_cuda:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            if (
                quantization.get("enable")
                and int(quantization.get("bits", 0)) == 4
                and BitsAndBytesConfig is not None
            ):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        else:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16

        return kwargs

    def runtime_report(self) -> Dict[str, Any]:
        report = {
            "model_name": self.model_name,
            "load_error": self._load_error,
            "kv_cache_enabled": self.kv_cache_enabled,
            "cache_implementation": self.cache_implementation,
            "cuda": safe_cuda_summary(),
            "offload_folder": OFFLOAD_DIR,
            "bitsandbytes_available": BitsAndBytesConfig is not None,
        }
        if self._model is not None:
            report.update(model_runtime_summary(self._model, self.model_name))
        return report

    def load(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            self.logger.info("Loading model '%s'", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self._build_model_kwargs(),
            )
            if hasattr(self._model, "config"):
                self._model.config.use_cache = self.kv_cache_enabled
            self._model.eval()
            self.logger.info("Model '%s' loaded successfully", self.model_name)
            self.logger.info("Model runtime summary: %s", json.dumps(self.runtime_report(), default=str))
            return True
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            self.logger.exception("Failed to load model '%s': %s", self.model_name, exc)
            return False

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        if not self.load():
            raise RuntimeError(self._load_error or f"Unable to load {self.model_name}")

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        model_device = getattr(self._model, "device", torch.device("cpu"))
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        do_sample = temperature > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 1e-5),
            "do_sample": do_sample,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": self.kv_cache_enabled,
        }
        if self.cache_implementation:
            generation_kwargs["cache_implementation"] = self.cache_implementation

        with torch.no_grad():
            output = self._model.generate(**inputs, **generation_kwargs)

        generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

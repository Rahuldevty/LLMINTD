import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.model_runtime import BitsAndBytesConfig, MistralRuntime, TextClassifierRuntime, load_config, safe_cuda_summary, setup_logger
from providers import create_generator_provider


def main() -> None:
    config = load_config()
    logger = setup_logger("runtime_diagnostics")

    report = {
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "cuda": safe_cuda_summary(),
        "bitsandbytes_available": BitsAndBytesConfig is not None,
        "models": config["models"],
        "quantization": config.get("quantization", {}),
    }

    generator_provider = create_generator_provider(config, logger)
    report["generator_provider"] = generator_provider.runtime_report()
    if hasattr(generator_provider, "list_models"):
        try:
            report["generator_provider_models"] = generator_provider.list_models()
        except Exception as exc:
            report["generator_provider_error"] = str(exc)

    planner_runtime = TextClassifierRuntime(config["models"]["planner"], logger)
    try:
        planner_runtime.load()
        report["planner_classifier_loaded"] = True
    except Exception as exc:
        report["planner_classifier_loaded"] = False
        report["planner_classifier_error"] = str(exc)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

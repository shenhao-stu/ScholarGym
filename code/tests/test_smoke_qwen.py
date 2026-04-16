import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import _call_llm
import config
from structures import PlannerOutput, SelectorOutput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="你是谁？")
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--schema", choices=["selector", "planner"], default="selector")
    args = parser.parse_args()

    response_format = SelectorOutput if args.schema == "selector" else PlannerOutput
    result = _call_llm(
        args.prompt,
        config.LLM_MODEL_NAME,
        config.LLM_GEN_PARAMS,
        config.IS_LOCAL_LLM,
        return_structured=args.structured,
        response_format=response_format,
        enable_thinking=config.ENABLE_REASONING,
    )

    print(type(result).__name__)
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)

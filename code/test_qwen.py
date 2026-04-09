from api import _call_llm
import config
from structures import SelectorOutput

if __name__ == "__main__":
    prompt = "你是谁？"
    result = _call_llm(
        prompt, config.LLM_MODEL_NAME, config.LLM_GEN_PARAMS, config.IS_LOCAL_LLM,
        return_structured=config.ENABLE_STRUCTURED_OUTPUT,
        response_format=SelectorOutput,
        enable_thinking=config.ENABLE_REASONING,
    )
    print(result)

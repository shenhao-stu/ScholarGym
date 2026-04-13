"""ScholarGym prompts package.

Re-exports all prompt constants so existing `from prompt import X` call
sites keep working after the split into submodules by agent/role:

  - query_generation : SimpleWorkflow query expansion prompts
  - planner          : Planner agent prompts (3 variants)
  - selector         : Selector agent prompts (4 variants for browser modes)
  - summarizer       : PaperSummarizer prompts
  - browser          : Browser agent decision + extraction prompts
  - report           : Final research report writer prompt
"""
from prompt.query_generation import (  # noqa: F401
    COMPLEX_QUERY_GENERATION_PROMPT,
    SIMPLE_QUERY_GENERATION_PROMPT,
)
from prompt.planner import (  # noqa: F401
    PLANNER_SYSTEM_PROMPT,
    PLANNER_ITERATION_PROMPT,
    PLANNER_SYSTEM_PROMPT_ABLATION,
    PLANNER_ITERATION_PROMPT_ABLATION,
    PLANNER_SYSTEM_PROMPT_FULL_HISTORY,
    PLANNER_ITERATION_PROMPT_FULL_HISTORY,
)
from prompt.selector import (  # noqa: F401
    SELECTOR_RECIPE,
    SELECTOR_SYSTEM_PROMPT,
    SELECTOR_DECISION_PROMPT,
    SELECTOR_SYSTEM_INCREMENTAL_PROMPT,
    SELECTOR_DECISION_INCREMENTAL_PROMPT,
    SELECTOR_SYSTEM_REFRESH_PROMPT,
    SELECTOR_DECISION_REFRESH_PROMPT,
    SELECTOR_SYSTEM_PRE_ENRICH_PROMPT,
    SELECTOR_DECISION_PRE_ENRICH_PROMPT,
)
from prompt.summarizer import (  # noqa: F401
    PAPER_SUMMARY_AGENT_PROMPT,
    PAPER_SUMMARY_BATCH_PROMPT,
)
from prompt.browser import (  # noqa: F401
    BROWSER_DECISION_SYSTEM_PROMPT,
    BROWSER_DECISION_USER_PROMPT,
    BROWSER_EXTRACTION_SYSTEM_PROMPT,
    BROWSER_EXTRACTION_USER_PROMPT,
)
from prompt.report import REPORT_AGENT_PROMPT  # noqa: F401

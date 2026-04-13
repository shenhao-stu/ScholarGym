"""ScholarGym workflows package.

Re-exports both workflow classes so callers can use a single import:

    from workflows import DeepResearchWorkflow, SimpleWorkflow
"""
from workflows.deep_research import DeepResearchWorkflow  # noqa: F401
from workflows.simple import SimpleWorkflow  # noqa: F401

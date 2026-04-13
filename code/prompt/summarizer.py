"""PaperSummarizer agent prompts (single + batch)."""

PAPER_SUMMARY_AGENT_PROMPT = """You are a research analyst. Your task is to summarize a paper's abstract.

<original_query>
{query}
</original_query>

<paper_to_summarize>
Title: {title}
Abstract: {abstract}
</paper_to_summarize>

<instructions>
1.  Read the original research query to understand the user's goal.
2.  Read the paper's title and abstract.
3.  Write a concise, one-sentence summary of the abstract, focusing on the aspects most relevant to the original query.
</instructions>

<output_format>
Provide ONLY the summary text.
</output_format>
"""

PAPER_SUMMARY_BATCH_PROMPT = """
You are a research analyst. Summarize multiple papers in one go.

<original_query>
{query}
</original_query>

<sub_query>
{text}
</sub_query>

<papers>
{papers_block}
</papers>

<instructions>
- For each paper, write ONE compact sentence (≤ 30 words) tailored to the original query and the sub_query.
- Use plain language; avoid citations, lists, and extra punctuation.
- Focus on the main contribution or finding most relevant to the sub_query.
- Output a flat JSON object: paper_id (string) → summary (string).
</instructions>

<output_format>
<summary_output>
{{
  "paper_id_1": "one-sentence summary",
  "paper_id_2": "one-sentence summary"
}}
</summary_output>
</output_format>
"""

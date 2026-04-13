"""Query generation prompts for the SimpleWorkflow."""

COMPLEX_QUERY_GENERATION_PROMPT = """<role>Academic Research Strategist</role>

<task>
Transform the given research query into multiple precise search sub-queries for academic paper retrieval.
</task>

<input_query>
{query}
</input_query>

<instructions>
## Analysis Approach
**For Complex Queries (paragraphs/descriptions):**
- Core concepts: Main theoretical frameworks and methodologies mentioned
- Technical terms: Specific algorithms, metrics, or approaches described
- Research problems: Challenges, gaps, or issues explicitly addressed
- Application domains: Use cases and practical implementations discussed

**For Simple Queries (titles/short phrases):**
- Domain identification: What research field or area this belongs to
- Concept expansion: Related terminology and broader research context
- Methodological angles: Different approaches within this research area

## Generation Strategy
Create sub-queries covering these perspectives:
1. Foundational concepts and theoretical frameworks
2. Methodological approaches and technical implementations
3. Empirical studies and performance evaluations
4. Survey literature and comprehensive reviews
5. Challenges, limitations, and future directions

## Output Requirements
- Use precise academic terminology
- Target different aspects of the research area
- Ensure queries are distinct and non-redundant
- Focus on terms likely to appear in paper titles and abstracts
</instructions>

<output_format>
Provide ONLY the search sub-queries, wrapped in <sub_queries> tags. Each query should be on a new line.

<example>
<sub_queries>
evaluation generalization language models
predictive scaling model capabilities
bounded evaluation unbounded capacity
interpretability metrics model assessment
</sub_queries>
</example>
</output_format>

Generate the search sub-queries:"""

# Simple version for quick generation
SIMPLE_QUERY_GENERATION_PROMPT = """<role>Academic Search Assistant</role>

<task>
Generate concise search terms for finding research papers related to the given query.
</task>

<input_query>
{query}
</input_query>

<instructions>
- Use academic terminology
- Cover different research angles
- Keep each sub-query to 3-8 words
- Focus on key concepts
</instructions>

<output_format>
Output ONLY the search terms, one per line, wrapped in <sub_queries> tags.

<example>
<sub_queries>
neural network optimization
gradient descent algorithms
backpropagation techniques
</sub_queries>
</example>
</output_format>

Generate the search sub-queries:"""

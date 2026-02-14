"""
Browser Agent: Access paper content (Async Version)
"""
# # Testing path issues
# import sys
# import os

# # Get the absolute path of the current file
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Assume the target module is in the parent directory
# parent_dir = os.path.dirname(current_dir)

# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import bs4
import httpx
import asyncio
import re
import os

import config
from api import _call_llm
from utils import parse_json_from_tag
from prompt import (
    BROWSER_DECISION_SYSTEM_PROMPT,
    BROWSER_DECISION_USER_PROMPT,
    BROWSER_EXTRACTION_SYSTEM_PROMPT,
    BROWSER_EXTRACTION_USER_PROMPT
)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# error types:
class FetchError(Exception):
    def __init__(self, message, error_type="Unknown Error"):
        super().__init__(message)
        self.error_type = error_type


logger = logging.getLogger(__name__)

class Ar5ivParser:
    def __init__(self):
        self.ignore_tags = ['a', 'figure', 'center', 'caption', 'td', 'h1', 'h2', 'h3', 'h4', 'sup']
        self.max_math_length = 300000

    def parse_authors(self, soup):
        """Parse author information and remove superscripts (institutional references)"""
        authors_tag = soup.find(class_='ltx_authors')
        if not authors_tag:
            return ""
        
        for sup in authors_tag.find_all('sup'):
            sup.decompose() # 从树中移除标签
            
        text = authors_tag.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text

    def clean_text(self, text):
        """remove unwanted substrings and normalize spaces in the text"""
        delete_items = ['=-1', '\t', u'\xa0', '[]', '()', 'mathbb', 'mathcal', 'bm', 'mathrm', 
                        'mathit', 'mathbf', 'mathbfcal', 'textbf', 'textsc', 'langle', 'rangle', 'mathbin']
        for item in delete_items:
            text = text.replace(item, '')
        text = re.sub(' +', ' ', text)
        text = re.sub(r'[\[\],]+', '', text)
        text = re.sub(r'\.(?!\d)', '. ', text)
        text = re.sub('bib. bib', 'bib.bib', text)
        return text

    def parse_metadata(self, metas):
        """Parse metadata of references"""
        metas = [item.replace('\n', ' ') for item in metas]
        meta_string = ' '.join(metas)
        authors, title, journal = "", "", ""
        
        if len(metas) == 3: 
            authors, title, journal = metas
        else:
            meta_string = re.sub(r'\.\s\d{4}[a-z]?\.', '.', meta_string)
            regex = r"^(.*?\.\s)(.*?)(\.\s.*|$)"
            match = re.match(regex, meta_string, re.DOTALL)
            if match:
                authors = match.group(1).strip() if match.group(1) else ""
                title = match.group(2).strip() if match.group(2) else ""
                journal = match.group(3).strip() if match.group(3) else ""
                if journal.startswith('. '):
                    journal = journal[2:]

        return {
            "meta_list": metas, 
            "meta_string": meta_string, 
            "authors": authors,
            "title": title,
            "journal": journal
        }

    def create_dict_for_citation(self, ul_element):
        """Extract citation information from an HTML list"""
        if not ul_element:
            return {}
        citation_dict, futures, id_attrs = {}, [], []
        for li in ul_element.find_all("li", recursive=False):
            id_attr = li.get('id')
            metas = [x.text.strip() for x in li.find_all('span', class_='ltx_bibblock')]
            if id_attr:
                id_attrs.append(id_attr)
                futures.append(self.parse_metadata(metas))
        results = list(zip(id_attrs, futures))
        return dict(results)

    def generate_full_toc(self, soup):
        """Generate hierarchical table of contents based on HTML headings (h1-h5)"""
        toc = []
        stack = [(0, toc)]
        heading_tags = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5}
        
        for tag in soup.find_all(heading_tags.keys()):
            level = heading_tags[tag.name]
            title = tag.get_text()
            
            while stack and stack[-1][0] >= level:
                stack.pop()
            
            current_level = stack[-1][1]
            section = tag.find_parent('section', id=True)
            section_id = section.get('id') if section else None
            
            new_entry = {'title': title, 'id': section_id, 'subsections': []}
            current_level.append(new_entry)
            stack.append((level, new_entry['subsections']))
        
        return toc

    def parse_text_recursive(self, local_text, tag):
        """Recursively parse DOM nodes to extract text, handling formulas and citations"""
        for child in tag.children:
            child_type = type(child)
            if child_type == bs4.element.NavigableString:
                txt = child.get_text()
                local_text.append(txt)

            elif child_type == bs4.element.Comment:
                continue
            
            elif child_type == bs4.element.Tag:
                if child.name in self.ignore_tags or (child.has_attr('class') and child['class'][0] == 'navigation'):
                    continue
                elif child.name == 'cite':
                    hrefs = [a.get('href').strip('#') for a in child.find_all('a', class_='ltx_ref') if a.get('href')]
                    if hrefs:
                        local_text.append('~\cite{' + ', '.join(hrefs) + '}')
                elif child.name == 'img' and child.has_attr('alt'):
                    math_txt = child.get('alt')
                    if len(math_txt) < self.max_math_length:
                        local_text.append(math_txt)
                elif child.has_attr('class') and (child['class'][0] == 'ltx_Math' or child['class'][0] == 'ltx_equation'):
                    math_txt = child.get_text()
                    if len(math_txt) < self.max_math_length:
                        local_text.append(math_txt)
                elif child.name == 'section':
                    return
                else:
                    self.parse_text_recursive(local_text, child)

    def remove_stop_word_sections_and_extract_text(self, toc, soup, stop_words=None):
        """Filter out irrelevant sections and extract main text content"""
        if stop_words is None:
            stop_words = ['references', 'acknowledgments', 'about this document', 'appendix']

        def has_stop_word(title, stop_words):
            return any(stop_word.lower() in title.lower() for stop_word in stop_words)
        
        filtered_entries = []
        for entry in toc:
            if not has_stop_word(entry['title'], stop_words):
                section_id = entry['id']
                if section_id:
                    section = soup.find(id=section_id)
                    if section is not None:
                        local_text = []
                        self.parse_text_recursive(local_text, section)
                        if local_text:
                            processed_text = self.clean_text(''.join(local_text))
                            entry['text'] = processed_text
                
                entry['subsections'] = self.remove_stop_word_sections_and_extract_text(entry['subsections'], soup, stop_words)
                filtered_entries.append(entry)
        return filtered_entries

    def parse_html_content(self, html_content):
        """Main parsing logic: input HTML string, output structured dictionary"""
        try:
            soup = bs4.BeautifulSoup(html_content, "lxml")
            
            # 1. Parse title
            title_tag = soup.head.title
            if not title_tag:
                 raise ValueError("Missing <title> tag")
            
            title = title_tag.get_text().replace("\n", " ")
            authors = self.parse_authors(soup)
            
            # 2. Parse abstract
            abstract_tag = soup.find(class_='ltx_abstract')
            abstract = abstract_tag.get_text().strip() if abstract_tag else ""
            
            # 3. Parse references
            citation_tag = soup.find(class_='ltx_biblist')
            citation_dict = self.create_dict_for_citation(citation_tag)
            
            # 4. Generate table of contents
            sections = self.generate_full_toc(soup)
            
            # 5. Extract main text and filter
            sections = self.remove_stop_word_sections_and_extract_text(sections, soup)
            
            return {
                "title": title, 
                "abstract": abstract, 
                "authors": authors,
                "sections": sections, 
                "references": citation_dict,
            }
        except Exception as e:
            raise FetchError(f"Parsing Logic Error: {str(e)}", error_type="Parse_Exception")

    async def fetch_and_parse(self, arxiv_id, max_retries=3, retry_delay=2):
        """
        Entry function: asynchronously fetch and parse paper by arXiv ID (based on httpx)
        """
        clean_id = arxiv_id.split('v')[0]
        url = f'https://ar5iv.labs.arxiv.org/html/{clean_id}'
        
        logger.info(f"Fetching from: {url}")
        
        async with httpx.AsyncClient(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.get(url)
                    
                    if response.status_code != 200:
                        raise FetchError(f"HTTP Status {response.status_code}", error_type=f"HTTP_{response.status_code}")
                    
                    html_content = response.text
                    
                    if 'https://ar5iv.labs.arxiv.org/html' not in html_content and '<html' not in html_content:
                         raise FetchError("Content likely not valid ar5iv HTML", error_type="Invalid_Content")
                    
                    return self.parse_html_content(html_content)

                except (httpx.TimeoutException, httpx.RequestError, Exception) as e:
                    if isinstance(e, httpx.TimeoutException):
                        err = FetchError("Request Timed Out", error_type="Network_Timeout")
                    elif isinstance(e, httpx.RequestError):
                        err = FetchError(f"Network Request Failed: {e}", error_type="Network_General")
                    elif isinstance(e, FetchError):
                        err = e
                    else:
                        err = FetchError(f"Unexpected Error: {e}", error_type="Unexpected_Error")
                    
                    if attempt == max_retries - 1 or (isinstance(err, FetchError) and err.error_type.startswith("HTTP_4")):
                        raise err
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {err}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)


def convert_dict_to_llm_readable(data):
    """
    Convert the paper's dictionary data into a well-structured Markdown text.
    """
    output_lines = []
    title = data.get("title", "Untitled").strip()
    output_lines.append(f"# {title}\n")
    if "authors" in data and data["authors"]:
        output_lines.append(f"**Authors:** {data['authors']}\n")

    if "abstract" in data:
        output_lines.append("## Abstract")
        output_lines.append(data["abstract"].strip() + "\n")

    def process_section(section, level=2):
        sec_title = section.get("title", "").strip()
        if sec_title:
            output_lines.append(f"{'#' * level} {sec_title}")
        
        text_content = section.get("text", "")
        if text_content:
            cleaned_text = "\n".join([line.strip() for line in text_content.split('\n') if line.strip()])
            output_lines.append(cleaned_text)
        
        output_lines.append("") 

        subsections = section.get("subsections", [])
        if subsections:
            for sub in subsections:
                process_section(sub, level + 1)

    for sec in data.get("sections", []):
        process_section(sec)

    if "references" in data:
        output_lines.append("## References")
        for ref_id, ref_data in data["references"].items():
            meta = ref_data.get("meta_string", "")
            if not meta and ref_data.get("meta_list"):
                meta = " ".join(ref_data["meta_list"])
            output_lines.append(f"[{ref_id}] {meta}")

    return "\n".join(output_lines)

class Browser:
    """
    Browser Agent responsible for examining papers in depth.
    """

    def __init__(self, llm_model: str, gen_params: Dict, is_local: bool, trace_recorder=None):
        self.llm_model = llm_model
        self.gen_params = gen_params.copy()
        self.gen_params['max_tokens'] = config.BROWSER_MAX_TOKENS
        self.is_local = is_local
        self.trace_recorder = trace_recorder

    async def browse_papers(
        self,
        subquery_text: str,
        paper: Any,
        iteration_index: int,
        idx: int,
        sq_id: int,
        paper_idx: int = 0,
        task: Optional[str] = None
    ):
        """
        Async entry point to browse a single paper.
        """
        try:
            logger.info(f"[🔎 Browser] Assessing paper '{getattr(paper, 'title', 'Unknown')}' for subquery: '{subquery_text}'")
            
            needs_browsing = False
            # 1. Decision Step (LLM call remains sync usually, unless _call_llm is async)
            if config.BROWSER_MODE == "PRE_ENRICH":
                needs_browsing, task, _ = self._decide_browsing(
                    subquery_text, paper, iteration_index, idx, sq_id, paper_idx
                )
            
            if (needs_browsing or config.BROWSER_MODE in ["REFRESH", "INCREMENTAL"]) and task:
                logger.info(f"   -> Browsing needed for '{getattr(paper, 'title', 'Unknown')}': {task}")
                
                # 2. Fetch Full Text (Async)
                full_text = await self._fetch_full_text(paper)
                if full_text:
                    if idx < 5 and config.DEBUG:
                        try:
                            os.makedirs("./case_study", exist_ok=True)
                            with open(f"./case_study/qid{idx}_browser_fulltext_iter{iteration_index}_sq{sq_id}_p{paper_idx}.txt", "w", encoding="utf-8") as f:
                                f.write(full_text)
                        except Exception as e:
                            logger.warning(f"Failed to save debug fulltext: {e}")

                if full_text:
                    # 3. Extraction Step
                    extraction_data = self._extract_content(
                        task, full_text, iteration_index, idx, sq_id, paper_idx=paper_idx
                    )
                    _ = extraction_data.get("rational", "")
                    _ = extraction_data.get("evidence", "")
                    summary = extraction_data.get("summary", "")
                    # combined_content = []
                    # if rational:
                    #     combined_content.append("RATIO:" + rational + ";")
                    # if evidence:
                    #     combined_content.append("EVIDENCE:" + evidence + ";")
                    # if summary:
                    #     combined_content.append("SUMMARY:" + summary + ";")
                    # browsing_content = "".join(combined_content)
                    if summary:
                        paper.browsing_content = summary
                    else:
                        logger.warning(f"   -> No content extracted for '{getattr(paper, 'title', 'Unknown')}'")
                        paper.browsing_content = "Browsing exception. Here is abstract: " + (getattr(paper, "abstract", "") or "")
                else:
                    logger.warning(f"   -> Full text not available for '{getattr(paper, 'title', 'Unknown')}'")
                    paper.browsing_content = "Browsing exception. Here is abstract: " + (getattr(paper, "abstract", "") or "")
            else:
                paper.browsing_content = "Browsing exception. Here is abstract: " + (getattr(paper, "abstract", "") or "")
        except Exception as e:
            logger.error(f"Error in browse_papers: {e}")
            paper.browsing_content = "Browsing exception. Here is abstract: " + (getattr(paper, "abstract", "") or "")

    def _decide_browsing(
        self, 
        subquery: str, 
        paper: Any, 
        iteration_index: int,
        idx: int,
        sq_id: int,
        paper_idx: int
    ) -> Tuple[bool, Optional[str], str]:
        """
        Step 1: Check if browsing is necessary based on Abstract.
        """
        title = getattr(paper, "title", "")
        abstract = getattr(paper, "abstract", "")
        
        prompt = (
            BROWSER_DECISION_SYSTEM_PROMPT 
            + "\n\n" 
            + BROWSER_DECISION_USER_PROMPT.format(
                subquery=subquery,
                title=title,
                abstract=abstract
            )
        )

        response_data = self._execute_llm_call(
            prompt, 
            tag_name="decision_output",
            step_name="browser_decision",
            iteration=iteration_index,
            idx=idx,
            sq_id=sq_id,
            paper_idx=paper_idx
        )

        needs_browsing = response_data.get("needs_browsing", False)
        task = response_data.get("browsing_task", None)
        reasoning = response_data.get("reasoning", "")
        
        if needs_browsing and not task:
            needs_browsing = False
            
        return needs_browsing, task, reasoning

    def _extract_content(
        self,
        task: str,
        full_text: str,
        iteration_index: int,
        idx: int,
        sq_id: int,
        paper_idx: int
    ) -> Dict:
        """
        Step 2: Extract information from full text.
        """
        prompt = (
            BROWSER_EXTRACTION_SYSTEM_PROMPT
            + "\n\n"
            + BROWSER_EXTRACTION_USER_PROMPT.format(
                full_text=full_text,
                task=task
            )
        )

        response_data = self._execute_llm_call(
            prompt,
            tag_name="extractor_output",
            step_name="browser_extraction",
            iteration=iteration_index,
            idx=idx,
            sq_id=sq_id,
            paper_idx=paper_idx
        )
        
        return response_data

    def _execute_llm_call(
        self, 
        prompt: str, 
        tag_name: str, 
        step_name: str, 
        iteration: int,
        idx: int,
        sq_id: int,
        paper_idx: int
    ) -> Dict:
        """
        Helper to handle LLM call.
        """
        reasoning_content = None
        
        result = _call_llm(
            prompt, self.llm_model, self.gen_params, self.is_local,
            enable_thinking=config.ENABLE_REASONING
        )
        
        if config.ENABLE_REASONING and isinstance(result, tuple):
            reasoning_content, response = result
        else:
            response = result

        data = parse_json_from_tag(response, tag_name) or {}

        if idx < 5 and config.DEBUG:
            case_study_dir = os.path.join(config.CASE_STUDY_OUTPUT_DIR, f"qid_{idx}", f"iter_{iteration}", "browser")
            os.makedirs(case_study_dir, exist_ok=True)
            with open(os.path.join(case_study_dir, f"{step_name}_prompt_sq{sq_id}_p{paper_idx}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(os.path.join(case_study_dir, f"{step_name}_response_sq{sq_id}_p{paper_idx}.txt"), "w", encoding="utf-8") as f:
                f.write(response)

        if self.trace_recorder and config.SAVE_AGENT_TRACES:
            stage_data = {
                'iteration': iteration,
                'sub_query_id': sq_id,
                'paper_idx': paper_idx,
                'step': step_name,
                'prompt': prompt,
                'response': response,
                'reasoning': reasoning_content,
                'parsed_data': data
            }
            self.trace_recorder.record_stage(idx, 'browser', stage_data)

        return data

    async def _fetch_full_text(self, paper: Any) -> Optional[str]:
        """
        Fetch full text using Ar5ivParser (Async).
        """
        arxiv_id = getattr(paper, "id", None) or getattr(paper, "arxiv_id", None)
        
        if not arxiv_id:
            logger.warning(f"No arXiv ID found for paper: {getattr(paper, 'title', 'Unknown')}")
            return None

        parser = Ar5ivParser()
        try:
            parsed_data = await parser.fetch_and_parse(arxiv_id, max_retries=3)
            return convert_dict_to_llm_readable(parsed_data)
        except Exception as e:
            logger.error(f"Failed to fetch full text for {arxiv_id}: {e}")
            return None
        
# -------------------------------------------------------------------------
# Test Function (Async)
# -------------------------------------------------------------------------
async def test_browser():
    browser = Browser(
        llm_model=config.LLM_MODEL_NAME,
        gen_params=config.LLM_GEN_PARAMS,
        is_local=config.IS_LOCAL_LLM
    )
    
    class Paper:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.browsing_content = None

    sample_paper = Paper(
        id="2308.04079",
        title="3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        abstract="Radiance Field methods have recently revolutionized novel-view synthesis..."
    )
    
    subquery = "Explain the main contributions of the paper."
    
    print("🚀 Starting async browser test...")

    await browser.browse_papers(
        subquery_text=subquery,
        paper=sample_paper,
        iteration_index=4,
        idx=3,
        sq_id=2,
        paper_idx=1,
        task="Identify and summarize the key contributions of the paper."
    )
    
    print("✅ Browsing Content Result:")
    print(getattr(sample_paper, "browsing_content", "No content extracted."))

if __name__ == "__main__":
    asyncio.run(test_browser())
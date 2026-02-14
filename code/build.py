import json
import time
import arxiv
import re
import os
import requests
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from functools import wraps
from logger import get_logger
from rouge import Rouge

DOCLING_AVAILABLE = False
USE_SEMANTIC_SCHOLAR = False  # Global flag to control Semantic Scholar API usage

if USE_SEMANTIC_SCHOLAR:
    from semantic_api import SemanticScholarAPI
if DOCLING_AVAILABLE:
    from docling.document_converter import DocumentConverter as DoclingConverter

logger = get_logger(__name__, log_file='./log/build.log')


def safe_operation(operation_name: str, increment_stat: str = None):
    """Decorator for safe operations with consistent error handling and logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                if result and increment_stat:
                    self.stats.increment(increment_stat)
                return result
            except Exception as e:
                logger.error(f"[❌] Error in {operation_name}: {e}")
                if increment_stat and increment_stat.endswith('_failures'):
                    self.stats.increment(increment_stat)
                elif increment_stat:
                    failure_stat = increment_stat.replace('_processed', '_failures').replace('_downloaded', '_failures')
                    self.stats.increment(failure_stat)
                return None
        return wrapper
    return decorator


def calc_rouge_scores(p: str, g: str) -> float:
    """Calculate ROUGE-L F1 score between prediction and ground truth."""
    if Rouge is None:
        logger.warning("Rouge not available, returning 0.0")
        return 0.0
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(p, g)[0]
        return scores["rouge-l"]["f"]
    except:
        return 0.0


class TextCleaner:
    """Centralized text cleaning operations with consistent behavior."""
    
    @staticmethod
    def _clean_text_common(text: str) -> str:
        """Apply common text cleaning operations."""
        if not text:
            return ""
        
        # Normalize line breaks and whitespace
        text = re.sub(r'\n(?!\n)', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove LaTeX formatting artifacts
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Fix hyphenated words broken by line breaks
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        return text.strip()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for general use."""
        return TextCleaner._clean_text_common(text)
    
    @staticmethod
    def clean_title_for_search(title: str) -> str:
        """Clean and format title optimized for arXiv search."""
        if not title:
            return ""
        
        title = TextCleaner._clean_text_common(title)
        
        # Remove year and optional single letter with dot pattern at the beginning
        # Handles patterns like: "2023d. Api-bank: A benchmark..." or "2023. The expresssive power..."
        title = re.sub(r'^\d{4}[a-zA-Z]?\.\s*', '', title)
        
        # Remove trailing punctuation but preserve important characters
        title = re.sub(r'[.,:;]+$', '', title)
        title = re.sub(r'[^\w\s\-:().,?!]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title


class FileManager:
    """Centralized file operations and naming conventions."""
    
    @staticmethod
    def clean_filename(filename: str, max_length: int = 200) -> str:
        """Clean filename by removing invalid characters and limiting length."""
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename[:max_length] if len(filename) > max_length else filename
    
    @staticmethod
    def create_paper_filename(arxiv_id: str, title: str, extension: str = 'pdf') -> str:
        """Create standardized paper filename."""
        clean_title = FileManager.clean_filename(title)
        return f"[arXiv-{arxiv_id}]{clean_title}.{extension}"
    
    @staticmethod
    def create_citation_filename(pdf_stem: str) -> str:
        """Create citation JSON filename from PDF stem."""
        return f"cited_{pdf_stem}.json"
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists, create if it doesn't."""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def extract_arxiv_id_from_url(url: str) -> Optional[str]:
        """Extract arXiv ID from various URL formats."""
        patterns = [
            r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
            r'arxiv\.org/(?:abs|pdf)/([a-z-]+/\d+)',
            # r'(\d+\.\d+)',
            # r'([a-z-]+/\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None


class ArxivDownloader:
    """Handles arXiv paper and source downloads with consistent error handling."""
    
    def __init__(self, client: arxiv.Client):
        self.client = client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; academic-crawler/1.0)'
        })
    
    def download_pdf(self, arxiv_id: str, output_dir: Path, filename: str) -> Optional[Path]:
        """Download PDF from arXiv with retry logic."""
        filepath = output_dir / filename
        
        if filepath.exists():
            logger.info(f"[📂] PDF already exists: {filename}")
            return filepath
        
        try:
            # Search for the paper
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(self.client.results(search))
            
            if not results:
                logger.warning(f"[❌] Paper not found on arXiv: {arxiv_id}")
                return None
            
            # Download the paper
            logger.info(f"[⬇️] Downloading PDF: {filename}")
            results[0].download_pdf(dirpath=str(output_dir), filename=filename)
            logger.info(f"[✅] Successfully downloaded PDF: {filename}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"[❌] Error downloading PDF {arxiv_id}: {e}")
            return None
    
    def download_source(self, arxiv_id: str, output_dir: Path) -> Optional[Path]:
        """Download and extract TeX source files from arXiv."""
        try:
            # Clean arXiv ID (remove version if present)
            clean_id = re.sub(r'v\d+$', '', arxiv_id.strip())
            
            # Construct source URL
            source_url = f"https://arxiv.org/src/{clean_id}"
            
            # Create temporary download path
            temp_file = output_dir / f"temp_{clean_id}.tar.gz"
            extract_dir = output_dir / f"arXiv-{clean_id}"
            
            # Skip if already extracted
            if extract_dir.exists():
                logger.info(f"[📂] TeX source already exists: {extract_dir.name}")
                return extract_dir
            
            logger.info(f"[⬇️] Downloading TeX source: {source_url}")
            
            # Download the source file
            response = self.session.get(source_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the tar.gz file directly to target directory
            logger.info(f"[📦] Extracting TeX source: {temp_file.name}")
            
            # Create the target directory
            extract_dir.mkdir(exist_ok=True)
            
            # Extract directly to the target directory
            with tarfile.open(temp_file, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
            
            logger.info(f"[✅] Successfully extracted TeX source: {extract_dir.name}")
            return extract_dir
            
        except requests.RequestException as e:
            logger.error(f"[❌] Network error downloading TeX source {arxiv_id}: {e}")
            return None
        except tarfile.TarError as e:
            logger.error(f"[❌] Error extracting TeX source {arxiv_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"[❌] Unexpected error downloading TeX source {arxiv_id}: {e}")
            return None
        finally:
            # Cleanup temporary file if it exists
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()


class ArxivSearcher:
    """Handles arXiv paper searching with ROUGE-L validation."""
    
    ROUGE_THRESHOLD = 0.85
    
    def __init__(self, client: arxiv.Client):
        self.client = client
        self.search_count = 0
        self.success_count = 0
    
    def _extract_paper_data(self, result: arxiv.Result) -> Dict:
        """Extract standardized paper data from arXiv result."""
        return {
            'id': result.entry_id.split('/')[-1],
            'title': TextCleaner.clean_text(result.title),
            'url': result.pdf_url,
            'date': result.published.strftime('%Y-%m-%d') if result.published else '',
            'abstract': TextCleaner.clean_text(result.summary),
            'category': [category for category in result.categories],
            'authors': [author.name for author in result.authors],
        }
    
    def _create_empty_paper_data(self, ref: Dict) -> Dict:
        """Create empty paper data structure for failed searches."""
        return {
            'id': ref.get('arxiv_id'),
            'title': TextCleaner.clean_text(ref.get('title', '')) if ref.get('title') else '',
            'url': '',
            'date': '',
            'abstract': '',
            'category': [],
            'authors': []
        }
    
    def search_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """Search arXiv by paper ID."""
        try:
            clean_id = re.sub(r'v\d+$', '', arxiv_id.strip())
            search = arxiv.Search(id_list=[clean_id])
            results = list(self.client.results(search))
            
            self.search_count += 1
            
            if results:
                self.success_count += 1
                paper_data = self._extract_paper_data(results[0])
                # logger.info(f"[✅] Found paper by ID: {clean_id} - {paper_data['title']}")
                return paper_data
            else:
                logger.warning(f"[❌] Paper not found by ID: {clean_id}")
                return None
                
        except Exception as e:
            logger.error(f"[❌] Error searching arXiv by ID {arxiv_id}: {e}")
            return None
    
    def search_by_ids(self, arxiv_ids: List[str]) -> Dict[str, Dict]:
        """Search arXiv by a list of paper IDs in a batch and return a dictionary of results."""
        try:
            # Create a mapping from cleaned ID (no version) to original ID
            id_map = {re.sub(r'v\d+$', '', an_id.strip()): an_id for an_id in arxiv_ids}
            clean_ids = list(id_map.keys())

            search = arxiv.Search(id_list=clean_ids, max_results=len(clean_ids))
            
            results_generator = self.client.results(search)

            self.search_count += 1
            
            paper_data_map = {}
            for result in results_generator:
                # get_short_id() might include version, e.g., '2103.10385v1'
                result_id_clean = re.sub(r'v\d+$', '', result.get_short_id())
                
                if result_id_clean in id_map:
                    original_id = id_map[result_id_clean]
                    paper_data_map[original_id] = self._extract_paper_data(result)

            self.success_count += len(paper_data_map)
            
            return paper_data_map

        except Exception as e:
            logger.error(f"[❌] Error searching arXiv by ID list: {e}")
            return {}
    
    def search_by_title(self, title: str) -> Optional[Dict]:
        """Search arXiv by paper title with ROUGE-L validation."""
        try:
            clean_title = TextCleaner.clean_title_for_search(title)
            if not clean_title:
                logger.warning(f"[⚠️] Title too short after cleaning: {title}")
                return None
            
            # Multiple search strategies for better results
            search_queries = [f'ti:"{clean_title}"', f'ti:{clean_title}']
            
            self.search_count += 1
            
            for search_query in search_queries:
                search = arxiv.Search(
                    query=search_query,
                    max_results=1,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                results = list(self.client.results(search))
                
                if results:
                    result = results[0]
                    result_title = TextCleaner.clean_text(result.title)
                    result_title = re.sub(r'[^\w\s]+$', '', result_title).strip()
                    result_title = re.sub('QuAC : ', 'QuAC: ', result_title)
                    # Validate using ROUGE-L similarity
                    rouge_score = calc_rouge_scores(clean_title.lower(), result_title.lower())
                    
                    if rouge_score > self.ROUGE_THRESHOLD:
                        self.success_count += 1
                        paper_data = self._extract_paper_data(result)
                        logger.info(f"[✅] Found paper by title: {clean_title} -> {paper_data['title']} (ROUGE-L: {rouge_score:.3f})")
                        return paper_data
                    else:
                        logger.warning(f"[⚠️] Low similarity: ROUGE-L {rouge_score:.3f} < {self.ROUGE_THRESHOLD} for '{result_title}'")
                        break
            
            logger.warning(f"[❌] Paper not found by title: {clean_title}")
            return None
            
        except Exception as e:
            logger.error(f"[❌] Error searching arXiv by title '{title}': {e}")
            return None
    
    def search_reference(self, ref: Dict) -> Dict:
        """Search for a reference using ID first, then title fallback."""
        time.sleep(1)  # Rate limiting
        
        # Try arXiv ID first if available
        if ref.get('arxiv_id'):
            result = self.search_by_id(ref['arxiv_id'])
            if result:
                return result
        
        # Fallback to title search
        if ref.get('title'):
            result = self.search_by_title(ref['title'])
            if result:
                return result
        
        # Return empty data if nothing found
        return self._create_empty_paper_data(ref)


class DocumentConverter:
    """Handles document conversion operations (PDF to Markdown)."""
    
    def __init__(self):
        self.converter = DoclingConverter() if DOCLING_AVAILABLE else None
        if self.converter:
            logger.info("[📄] DocLing converter initialized")
        else:
            logger.warning("[⚠️] DocLing not available - PDF to Markdown conversion disabled")
    
    def convert_pdf_to_markdown(self, pdf_path: Path, output_dir: Path, filename: str) -> Optional[Path]:
        """Convert PDF to Markdown using DocLing."""
        if not self.converter:
            logger.warning(f"[⚠️] DocLing not available, skipping conversion: {pdf_path.name}")
            return None
        
        markdown_path = output_dir / filename
        
        # Skip if markdown already exists
        if markdown_path.exists():
            logger.info(f"[📝] Markdown already exists: {filename}")
            return markdown_path
        
        try:
            logger.info(f"[🔄] Converting PDF to Markdown: {pdf_path.name}")
            
            # Ensure output directory exists
            FileManager.ensure_directory(output_dir)
            
            # Convert using DocLing
            result = self.converter.convert(str(pdf_path))
            markdown_content = result.document.export_to_markdown()
            
            # Save markdown content
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"[✅] Successfully converted to Markdown: {filename}")
            return markdown_path
            
        except Exception as e:
            logger.error(f"[❌] Error converting PDF to Markdown for {pdf_path.name}: {e}")
            return None


class StatisticsTracker:
    """Centralized statistics tracking and reporting."""
    
    def __init__(self):
        self.stats = {
            'papers_processed': 0,
            'papers_downloaded': 0,
            'download_failures': 0,
            'tex_sources_downloaded': 0,
            'tex_download_failures': 0,
            'markdowns_converted': 0,
            'markdown_failures': 0,
            'citations_found': 0,
            'citations_processed': 0,
            'semantic_references_found': 0,
            'semantic_api_failures': 0,
            'metadata_saved': 0,
            'title_match_failures': 0
        }
    
    def increment(self, key: str, value: int = 1) -> None:
        """Increment a statistic counter."""
        if key in self.stats:
            self.stats[key] += value
    
    def print_statistics(self, searcher: ArxivSearcher) -> None:
        """Print comprehensive statistics."""
        logger.info("\n" + "="*60)
        logger.info("SURVEY SCOPE CRAWLER STATISTICS")
        logger.info("="*60)
        logger.info(f"Papers processed: {self.stats['papers_processed']}")
        logger.info(f"PDFs downloaded: {self.stats['papers_downloaded']}")
        logger.info(f"PDF download failures: {self.stats['download_failures']}")
        logger.info(f"TeX sources downloaded: {self.stats['tex_sources_downloaded']}")
        logger.info(f"TeX download failures: {self.stats['tex_download_failures']}")
        logger.info(f"Markdowns converted: {self.stats['markdowns_converted']}")
        logger.info(f"Markdown failures: {self.stats['markdown_failures']}")
        logger.info(f"Citations found: {self.stats['citations_found']}")
        logger.info(f"Citations processed: {self.stats['citations_processed']}")
        
        if USE_SEMANTIC_SCHOLAR:
            logger.info(f"Semantic references found: {self.stats['semantic_references_found']}")
            logger.info(f"Semantic API failures: {self.stats['semantic_api_failures']}")
            logger.info(f"Metadata saved: {self.stats['metadata_saved']}")
            logger.info(f"Title match failures: {self.stats['title_match_failures']}")
        else:
            logger.info(f"Semantic Scholar API: Disabled")
            
        logger.info(f"ArXiv searches performed: {searcher.search_count}")
        logger.info(f"Successful searches: {searcher.success_count}")
        
        if searcher.search_count > 0:
            success_rate = (searcher.success_count / searcher.search_count) * 100
            logger.info(f"Search success rate: {success_rate:.1f}%")
        
        logger.info("="*60)


class SurveyScopeCrawler:
    """
    Comprehensive system for downloading survey papers and crawling their citations.
    
    Features:
    1. Download survey papers (PDF) to datasets/survey/
    2. Download TeX source files to datasets/survey/tex/
    3. Convert PDFs to Markdown (optional)
    4. Extract and process citations using Semantic Scholar API (optional)
    5. Create citation JSON files in datasets/source/
    6. Save paper metadata to meta_survey.jsonl with ROUGE-L validation
    
    New Features:
    - Global flag USE_SEMANTIC_SCHOLAR to control API usage (default: False)
    - ROUGE-L title matching for both Semantic Scholar and arXiv searches
    - Automatic paper metadata saving to meta_survey.jsonl (when enabled)
    - Enhanced reference extraction using Semantic Scholar API (when enabled)
    - Comprehensive statistics tracking including title match failures
    
    Example Usage:
        # Enable Semantic Scholar API globally (optional)
        # from code.build import USE_SEMANTIC_SCHOLAR
        # USE_SEMANTIC_SCHOLAR = True
        
        # Basic usage (Semantic Scholar disabled by default)
        crawler = SurveyScopeCrawler(
            download_only=False,
            download_tex=False,
            semantic_api_key="your_api_key"  # Only used if globally enabled
        )
        crawler.run("surveyscope.jsonl", max_papers=10)
        
        # Files generated:
        # - datasets/survey/*.pdf (downloaded papers)
        # - datasets/survey/meta_survey.jsonl (paper metadata)
        # - datasets/source/cited_*.json (citation data)
    """
    
    def __init__(self, 
                 survey_dir: str = "datasets/survey",
                 source_dir: str = "datasets/source",
                 download_only: bool = False,
                 download_tex: bool = True,
                 semantic_api_key: Optional[str] = None):
        """
        Initialize the crawler with configuration.
        
        Args:
            survey_dir: Directory for survey papers and related files
            source_dir: Directory for citation JSON files
            download_only: If True, skip citation extraction
            download_tex: If True, download TeX source files
            semantic_api_key: Optional Semantic Scholar API key for better rate limits
        """
        self.survey_dir = Path(survey_dir)
        self.source_dir = Path(source_dir)
        self.pdf_dir = self.survey_dir / 'pdf'
        self.tex_dir = self.survey_dir / 'tex_ori'
        self.markdown_dir = self.survey_dir / 'markdown'
        self.download_only = download_only
        self.download_tex = download_tex
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize components
        self.arxiv_client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)
        self.downloader = ArxivDownloader(self.arxiv_client)
        self.searcher = ArxivSearcher(self.arxiv_client)
        self.doc_converter = DocumentConverter()
        self.semantic_api = SemanticScholarAPI(api_key=semantic_api_key) if USE_SEMANTIC_SCHOLAR else None
        self.stats = StatisticsTracker()
        
        self._log_initialization()
    
    def _setup_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [self.survey_dir]
        directories.append(self.pdf_dir)
        if self.download_tex:
            directories.append(self.tex_dir)
        
        if DOCLING_AVAILABLE:
            directories.append(self.markdown_dir)
        
        if not self.download_only:
            directories.append(self.source_dir)
        
        for directory in directories:
            FileManager.ensure_directory(directory)
    
    def _log_initialization(self) -> None:
        """Log initialization status and configuration."""
        mode_parts = []
        mode_parts.append("PDF Download")
        
        if self.download_tex:
            mode_parts.append("TeX Source Download")
        
        if DOCLING_AVAILABLE:
            mode_parts.append("Markdown Conversion")
        
        if not self.download_only:
            mode_parts.append("Citation Extraction")
            if USE_SEMANTIC_SCHOLAR:
                mode_parts.append("Semantic Scholar API")
                mode_parts.append("Metadata Saving")
        
        mode = " + ".join(mode_parts)
        logger.info(f"[🚀] Initialized SurveyScopeCrawler: {mode}")
        logger.info(f"[🔍] ROUGE-L threshold for title matching: {ArxivSearcher.ROUGE_THRESHOLD}")
        logger.info(f"[🌐] Semantic Scholar API: {'Enabled' if USE_SEMANTIC_SCHOLAR else 'Disabled'}")
    
    @safe_operation("paper download", "papers_downloaded")
    def download_paper_files(self, arxiv_id: str, title: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Download all requested file types for a paper."""
        # Generate filenames
        pdf_filename = FileManager.create_paper_filename(arxiv_id, title, 'pdf')
        md_filename = FileManager.create_paper_filename(arxiv_id, title, 'md')
        
        # Download PDF
        pdf_path = self.downloader.download_pdf(arxiv_id, self.pdf_dir, pdf_filename)
        
        # Download TeX source if requested
        tex_path = None
        if self.download_tex:
            tex_path = self.downloader.download_source(arxiv_id, self.tex_dir)
            if tex_path:
                self.stats.increment('tex_sources_downloaded')
            else:
                self.stats.increment('tex_download_failures')
        
        # Convert to Markdown if PDF downloaded successfully
        markdown_path = None
        if pdf_path and DOCLING_AVAILABLE:
            markdown_path = self.doc_converter.convert_pdf_to_markdown(
                pdf_path, self.markdown_dir, md_filename
            )
            if markdown_path:
                self.stats.increment('markdowns_converted')
            else:
                self.stats.increment('markdown_failures')
        
        return pdf_path, tex_path, markdown_path
    
    def process_references(self, references: List[Dict], title: str = None) -> Dict[str, Dict]:
        """Process all references and return citation data, preferring Semantic Scholar API if enabled."""
        # First try to get references from Semantic Scholar API using title (if enabled)
        semantic_references = []
        if title and USE_SEMANTIC_SCHOLAR:
            semantic_references = self.get_semantic_references(title)
        
        # Use Semantic Scholar references if available, otherwise fallback to original references
        final_references = semantic_references if semantic_references else references
        
        self.stats.increment('citations_found', len(final_references))
        logger.info(f"[🔍] Processing {len(final_references)} references")
        
        if semantic_references:
            logger.info(f"[✅] Using {len(semantic_references)} references from Semantic Scholar API")
        elif USE_SEMANTIC_SCHOLAR:
            logger.info(f"[⚠️] Semantic Scholar API failed, fallback to {len(references)} references from original data")
        else:
            logger.info(f"[📄] Using {len(references)} references from original data (Semantic Scholar disabled)")
        
        citation_data = {}
        
        for i, ref in enumerate(final_references, 1):
            if i % 10 == 0:
                logger.info(f"[📊] Processing reference {i}/{len(final_references)}")
            
            # If reference comes from Semantic Scholar, we already have detailed info
            if semantic_references and 'paperId' in ref:
                # Create paper data directly from Semantic Scholar reference
                paper_data = {
                    'id': ref.get('arxiv_id'),
                    'title': ref.get('title'),
                    'url': f"https://arxiv.org/abs/{ref['arxiv_id']}" if ref.get('arxiv_id') else None,
                    'date': str(ref.get('year')) if ref.get('year') else None,
                    'abstract': None,  # Abstract not available in references
                    'category': None,  # Categories not available in references
                    'authors': ref.get('authors', []),
                    'venue': ref.get('venue'),
                    'citationCount': ref.get('citationCount'),
                    'paperId': ref.get('paperId')
                }
            else:
                # Use traditional arXiv search for original data references
                paper_data = self.searcher.search_reference(ref)
            
            citation_data[str(i)] = paper_data
            self.stats.increment('citations_processed')
        
        return citation_data
    
    @safe_operation("citation JSON creation")
    def create_citation_json(self, survey_data: Dict, filepath: Path) -> Optional[str]:
        """Create and save citation JSON file from survey data."""
        references = survey_data.get('references', [])
        title = survey_data.get('title')
        
        # Process references using Semantic Scholar API first, then fallback to original data
        citation_data = self.process_references(references, title=title)
        
        # Create output file path
        output_filename = FileManager.create_citation_filename(filepath.stem)
        output_path = self.source_dir / output_filename
        
        # Save citation data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(citation_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"[💾] Citation data saved: {output_path}")
        return str(output_path)
    
    def process_survey_paper(self, survey_data: Dict) -> bool:
        """Process a single survey paper with all requested operations."""
        arxiv_id = survey_data.get('arxiv_id')
        title = survey_data.get('title', '')
        
        if not arxiv_id:
            logger.warning(f"[⚠️] No arXiv ID found for paper: {title}")
            return False
        
        logger.info(f"[📄] Processing survey paper: {arxiv_id} - {title}")
        
        # Download all requested file types
        pdf_path, tex_path, markdown_path = self.download_paper_files(arxiv_id, title)
        
        if not pdf_path:
            logger.warning(f"[❌] Failed to download paper: {arxiv_id}")
            return False
        
        # Log successful downloads
        if tex_path:
            logger.info(f"[📂] TeX source saved: {tex_path.name}")
        if markdown_path:
            logger.info(f"[📝] Markdown saved: {markdown_path.name}")
        
        # Process citations if not in download-only mode
        if not self.download_only:
            logger.info(f"[🔗] Extracting citations: {arxiv_id}")
            self.create_citation_json(survey_data, pdf_path)
        else:
            logger.info(f"[⏭️] Skipping citation extraction (download-only mode)")
        
        self.stats.increment('papers_processed')
        logger.info(f"[✅] Successfully processed: {arxiv_id}")
        return True
    
    def process_jsonl_line(self, line: str) -> bool:
        """Process a single line from surveyscope.jsonl."""
        try:
            survey_data = json.loads(line.strip())
            return self.process_survey_paper(survey_data)
        except json.JSONDecodeError as e:
            logger.error(f"[❌] JSON parsing error: {e}")
            return False
        except Exception as e:
            logger.error(f"[❌] Line processing error: {e}")
            return False
    
    def run(self, surveyscope_file: str = "surveyscope.jsonl", max_papers: Optional[int] = None) -> None:
        """Execute the main crawling process."""
        logger.info(f"[🚀] Starting SurveyScope crawling...")
        
        if not os.path.exists(surveyscope_file):
            logger.error(f"[❌] File not found: {surveyscope_file}")
            return
        
        processed_count = 0
        
        with open(surveyscope_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():  # Skip empty lines
                    continue
                
                success = self.process_jsonl_line(line)
                if success:
                    processed_count += 1
                
                # Check maximum papers limit
                if max_papers and processed_count >= max_papers:
                    logger.info(f"[⏹️] Reached maximum papers limit: {max_papers}")
                    break
                
                # Progress logging
                if line_num % 10 == 0:
                    logger.info(f"[📊] Processed {line_num} lines, {processed_count} papers successful")
        
        # Print final statistics
        self.stats.print_statistics(self.searcher)

    def save_paper_metadata(self, paper_data: Dict, original_title: str) -> None:
        """Save paper metadata to meta_survey.jsonl."""
        if not USE_SEMANTIC_SCHOLAR:
            logger.info(f"[⏭️] Semantic Scholar API disabled, skipping metadata save")
            return
            
        try:
            meta_file = self.survey_dir / "meta_survey.jsonl"
            
            # Add original title for reference
            metadata = {
                'original_title': original_title,
                'semantic_scholar_data': paper_data,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Append to meta_survey.jsonl
            with open(meta_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            
            logger.info(f"[💾] Paper metadata saved to meta_survey.jsonl")
            self.stats.increment('metadata_saved')
            
        except Exception as e:
            logger.error(f"[❌] Error saving paper metadata: {e}")

    def get_semantic_references(self, title: str) -> List[Dict]:
        """Get references from Semantic Scholar API using paper title with ROUGE-L validation."""
        if not USE_SEMANTIC_SCHOLAR or not self.semantic_api:
            logger.info(f"[⏭️] Semantic Scholar API disabled, skipping reference fetch")
            return []
            
        if not title:
            return []
            
        logger.info(f"[🔍] Fetching references from Semantic Scholar API for: {title}")
        
        try:
            # Get paper info using title
            paper_info = self.semantic_api.get_paper_info(title)
            
            if not paper_info:
                logger.warning(f"[❌] Paper not found in Semantic Scholar: {title}")
                return []
            
            # Extract paper data (handle both search results and direct queries)
            paper = paper_info.get('data', [paper_info])[0] if 'data' in paper_info else paper_info
            paper_id = paper.get('paperId')
            paper_title = paper.get('title')
            
            if not paper_id:
                logger.warning(f"[❌] No paper ID found in Semantic Scholar response")
                return []
            
            if not paper_title:
                logger.warning(f"[❌] No paper title found in Semantic Scholar response")
                return []
            
            # Validate title match using ROUGE-L
            clean_input_title = TextCleaner.clean_title_for_search(title)
            clean_found_title = TextCleaner.clean_title_for_search(paper_title)
            
            rouge_score = calc_rouge_scores(clean_input_title.lower(), clean_found_title.lower())
            
            # Use the same threshold as ArxivSearcher
            if rouge_score < ArxivSearcher.ROUGE_THRESHOLD:
                logger.warning(f"[⚠️] Title mismatch: ROUGE-L {rouge_score:.3f} < {ArxivSearcher.ROUGE_THRESHOLD}")
                logger.warning(f"[⚠️] Input: {clean_input_title}")
                logger.warning(f"[⚠️] Found: {clean_found_title}")
                self.stats.increment('title_match_failures')
                return []
            
            logger.info(f"[✅] Title match confirmed: ROUGE-L {rouge_score:.3f}")
            logger.info(f"[📄] Matched paper: {paper_title}")
            
            # Save paper metadata
            self.save_paper_metadata(paper, title)
            
            # Get references using paper ID
            references_result = self.semantic_api.get_references(paper_id)
            
            if not references_result or not references_result.get('data'):
                logger.warning(f"[❌] No references found in Semantic Scholar for paper: {paper_id}")
                return []
            
            # Convert Semantic Scholar references to our format
            references = []
            for ref_data in references_result['data']:
                cited_paper = ref_data.get('citedPaper', {})
                if cited_paper and cited_paper.get('title'):
                    ref = {
                        'title': cited_paper['title'],
                        'authors': [author['name'] for author in cited_paper.get('authors', [])],
                        'year': cited_paper.get('year'),
                        'venue': cited_paper.get('venue'),
                        'paperId': cited_paper.get('paperId'),
                        'citationCount': cited_paper.get('citationCount')
                    }
                    
                    # Try to extract arXiv ID if available
                    external_ids = cited_paper.get('externalIds', {})
                    if external_ids and external_ids.get('ArXiv'):
                        ref['arxiv_id'] = external_ids['ArXiv']
                    
                    references.append(ref)
            
            logger.info(f"[✅] Retrieved {len(references)} references from Semantic Scholar")
            self.stats.increment('semantic_references_found', len(references))
            return references
            
        except Exception as e:
            logger.error(f"[❌] Error fetching references from Semantic Scholar: {e}")
            self.stats.increment('semantic_api_failures')
            return []


def enable_semantic_scholar(enabled: bool = True):
    """Enable or disable Semantic Scholar API globally."""
    global USE_SEMANTIC_SCHOLAR
    USE_SEMANTIC_SCHOLAR = enabled
    status = "enabled" if enabled else "disabled"
    logger.info(f"[🌐] Semantic Scholar API {status} globally")


def main():
    """Main function demonstrating different crawler configurations."""
    # Configuration options:
    
    # Enable Semantic Scholar API globally (optional):
    # enable_semantic_scholar(True)
    
    # Option 1: Download only (PDFs and TeX sources, no citation extraction)
    # crawler = SurveyScopeCrawler(download_only=True, download_tex=True)
    
    # Option 2: Full processing (PDFs, TeX, Markdown, Citations)
    # crawler = SurveyScopeCrawler(download_only=False, download_tex=True)
    
    # Option 3: PDFs and citations only (no TeX sources) - Default mode
    # crawler = SurveyScopeCrawler(download_only=False, download_tex=False)
    
    crawler = SurveyScopeCrawler(download_only=True, download_tex=False)

    crawler.run(surveyscope_file="surveyscope.jsonl", max_papers=None)


if __name__ == "__main__":
    main() 
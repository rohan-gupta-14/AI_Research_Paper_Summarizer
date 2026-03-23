"""
PDF Parser Module
Handles PDF text extraction with metadata detection.
"""

import re
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import pdfplumber


class PDFParser:
    """
    A robust PDF parser that extracts text and metadata from research papers.
    Uses pdfplumber for reliable text extraction.
    """

    def __init__(self):
        self.text = ""
        self.pages = []
        self.metadata = {}
        self.num_pages = 0

    def extract_from_bytes(self, pdf_bytes: bytes) -> Dict:
        """
        Extract text and metadata from PDF bytes.

        Args:
            pdf_bytes: PDF file as bytes

        Returns:
            Dictionary containing extracted text, pages, and metadata
        """
        try:
            # Open PDF from bytes
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                self.num_pages = len(pdf.pages)
                self.pages = []
                all_text = []

                # Extract text from each page
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    self.pages.append({
                        'page_num': i + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                    all_text.append(page_text)

                self.text = "\n\n".join(all_text)

                # Extract metadata
                self.metadata = self._extract_metadata(pdf, self.text)

            return {
                'success': True,
                'text': self.text,
                'pages': self.pages,
                'metadata': self.metadata,
                'num_pages': self.num_pages,
                'total_chars': len(self.text)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'pages': [],
                'metadata': {},
                'num_pages': 0,
                'total_chars': 0
            }

    def _extract_metadata(self, pdf, text: str) -> Dict:
        """
        Extract metadata from PDF including title, authors, and abstract.

        Args:
            pdf: pdfplumber PDF object
            text: Full extracted text

        Returns:
            Dictionary with metadata fields
        """
        metadata = {
            'title': '',
            'authors': '',
            'abstract': '',
            'keywords': [],
            'year': '',
            'doi': ''
        }

        # Try to get PDF info metadata
        if pdf.metadata:
            if pdf.metadata.get('Title'):
                metadata['title'] = pdf.metadata.get('Title', '')
            if pdf.metadata.get('Author'):
                metadata['authors'] = pdf.metadata.get('Author', '')

        # Extract title from first page if not in metadata
        if not metadata['title'] and self.pages:
            first_page_text = self.pages[0]['text']
            lines = first_page_text.split('\n')
            # Title is usually the first non-empty line
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    metadata['title'] = line
                    break

        # Extract abstract
        abstract_match = re.search(
            r'(?:abstract|summary)[:\s]*\n*(.*?)(?:\n\s*(?:introduction|keywords|1\.|I\.))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 100:
                metadata['abstract'] = abstract[:2000]  # Limit length

        # Extract keywords
        keywords_match = re.search(
            r'(?:keywords?|key\s*words?|index\s*terms?)[:\s]*([^\n]+)',
            text,
            re.IGNORECASE
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            # Split by common delimiters
            keywords = re.split(r'[;,•·]', keywords_text)
            metadata['keywords'] = [k.strip() for k in keywords if k.strip() and len(k.strip()) < 50][:10]

        # Extract year (look for 4-digit years between 1990-2030)
        year_match = re.search(r'\b(19[9]\d|20[0-2]\d)\b', text[:3000])
        if year_match:
            metadata['year'] = year_match.group(1)

        # Extract DOI
        doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', text)
        if doi_match:
            metadata['doi'] = doi_match.group(1).rstrip('.')

        return metadata

    def get_text_chunks(self, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split text into overlapping chunks for processing.

        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not self.text:
            return []

        chunks = []
        text = self.text
        start = 0
        chunk_id = 0

        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                if last_period > start:
                    end = last_period + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'char_count': len(chunk_text)
                })
                chunk_id += 1

            # Move start position with overlap
            start = end - overlap if end < len(text) else end

        return chunks

    def get_sections(self) -> List[Dict]:
        """
        Attempt to identify and extract paper sections.

        Returns:
            List of section dictionaries with title and content
        """
        if not self.text:
            return []

        # Common section patterns in research papers
        section_patterns = [
            r'^(?:\d+\.?\s*)?(?:INTRODUCTION|Introduction)\s*$',
            r'^(?:\d+\.?\s*)?(?:BACKGROUND|Background|RELATED\s+WORK|Related\s+Work|LITERATURE\s+REVIEW|Literature\s+Review)\s*$',
            r'^(?:\d+\.?\s*)?(?:METHODOLOGY|Methodology|METHODS?|Methods?|APPROACH|Approach)\s*$',
            r'^(?:\d+\.?\s*)?(?:RESULTS?|Results?|EXPERIMENTS?|Experiments?|EVALUATION|Evaluation)\s*$',
            r'^(?:\d+\.?\s*)?(?:DISCUSSION|Discussion)\s*$',
            r'^(?:\d+\.?\s*)?(?:CONCLUSION|Conclusion|CONCLUSIONS?|Conclusions?)\s*$',
            r'^(?:\d+\.?\s*)?(?:REFERENCES?|References?|BIBLIOGRAPHY|Bibliography)\s*$',
        ]

        combined_pattern = '|'.join(f'({p})' for p in section_patterns)

        sections = []
        lines = self.text.split('\n')
        current_section = {'title': 'Preamble', 'content': [], 'start_line': 0}

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if re.match(combined_pattern, line_stripped, re.MULTILINE):
                # Save previous section
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section)
                # Start new section
                current_section = {
                    'title': line_stripped,
                    'content': [],
                    'start_line': i
                }
            else:
                current_section['content'].append(line)

        # Add final section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)

        return sections

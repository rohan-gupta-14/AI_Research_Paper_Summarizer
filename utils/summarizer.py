"""
Paper Summarizer Module
Uses LLM (OpenAI) to generate concise summaries of research papers.
"""

import os
from typing import Dict, Optional, Callable
from openai import OpenAI


class PaperSummarizer:
    """
    Generates comprehensive summaries of research papers using LLM.
    Handles large documents through chunked processing.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the summarizer with OpenAI API key.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        self.model = "openai/gpt-3.5-turbo"  # OpenRouter model path
        self.max_context_length = 100000  # Approximate token limit

    def generate_summary(
        self,
        text: str,
        metadata: Dict = None,
        progress_callback: Optional[Callable] = None,
        length: str = "medium"
    ) -> Dict:
        """
        Generate a comprehensive summary of the research paper.

        Args:
            text: Full text of the paper
            metadata: Optional metadata (title, authors, etc.)
            progress_callback: Optional callback for progress updates
            length: Summary length - "short", "medium", or "long"

        Returns:
            Dictionary containing the generated summary
        """
        if not text or len(text.strip()) < 100:
            return {
                'success': False,
                'error': 'Insufficient text content for summarization',
                'summary': ''
            }

        try:
            # Prepare context with available metadata
            context_parts = []
            if metadata:
                if metadata.get('title'):
                    context_parts.append(f"Title: {metadata['title']}")
                if metadata.get('authors'):
                    context_parts.append(f"Authors: {metadata['authors']}")
                if metadata.get('abstract'):
                    context_parts.append(f"Abstract: {metadata['abstract']}")

            context_info = "\n".join(context_parts) if context_parts else ""

            # Truncate text if too long (roughly 4 chars per token)
            max_chars = self.max_context_length * 3
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Text truncated due to length...]"

            if progress_callback:
                progress_callback(0.2, "Preparing document for summarization...")

            # Set length-specific instructions and token limits
            length_config = {
                "short": {
                    "instruction": "Provide a brief, concise summary in 2-3 short paragraphs. Focus only on the most essential points.",
                    "max_tokens": 600
                },
                "medium": {
                    "instruction": "Provide a balanced summary in 4-5 paragraphs covering all key aspects of the research.",
                    "max_tokens": 1200
                },
                "long": {
                    "instruction": "Provide a detailed, comprehensive summary in 6 or more paragraphs. Include thorough analysis of methodology, findings, implications, and context.",
                    "max_tokens": 2500
                }
            }

            config = length_config.get(length, length_config["medium"])

            # Create the prompt for summary generation
            prompt = f"""You are an expert academic researcher. Analyze the following research paper and provide a summary.

{f"Paper Information:{chr(10)}{context_info}{chr(10)}{chr(10)}" if context_info else ""}Paper Content:
{text}

{config["instruction"]}

Structure your summary with these sections:

1. **Overview**: What is this paper about? What problem does it address?

2. **Research Question/Objective**: What specific question or objective does this research aim to address?

3. **Key Findings**: What are the main results and discoveries?

4. **Significance**: Why is this research important?

5. **Conclusion**: What do the authors conclude?

Write in clear, academic language."""

            if progress_callback:
                progress_callback(0.4, "Generating summary...")

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic researcher who specializes in analyzing and summarizing research papers. Provide clear, accurate, and well-structured summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=config["max_tokens"]
            )

            summary = response.choices[0].message.content

            if progress_callback:
                progress_callback(1.0, "Summary complete!")

            return {
                'success': True,
                'summary': summary,
                'model_used': self.model,
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'summary': ''
            }

    def generate_abstract_summary(self, text: str, max_words: int = 150) -> Dict:
        """
        Generate a brief abstract-style summary.

        Args:
            text: Full text of the paper
            max_words: Maximum words for the summary

        Returns:
            Dictionary containing the brief summary
        """
        try:
            # Truncate if needed
            max_chars = 50000
            truncated_text = text[:max_chars] if len(text) > max_chars else text

            prompt = f"""Summarize this research paper in approximately {max_words} words.
Write as if creating an abstract - concise, informative, and covering the key points.

Paper:
{truncated_text}

Provide only the summary, no additional formatting or headers."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating concise academic summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            return {
                'success': True,
                'summary': response.choices[0].message.content
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'summary': ''
            }

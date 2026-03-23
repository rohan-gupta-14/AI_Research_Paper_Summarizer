"""
Insight Extractor Module
Extracts structured insights from research papers using LLM.
"""

import os
from typing import Dict, Optional, Callable
from openai import OpenAI


class InsightExtractor:
    """
    Extracts detailed insights from research papers including:
    - Key findings
    - Methodology
    - Contributions
    - Limitations
    - Future research directions
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the insight extractor with OpenAI API key.

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
        self.model = "openai/gpt-3.5-turbo"

    def extract_all_insights(
        self,
        text: str,
        metadata: Dict = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Extract all insights from the research paper in a single call.

        Args:
            text: Full text of the paper
            metadata: Optional metadata
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing all extracted insights
        """
        if not text or len(text.strip()) < 100:
            return {
                'success': False,
                'error': 'Insufficient text content for insight extraction'
            }

        try:
            # Truncate text if too long
            max_chars = 100000 * 3  # Roughly 100k tokens
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Text truncated due to length...]"

            if progress_callback:
                progress_callback(0.1, "Analyzing paper structure...")

            # Build context
            context = ""
            if metadata:
                if metadata.get('title'):
                    context += f"Title: {metadata['title']}\n"
                if metadata.get('abstract'):
                    context += f"Abstract: {metadata['abstract']}\n"

            prompt = f"""Analyze this research paper and extract detailed insights.

{f"Paper Context:{chr(10)}{context}{chr(10)}" if context else ""}Paper Content:
{text}

Provide a comprehensive analysis with the following sections. Use markdown formatting.

## Key Insights and Important Findings
List the most important discoveries, results, and insights from this paper. Be specific about:
- Quantitative results (numbers, percentages, improvements)
- Qualitative findings
- Novel observations

## Methodology
Describe the research methodology:
- Research design (experimental, observational, theoretical, etc.)
- Data collection methods
- Analysis techniques
- Tools or frameworks used
- Sample size or dataset information (if applicable)

## Key Contributions
What are the main contributions of this paper to the field?
- Theoretical contributions
- Practical contributions
- Novel techniques or approaches introduced

## Limitations
What are the acknowledged or apparent limitations of this research?
- Methodological limitations
- Scope limitations
- Data limitations
- Generalizability concerns

## Future Research Directions
What future research does this paper suggest or enable?
- Explicitly mentioned future work
- Implied research opportunities
- Open questions that remain

Be thorough but concise. Focus on extracting concrete, actionable insights."""

            if progress_callback:
                progress_callback(0.3, "Extracting key insights...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst who excels at extracting key insights and synthesizing information from academic papers. Be thorough, accurate, and specific."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=3000
            )

            if progress_callback:
                progress_callback(0.8, "Processing extracted insights...")

            full_insights = response.choices[0].message.content

            # Parse the response into sections
            insights = self._parse_insights(full_insights)
            insights['full_text'] = full_insights
            insights['success'] = True
            insights['tokens_used'] = response.usage.total_tokens if response.usage else 0

            if progress_callback:
                progress_callback(1.0, "Insight extraction complete!")

            return insights

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _parse_insights(self, text: str) -> Dict:
        """
        Parse the LLM response into structured sections.

        Args:
            text: Full insight text from LLM

        Returns:
            Dictionary with parsed sections
        """
        sections = {
            'key_insights': '',
            'methodology': '',
            'contributions': '',
            'limitations': '',
            'future_research': ''
        }

        # Define section markers
        markers = [
            ('key_insights', ['## Key Insights', '## Key Findings', '## Important Findings']),
            ('methodology', ['## Methodology', '## Research Methodology', '## Methods']),
            ('contributions', ['## Key Contributions', '## Contributions', '## Main Contributions']),
            ('limitations', ['## Limitations', '## Study Limitations']),
            ('future_research', ['## Future Research', '## Future Directions', '## Future Work'])
        ]

        # Find each section
        lines = text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Check if this line is a section header
            found_section = None
            for section_key, headers in markers:
                for header in headers:
                    if line.strip().lower().startswith(header.lower()):
                        found_section = section_key
                        break
                if found_section:
                    break

            if found_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = found_section
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def extract_specific_insight(self, text: str, insight_type: str) -> Dict:
        """
        Extract a specific type of insight from the paper.

        Args:
            text: Paper text
            insight_type: Type of insight to extract
                         ('methodology', 'contributions', 'limitations', 'future')

        Returns:
            Dictionary with the extracted insight
        """
        prompts = {
            'methodology': """Describe the research methodology used in this paper in detail:
- Research design
- Data collection methods
- Analysis techniques
- Tools and frameworks
- Sample/dataset information""",

            'contributions': """What are the key contributions of this paper?
- Theoretical contributions
- Practical contributions
- Novel techniques introduced
- Improvements over prior work""",

            'limitations': """What are the limitations of this research?
- Methodological limitations
- Data limitations
- Scope limitations
- Acknowledged weaknesses""",

            'future': """What future research directions does this paper suggest?
- Explicitly mentioned future work
- Open questions
- Potential extensions
- Research opportunities"""
        }

        if insight_type not in prompts:
            return {'success': False, 'error': f'Invalid insight type: {insight_type}'}

        try:
            max_chars = 80000
            truncated_text = text[:max_chars] if len(text) > max_chars else text

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing research papers. Provide specific, detailed analysis."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this paper:\n\n{truncated_text}\n\n{prompts[insight_type]}"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return {
                'success': True,
                'insight': response.choices[0].message.content,
                'type': insight_type
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_research_questions(self, text: str, num_questions: int = 5) -> Dict:
        """
        Generate potential research questions based on the paper.

        Args:
            text: Paper text
            num_questions: Number of questions to generate

        Returns:
            Dictionary with generated questions
        """
        try:
            max_chars = 60000
            truncated_text = text[:max_chars] if len(text) > max_chars else text

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate insightful research questions based on academic papers."
                    },
                    {
                        "role": "user",
                        "content": f"""Based on this research paper, generate {num_questions} thought-provoking research questions that could guide further investigation or discussion.

Paper:
{truncated_text}

Provide questions that:
1. Go beyond what the paper explicitly answers
2. Explore implications and applications
3. Identify gaps or areas for further research
4. Challenge assumptions or explore alternatives

Format as a numbered list."""
                    }
                ],
                temperature=0.5,
                max_tokens=800
            )

            return {
                'success': True,
                'questions': response.choices[0].message.content
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

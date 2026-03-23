"""
Chat Engine Module
Provides RAG-based Q&A functionality for research papers.
Uses vector embeddings and similarity search for context retrieval.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss


class ChatEngine:
    """
    RAG-based chat engine for answering questions about research papers.
    Uses FAISS for efficient similarity search and sentence-transformers for embeddings.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the chat engine.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        self.model = "openai/gpt-3.5-turbo"

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # FAISS index and document chunks
        self.index = None
        self.chunks = []
        self.paper_metadata = {}
        self.is_initialized = False

        # Chat history
        self.chat_history: List[Dict] = []

    def initialize_from_text(
        self,
        text: str,
        metadata: Dict = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> bool:
        """
        Initialize the chat engine with paper text.
        Creates embeddings and builds FAISS index.

        Args:
            text: Full paper text
            metadata: Paper metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            True if initialization successful
        """
        try:
            # Store metadata
            self.paper_metadata = metadata or {}

            # Create text chunks
            self.chunks = self._create_chunks(text, chunk_size, chunk_overlap)

            if not self.chunks:
                return False

            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in self.chunks]
            embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)

            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Build FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.index.add(embeddings.astype('float32'))

            self.is_initialized = True
            self.chat_history = []

            return True

        except Exception as e:
            print(f"Error initializing chat engine: {e}")
            return False

    def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 0

        # Clean the text
        text = ' '.join(text.split())

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for punct in ['. ', '? ', '! ', '\n']:
                    last_break = text.rfind(punct, start + chunk_size // 2, end)
                    if last_break > start:
                        end = last_break + 1
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start': start,
                    'end': end
                })
                chunk_id += 1

            start = end - overlap if end < len(text) else end

        return chunks

    def get_relevant_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve most relevant text chunks for a query.

        Args:
            query: User question
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with scores
        """
        if not self.is_initialized or not self.index:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Get relevant chunks with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(scores[0][i]),
                    'rank': i + 1
                })

        return results

    def ask(
        self,
        question: str,
        include_history: bool = True,
        top_k: int = 5
    ) -> Dict:
        """
        Answer a question about the paper using RAG.

        Args:
            question: User's question
            include_history: Whether to include chat history
            top_k: Number of context chunks to use

        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'Chat engine not initialized. Please upload a paper first.',
                'answer': ''
            }

        try:
            # Get relevant context
            relevant_chunks = self.get_relevant_context(question, top_k)

            if not relevant_chunks:
                return {
                    'success': False,
                    'error': 'Could not find relevant context for the question.',
                    'answer': ''
                }

            # Build context string
            context_parts = []
            for item in relevant_chunks:
                context_parts.append(item['chunk']['text'])

            context = "\n\n---\n\n".join(context_parts)

            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful research assistant answering questions about a specific research paper.

Paper Title: {self.paper_metadata.get('title', 'Unknown')}

IMPORTANT RULES:
1. ONLY answer based on the provided context from the paper
2. If the answer is not in the context, say "I cannot find this information in the paper"
3. Be precise and cite specific details from the paper
4. If you're uncertain, indicate your level of confidence
5. Keep answers focused and relevant to the question"""
                }
            ]

            # Add chat history for context
            if include_history and self.chat_history:
                # Include last 3 exchanges for context
                for exchange in self.chat_history[-6:]:
                    messages.append(exchange)

            # Add current question with context
            messages.append({
                "role": "user",
                "content": f"""Context from the paper:
{context}

Question: {question}

Please answer based only on the provided context."""
            })

            # Get response from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )

            answer = response.choices[0].message.content

            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})

            return {
                'success': True,
                'answer': answer,
                'context_chunks': len(relevant_chunks),
                'top_relevance_score': relevant_chunks[0]['score'] if relevant_chunks else 0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': ''
            }

    def get_chat_history(self) -> List[Dict]:
        """
        Get the full chat history.

        Returns:
            List of chat messages
        """
        return self.chat_history

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []

    def export_chat_history(self) -> str:
        """
        Export chat history as formatted text.

        Returns:
            Formatted chat history string
        """
        if not self.chat_history:
            return "No chat history available."

        lines = []
        lines.append("=" * 60)
        lines.append("RESEARCH PAPER Q&A CONVERSATION")
        if self.paper_metadata.get('title'):
            lines.append(f"Paper: {self.paper_metadata['title']}")
        lines.append("=" * 60)
        lines.append("")

        for i in range(0, len(self.chat_history), 2):
            if i < len(self.chat_history):
                q = self.chat_history[i]
                lines.append(f"Q: {q['content']}")
                lines.append("")

            if i + 1 < len(self.chat_history):
                a = self.chat_history[i + 1]
                lines.append(f"A: {a['content']}")
                lines.append("")
                lines.append("-" * 40)
                lines.append("")

        return "\n".join(lines)

    def suggest_questions(self) -> List[str]:
        """
        Suggest relevant questions about the paper.

        Returns:
            List of suggested questions
        """
        if not self.is_initialized:
            return []

        suggestions = [
            "What is the main contribution of this paper?",
            "What methodology did the authors use?",
            "What are the key findings?",
            "What are the limitations of this research?",
            "How does this work compare to previous approaches?",
            "What datasets were used in this study?",
            "What future work do the authors suggest?"
        ]

        return suggestions

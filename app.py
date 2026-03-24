"""
Research Paper Insight Extractor
A Streamlit web application for analyzing research papers.
"""

import streamlit as st
import os
from io import BytesIO
from datetime import datetime

from utils.pdf_parser import PDFParser
from utils.summarizer import PaperSummarizer
from utils.insight_extractor import InsightExtractor
from utils.chat_engine import ChatEngine

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


st.set_page_config(
    page_title="AI Summarizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


def get_custom_css() -> str:
    """Generate custom CSS for the application."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global text color fix */
        html, body, [class*="css"], .stMarkdown, .stMarkdown p,
        .stMarkdown span, .stMarkdown div, p, span, div, li, ul, ol {
            font-family: 'Inter', -apple-system, sans-serif !important;
            -webkit-font-smoothing: antialiased;
            color: #0f172a !important;
        }

        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}

        [data-testid="stAppViewContainer"] {
            background: #ffffff;
        }

        .block-container {
            max-width: 1200px;
            padding: 0 2rem 4rem 2rem;
        }

        /* Navbar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 0;
        }

        .navbar-brand {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a !important;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .navbar-links {
            display: flex;
            gap: 0.5rem;
        }

        .nav-link {
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            color: #475569 !important;
            text-decoration: none;
            cursor: pointer;
            border: none;
            background: transparent;
            transition: all 0.15s ease;
        }

        .nav-link:hover {
            background: #f8fafc;
            color: #0f172a !important;
        }

        .nav-link.active {
            background: #6366f1;
            color: white !important;
        }

        /* Hero Section */
        .hero {
            text-align: center;
            padding: 4rem 2rem;
            background: #f8fafc;
            border-radius: 16px;
            margin: 2rem 0;
        }

        .hero h1 {
            font-size: 2.5rem;
            font-weight: 800;
            color: #0f172a !important;
            margin: 0 0 1rem 0;
            letter-spacing: -0.02em;
        }

        .hero p {
            font-size: 1.125rem;
            color: #475569 !important;
            margin: 0 auto 2rem auto;
            max-width: 600px;
            line-height: 1.7;
        }

        .hero-stats {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 2rem;
        }

        .hero-stat {
            text-align: center;
        }

        .hero-stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #6366f1 !important;
        }

        .hero-stat-label {
            font-size: 0.875rem;
            color: #94a3b8 !important;
            margin-top: 0.25rem;
        }

        /* Section Headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a !important;
            margin: 2rem 0 1.5rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #e2e8f0;
        }

        /* Cards */
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .card * {
            color: #0f172a !important;
        }

        .card p {
            color: #475569 !important;
        }

        .card-header {
            font-size: 1rem;
            font-weight: 600;
            color: #0f172a !important;
            margin-bottom: 1rem;
        }

        /* Summary content */
        .summary-content {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0;
            line-height: 1.8;
        }

        .summary-content * {
            color: #0f172a !important;
        }

        .summary-content h1, .summary-content h2, .summary-content h3,
        .summary-content h4, .summary-content h5, .summary-content h6 {
            color: #0f172a !important;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        .summary-content p {
            color: #475569 !important;
            margin-bottom: 1rem;
        }

        .summary-content strong, .summary-content b {
            color: #0f172a !important;
        }

        .summary-content ul, .summary-content ol {
            color: #475569 !important;
            padding-left: 1.5rem;
        }

        .summary-content li {
            color: #475569 !important;
            margin-bottom: 0.5rem;
        }

        /* Summary Length Buttons */
        .length-selector {
            display: flex;
            gap: 0.75rem;
            margin: 1rem 0;
        }

        /* Buttons */
        .stButton > button {
            background: #6366f1 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0.75rem 1.5rem;
            transition: background 0.15s ease;
        }

        .stButton > button:hover {
            background: #4f46e5 !important;
        }

        .stDownloadButton > button {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px;
            font-weight: 500;
        }

        .stDownloadButton > button:hover {
            border-color: #6366f1 !important;
            background: #eef2ff !important;
        }

        /* File Uploader */
        .stFileUploader > div {
            background: #f8fafc;
            border: 2px dashed #6366f1;
            border-radius: 12px;
            padding: 2rem;
        }

        .stFileUploader > div:hover {
            border-color: #4f46e5;
            background: #eef2ff;
        }

        .stFileUploader label {
            color: #0f172a !important;
            font-weight: 600 !important;
        }

        .stFileUploader small {
            color: #475569 !important;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            color: #0f172a !important;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] span {
            color: #475569 !important;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] button {
            color: #6366f1 !important;
            background: transparent !important;
            border: 1px solid #6366f1 !important;
        }

        /* Upload page container */
        .upload-container {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
        }

        /* Fix arrow icons */
        .stButton > button svg {
            display: none !important;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            color: #0f172a !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px #eef2ff !important;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem;
        }

        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 700;
        }

        /* Messages */
        .user-message {
            background: #eef2ff;
            border-left: 3px solid #6366f1;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin: 0.75rem 0 0.75rem 20%;
        }

        .user-message * {
            color: #0f172a !important;
        }

        .user-message strong {
            color: #6366f1 !important;
        }

        .assistant-message {
            background: #f8fafc;
            border-left: 3px solid #e2e8f0;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin: 0.75rem 20% 0.75rem 0;
        }

        .assistant-message * {
            color: #0f172a !important;
        }

        /* Expander */
        .stExpander {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }

        /* Progress */
        .stProgress > div > div {
            background: #6366f1;
        }

        /* Badges */
        .badge {
            display: inline-block;
            background: #eef2ff;
            color: #6366f1 !important;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        /* Info boxes */
        [data-testid="stInfo"], [data-testid="stSuccess"],
        [data-testid="stWarning"], [data-testid="stError"] {
            border-radius: 8px;
        }

        /* Markdown text overrides */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #0f172a !important;
        }

        .stMarkdown p, .stMarkdown li {
            color: #475569 !important;
        }

        .stMarkdown strong, .stMarkdown b {
            color: #0f172a !important;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 1.75rem;
            }

            .hero-stats {
                flex-direction: column;
                gap: 1.5rem;
            }

            .user-message, .assistant-message {
                margin-left: 0;
                margin-right: 0;
            }

            .navbar-links {
                flex-wrap: wrap;
            }
        }
    </style>
    """


def create_text_download(content: str, filename: str) -> bytes:
    """Create downloadable text file."""
    return content.encode('utf-8')


def create_pdf_download(content: str, title: str) -> bytes:
    """Create downloadable PDF file using reportlab."""
    if not REPORTLAB_AVAILABLE:
        return content.encode('utf-8')

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20
    ))

    story = []
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 0.5 * inch))

    lines = content.split('\n')
    for line in lines:
        if line.startswith('## ') or line.startswith('# '):
            header_text = line.lstrip('#').strip()
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(header_text, styles['CustomHeading']))
        elif line.startswith('**') and line.endswith('**'):
            bold_text = line.strip('*')
            story.append(Paragraph(f"<b>{bold_text}</b>", styles['CustomBody']))
        elif line.strip():
            safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_line, styles['CustomBody']))
        else:
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'current_page': 'home',
        'paper_text': None,
        'paper_metadata': None,
        'paper_pages': None,
        'summary': None,
        'insights': None,
        'chat_engine': None,
        'chat_history': [],
        'processing_complete': False,
        'current_file_name': None,
        'summary_length': 'medium'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_paper_state():
    """Reset all paper-related state for new upload."""
    st.session_state.paper_text = None
    st.session_state.paper_metadata = None
    st.session_state.paper_pages = None
    st.session_state.summary = None
    st.session_state.insights = None
    st.session_state.chat_engine = None
    st.session_state.chat_history = []
    st.session_state.processing_complete = False
    st.session_state.current_file_name = None


def render_navbar():
    """Render the navigation bar."""
    cols = st.columns([3, 5])

    with cols[0]:
        if st.button("AI Summarizer", key="brand", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()

    with cols[1]:
        nav_cols = st.columns(3)

        with nav_cols[0]:
            is_home_active = st.session_state.current_page == "home"
            home_type = "primary" if is_home_active else "secondary"
            if st.button("Home", key="nav_home", use_container_width=True, type=home_type):
                st.session_state.current_page = "home"
                st.rerun()

        with nav_cols[1]:
            is_upload_active = st.session_state.current_page == "upload"
            upload_type = "primary" if is_upload_active else "secondary"
            if st.button("Upload", key="nav_upload", use_container_width=True, type=upload_type):
                st.session_state.current_page = "upload"
                st.rerun()

        with nav_cols[2]:
            if st.button("Get Started", key="nav_getstarted", use_container_width=True, type="primary"):
                st.session_state.current_page = "upload"
                st.rerun()

    st.markdown("---")


def render_home_page():
    """Render the home/hero page."""
    st.markdown("""
    <div class="hero">
        <h1>AI Summarizer</h1>
        <p>Transform complex research papers into clear, actionable insights.
        Upload your PDF, choose your summary length, and let our system analyze
        the key findings, methodology, and contributions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Features section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">Smart Summaries</div>
            <p style="color: #475569; font-size: 0.9rem;">
            Generate short, medium, or detailed summaries tailored to your needs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">Key Insights</div>
            <p style="color: #475569; font-size: 0.9rem;">
            Extract methodology, contributions, limitations, and future research directions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-header">Interactive Q&A</div>
            <p style="color: #475569; font-size: 0.9rem;">
            Ask questions about the paper and get contextual answers.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started - Upload Paper", type="primary", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()


def render_upload_page():
    """Render the upload page."""
    st.markdown('<h2 class="section-header" style="color: #0f172a;">Upload Your Research Paper</h2>', unsafe_allow_html=True)

    # Show results navigation if paper is uploaded
    if st.session_state.paper_text:
        st.markdown('<h3 style="color: #0f172a; margin-bottom: 1rem;">Analysis Options</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            disabled = not st.session_state.processing_complete
            if st.button("Summary", key="goto_summary", use_container_width=True, disabled=disabled):
                st.session_state.current_page = "summary"
                st.rerun()
        with col2:
            disabled = not st.session_state.processing_complete
            if st.button("Key Insights", key="goto_insights", use_container_width=True, disabled=disabled):
                st.session_state.current_page = "insights"
                st.rerun()
        with col3:
            disabled = not st.session_state.processing_complete
            if st.button("Discussion", key="goto_discussion", use_container_width=True, disabled=disabled):
                st.session_state.current_page = "discussion"
                st.rerun()
        with col4:
            disabled = not st.session_state.processing_complete
            if st.button("Export", key="goto_export", use_container_width=True, disabled=disabled):
                st.session_state.current_page = "export"
                st.rerun()

        if not st.session_state.processing_complete:
            st.caption("Analyze the paper first to access these options")
        st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drag and drop a PDF file or click to browse",
        type=['pdf'],
        help="Upload a research paper in PDF format"
    )

    if uploaded_file is None:
        if st.session_state.paper_metadata:
            st.info("A paper is already loaded. Upload another PDF to replace it.")
        return

    # Check if this is a new file
    if st.session_state.current_file_name != uploaded_file.name:
        reset_paper_state()
        st.session_state.current_file_name = uploaded_file.name

    # Process the uploaded file
    if st.session_state.paper_text is None:
        with st.spinner("Extracting text from PDF..."):
            parser = PDFParser()
            result = parser.extract_from_bytes(uploaded_file.getvalue())

            if result['success']:
                st.session_state.paper_text = result['text']
                st.session_state.paper_metadata = result['metadata']
                st.session_state.paper_pages = result['pages']
                st.success(f"Successfully extracted {result['num_pages']} pages ({result['total_chars']:,} characters)")
            else:
                st.error(f"Failed to extract text: {result.get('error', 'Unknown error')}")
                return

    # Display paper metadata
    if st.session_state.paper_metadata:
        meta = st.session_state.paper_metadata

        st.markdown('<h3 style="margin-top: 2rem;">Paper Information</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            if meta.get('title'):
                st.markdown(f"**Title:** {meta['title']}")
            if meta.get('authors'):
                st.markdown(f"**Authors:** {meta['authors']}")
            if meta.get('year'):
                st.markdown(f'<span class="badge">Year: {meta["year"]}</span>', unsafe_allow_html=True)
            if meta.get('doi'):
                st.markdown(f'<span class="badge">DOI: {meta["doi"]}</span>', unsafe_allow_html=True)

        with col2:
            st.metric("Pages", len(st.session_state.paper_pages))
            st.metric("Characters", f"{len(st.session_state.paper_text):,}")

        if meta.get('abstract'):
            with st.expander("View Abstract"):
                st.write(meta['abstract'])

    st.markdown("---")

    # Summary length selection
    st.markdown("### Select Summary Length")

    col1, col2, col3 = st.columns(3)

    with col1:
        short_type = "primary" if st.session_state.summary_length == "short" else "secondary"
        if st.button("Short", use_container_width=True, type=short_type, key="len_short"):
            st.session_state.summary_length = "short"
            st.rerun()

    with col2:
        medium_type = "primary" if st.session_state.summary_length == "medium" else "secondary"
        if st.button("Medium", use_container_width=True, type=medium_type, key="len_medium"):
            st.session_state.summary_length = "medium"
            st.rerun()

    with col3:
        long_type = "primary" if st.session_state.summary_length == "long" else "secondary"
        if st.button("Long", use_container_width=True, type=long_type, key="len_long"):
            st.session_state.summary_length = "long"
            st.rerun()

    length_info = {
        "short": "Brief overview (2-3 paragraphs)",
        "medium": "Balanced summary (4-5 paragraphs)",
        "long": "Detailed analysis (6+ paragraphs)"
    }
    st.caption(f"Selected: {length_info[st.session_state.summary_length]}")

    st.markdown("---")

    # Analyze button
    button_label = "Analyze Paper" if not st.session_state.processing_complete else "Reanalyze Paper"
    if st.button(button_label, type="primary", use_container_width=True):
        process_paper()


def render_summary_page():
    """Render summary page."""
    st.markdown('<h2 class="section-header">Summary</h2>', unsafe_allow_html=True)

    if st.session_state.summary:
        st.markdown(f"""
        <div class="summary-content">
            {st.session_state.summary}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Summary not available. Please analyze a paper first.")


def render_insights_page():
    """Render insights page."""
    st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)

    if not st.session_state.insights:
        st.info("Insights not available. Please analyze a paper first.")
        return

    insights = st.session_state.insights

    if insights.get('key_insights'):
        st.markdown("### Key Findings")
        st.markdown(insights['key_insights'])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if insights.get('methodology'):
            st.markdown("### Methodology")
            st.markdown(insights['methodology'])

    with col2:
        if insights.get('contributions'):
            st.markdown("### Key Contributions")
            st.markdown(insights['contributions'])

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        if insights.get('limitations'):
            st.markdown("### Limitations")
            st.markdown(insights['limitations'])

    with col4:
        if insights.get('future_research'):
            st.markdown("### Future Research")
            st.markdown(insights['future_research'])


def render_discussion_page():
    """Render discussion/chat page."""
    st.markdown('<h2 class="section-header">Discussion</h2>', unsafe_allow_html=True)

    if st.session_state.chat_engine is None and st.session_state.paper_text:
        with st.spinner("Initializing chat engine..."):
            try:
                chat_engine = ChatEngine()
                success = chat_engine.initialize_from_text(
                    st.session_state.paper_text,
                    st.session_state.paper_metadata
                )
                if success:
                    st.session_state.chat_engine = chat_engine
            except Exception as e:
                st.error(f"Failed to initialize chat: {str(e)}")

    if not st.session_state.chat_engine:
        st.info("Chat engine is initializing...")
        return

    with st.expander("Sample Questions"):
        suggestions = st.session_state.chat_engine.suggest_questions()
        cols = st.columns(2)
        for i, question in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(question, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()

    st.markdown("### Conversation")

    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Response:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Ask a question...",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask a question about the paper..."
        )
    with col2:
        send_clicked = st.button("Send", type="primary", use_container_width=True)

    if 'pending_question' in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        send_clicked = True

    if question and send_clicked:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.ask(question)

            if response['success']:
                st.session_state.chat_history.append({'role': 'user', 'content': question})
                st.session_state.chat_history.append({'role': 'assistant', 'content': response['answer']})
                st.rerun()
            else:
                st.error(f"Error: {response.get('error', 'Unknown error')}")

    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.chat_engine:
                st.session_state.chat_engine.clear_history()
            st.rerun()


def render_export_page():
    """Render export page."""
    st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Summary")
        if st.session_state.summary:
            summary_text = f"# Paper Summary\n\n{st.session_state.summary}"
            st.download_button(
                label="Download TXT",
                data=create_text_download(summary_text, "summary.txt"),
                file_name="paper_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
            if REPORTLAB_AVAILABLE:
                st.download_button(
                    label="Download PDF",
                    data=create_pdf_download(summary_text, "Paper Summary"),
                    file_name="paper_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("No summary available")

    with col2:
        st.markdown("### Insights")
        if st.session_state.insights and st.session_state.insights.get('full_text'):
            insights_text = f"# Paper Insights\n\n{st.session_state.insights['full_text']}"
            st.download_button(
                label="Download TXT",
                data=create_text_download(insights_text, "insights.txt"),
                file_name="paper_insights.txt",
                mime="text/plain",
                use_container_width=True
            )
            if REPORTLAB_AVAILABLE:
                st.download_button(
                    label="Download PDF",
                    data=create_pdf_download(insights_text, "Paper Insights"),
                    file_name="paper_insights.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("No insights available")

    with col3:
        st.markdown("### Discussion")
        if st.session_state.chat_history:
            chat_text = "# Discussion History\n\n"
            for msg in st.session_state.chat_history:
                role = "You" if msg['role'] == 'user' else "Response"
                chat_text += f"**{role}:** {msg['content']}\n\n"
            st.download_button(
                label="Download TXT",
                data=create_text_download(chat_text, "discussion.txt"),
                file_name="discussion_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("No discussion history")

    st.divider()
    st.markdown("### Complete Report")

    if st.session_state.summary or st.session_state.insights:
        full_report = generate_full_report()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Full Report (TXT)",
                data=create_text_download(full_report, "full_report.txt"),
                file_name="full_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if REPORTLAB_AVAILABLE:
                st.download_button(
                    label="Download Full Report (PDF)",
                    data=create_pdf_download(full_report, "Research Paper Analysis"),
                    file_name="full_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


def process_paper():
    """Process the uploaded paper: generate summary and extract insights."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        summarizer = PaperSummarizer()
        extractor = InsightExtractor()

        status_text.text("Generating summary...")
        progress_bar.progress(10)

        def update_summary_progress(progress, message):
            progress_bar.progress(int(10 + progress * 30))
            status_text.text(message)

        summary_result = summarizer.generate_summary(
            st.session_state.paper_text,
            st.session_state.paper_metadata,
            progress_callback=update_summary_progress,
            length=st.session_state.summary_length
        )

        if summary_result['success']:
            st.session_state.summary = summary_result['summary']
        else:
            st.warning(f"Summary generation failed: {summary_result.get('error')}")

        status_text.text("Extracting insights...")
        progress_bar.progress(50)

        def update_insight_progress(progress, message):
            progress_bar.progress(int(50 + progress * 40))
            status_text.text(message)

        insights_result = extractor.extract_all_insights(
            st.session_state.paper_text,
            st.session_state.paper_metadata,
            progress_callback=update_insight_progress
        )

        if insights_result['success']:
            st.session_state.insights = insights_result
        else:
            st.warning(f"Insight extraction failed: {insights_result.get('error')}")

        progress_bar.progress(100)
        status_text.text("Analysis complete!")

        st.session_state.processing_complete = True
        st.session_state.current_page = "summary"
        st.rerun()

    except Exception as e:
        st.error(f"Error processing paper: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def generate_full_report() -> str:
    """Generate a complete report combining all analysis results."""
    report_parts = []

    report_parts.append("=" * 60)
    report_parts.append("RESEARCH PAPER ANALYSIS REPORT")
    report_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_parts.append("=" * 60)
    report_parts.append("")

    if st.session_state.paper_metadata:
        meta = st.session_state.paper_metadata
        report_parts.append("## Paper Information")
        if meta.get('title'):
            report_parts.append(f"Title: {meta['title']}")
        if meta.get('authors'):
            report_parts.append(f"Authors: {meta['authors']}")
        if meta.get('year'):
            report_parts.append(f"Year: {meta['year']}")
        report_parts.append("")

    if st.session_state.summary:
        report_parts.append("## Summary")
        report_parts.append(st.session_state.summary)
        report_parts.append("")

    if st.session_state.insights and st.session_state.insights.get('full_text'):
        report_parts.append("## Detailed Insights")
        report_parts.append(st.session_state.insights['full_text'])
        report_parts.append("")

    if st.session_state.chat_history:
        report_parts.append("## Discussion")
        for msg in st.session_state.chat_history:
            role = "Q" if msg['role'] == 'user' else "A"
            report_parts.append(f"{role}: {msg['content']}")
            report_parts.append("")

    return "\n".join(report_parts)


def main():
    """Main application entry point."""
    initialize_session_state()

    # Set API keys
    os.environ['OPENAI_API_KEY'] = "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"
    os.environ['OPENROUTER_BASE_URL'] = "https://openrouter.ai/api/v1"

    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Render navbar
    render_navbar()

    # Render current page
    page = st.session_state.current_page

    if page == "home":
        render_home_page()
    elif page == "upload":
        render_upload_page()
    elif page == "summary":
        render_summary_page()
    elif page == "insights":
        render_insights_page()
    elif page == "discussion":
        render_discussion_page()
    elif page == "export":
        render_export_page()


if __name__ == "__main__":
    main()

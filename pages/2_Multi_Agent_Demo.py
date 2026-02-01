"""
CrewAI Multi-Agent Demo - Streamlit Page
MIT Professional Education: Agentic AI

Watch three AI agents collaborate in real-time:
- Researcher: Gathers information
- Writer: Creates content
- Editor: Polishes output
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import crew logic
try:
    from crews.research_crew import (
        run_research_crew,
        get_available_providers,
        check_ollama_running,
        check_ollama_model,
        PROVIDER_CONFIGS
    )
    CREW_AVAILABLE = True
except ImportError as e:
    CREW_AVAILABLE = False
    IMPORT_ERROR = str(e)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Multi-Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #6c757d;
    }
    .agent-researcher { border-left-color: #0066cc; }
    .agent-writer { border-left-color: #28a745; }
    .agent-editor { border-left-color: #9933cc; }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-waiting { background: #e9ecef; color: #6c757d; }
    .status-working { background: #cce5ff; color: #004085; }
    .status-done { background: #d4edda; color: #155724; }
    
    .provider-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .provider-card:hover {
        border-color: #0066cc;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .provider-selected {
        border-color: #0066cc;
        background: #f0f7ff;
    }
    
    .output-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .cost-badge {
        background: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .free-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<p class="main-header">🤖 Multi-Agent Research Team</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Watch three AI agents collaborate: Researcher → Writer → Editor</p>', unsafe_allow_html=True)

# Check if dependencies are available
if not CREW_AVAILABLE:
    st.error(f"""
    **CrewAI dependencies not installed.**
    
    Run the following to install:
    ```bash
    pip install crewai langchain-community langchain-openai
    ```
    
    Error: {IMPORT_ERROR}
    """)
    st.stop()


# =============================================================================
# SIDEBAR - PROVIDER SELECTION
# =============================================================================

st.sidebar.markdown("## ⚙️ Configuration")

# Get available providers
available_providers = get_available_providers()

if not available_providers:
    st.sidebar.error("No LLM providers available. Install langchain-community or langchain-openai.")
    st.stop()

# Provider selection
st.sidebar.markdown("### Select Provider")

provider_choice = st.sidebar.radio(
    "Choose your LLM provider:",
    options=list(available_providers.keys()),
    format_func=lambda x: PROVIDER_CONFIGS[x].display_name,
    help="Ollama runs locally (free). OpenAI requires an API key."
)

config = PROVIDER_CONFIGS[provider_choice]

# Provider-specific settings
if provider_choice == "ollama":
    st.sidebar.markdown("---")
    
    # Check Ollama status
    ollama_running = check_ollama_running()
    
    if ollama_running:
        st.sidebar.success("✅ Ollama is running")
        
        # Model selection for Ollama
        ollama_model = st.sidebar.selectbox(
            "Select Model",
            options=["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"],
            index=0,
            help="Choose which Ollama model to use"
        )
        
        # Check if model is available
        if not check_ollama_model(ollama_model):
            st.sidebar.warning(f"⚠️ Model '{ollama_model}' not found locally")
            st.sidebar.code(f"ollama pull {ollama_model}", language="bash")
    else:
        st.sidebar.error("❌ Ollama not running")
        st.sidebar.markdown("""
        **To start Ollama:**
        ```bash
        ollama serve
        ```
        
        **To install a model:**
        ```bash
        ollama pull llama3.2
        ```
        """)
        ollama_model = "llama3.2"

elif provider_choice == "openai":
    st.sidebar.markdown("---")
    
    # API key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ API key required")
        st.sidebar.markdown("[Get an API key →](https://platform.openai.com/api-keys)")
    
    # Model selection for OpenAI
    openai_model = st.sidebar.selectbox(
        "Select Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini is recommended (fast & cheap)"
    )
    
    # Cost estimate
    if openai_model == "gpt-4o-mini":
        st.sidebar.markdown('<span class="cost-badge">~$0.01 per run</span>', unsafe_allow_html=True)
    elif openai_model == "gpt-4o":
        st.sidebar.markdown('<span class="cost-badge">~$0.10 per run</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="cost-badge">~$0.005 per run</span>', unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# How it works
with st.expander("ℹ️ How This Works", expanded=False):
    st.markdown("""
    This demo shows **multi-agent collaboration** using CrewAI:
    
    | Agent | Role | What They Do |
    |-------|------|--------------|
    | 🔍 **Researcher** | Research Analyst | Gathers facts, statistics, and key insights |
    | ✍️ **Writer** | Content Writer | Transforms research into clear prose |
    | 📝 **Editor** | Editor | Polishes for clarity and accuracy |
    
    **The Process:**
    1. You provide a topic
    2. Researcher gathers information
    3. Writer creates a draft from the research
    4. Editor polishes the final output
    
    Each agent sees only the output of the previous agent — just like a real team!
    """)

st.markdown("---")

# Topic input
st.markdown("### 📋 Enter a Research Topic")

col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_area(
        "What would you like the team to research?",
        value="Research the current state of AI agents in customer service, including benefits, challenges, and real-world examples.",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Example Topics:**")
    examples = [
        "AI in healthcare",
        "Remote work trends",
        "Sustainable energy",
        "Quantum computing"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.topic = f"Research the current state of {ex.lower()}, including key developments, challenges, and future outlook."
            st.rerun()

# Use session state topic if set
if "topic" in st.session_state:
    topic = st.session_state.topic
    del st.session_state.topic

# Validate before running
can_run = True
error_message = ""

if provider_choice == "ollama":
    if not check_ollama_running():
        can_run = False
        error_message = "Ollama is not running. Start with: `ollama serve`"
elif provider_choice == "openai":
    if not api_key:
        can_run = False
        error_message = "Please enter your OpenAI API key in the sidebar."

# Run button
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    run_button = st.button(
        "🚀 Run Research Team",
        type="primary",
        use_container_width=True,
        disabled=not can_run
    )

if not can_run:
    st.warning(error_message)


# =============================================================================
# EXECUTION
# =============================================================================

if run_button and can_run:
    
    # Agent status display
    st.markdown("### 🤖 Agent Activity")
    
    agent_cols = st.columns(3)
    
    with agent_cols[0]:
        researcher_status = st.empty()
        researcher_status.markdown("""
        <div class="agent-card agent-researcher">
            <strong>🔍 Researcher</strong><br/>
            <span class="status-badge status-waiting">Waiting</span>
        </div>
        """, unsafe_allow_html=True)
    
    with agent_cols[1]:
        writer_status = st.empty()
        writer_status.markdown("""
        <div class="agent-card agent-writer">
            <strong>✍️ Writer</strong><br/>
            <span class="status-badge status-waiting">Waiting</span>
        </div>
        """, unsafe_allow_html=True)
    
    with agent_cols[2]:
        editor_status = st.empty()
        editor_status.markdown("""
        <div class="agent-card agent-editor">
            <strong>📝 Editor</strong><br/>
            <span class="status-badge status-waiting">Waiting</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Output containers
    output_container = st.container()
    
    # Update researcher to working
    researcher_status.markdown("""
    <div class="agent-card agent-researcher">
        <strong>🔍 Researcher</strong><br/>
        <span class="status-badge status-working">Working...</span>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(10)
    status_text.text("🔍 Researcher is gathering information...")
    
    # Prepare parameters
    run_params = {
        "topic": topic,
        "provider": provider_choice,
        "verbose": False
    }
    
    if provider_choice == "ollama":
        run_params["model"] = ollama_model
    elif provider_choice == "openai":
        run_params["api_key"] = api_key
        run_params["model"] = openai_model
    
    # Run the crew
    with st.spinner("Agents are working..."):
        result = run_research_crew(**run_params)
    
    # Update statuses based on result
    if result.success:
        # Researcher done
        researcher_status.markdown("""
        <div class="agent-card agent-researcher">
            <strong>🔍 Researcher</strong><br/>
            <span class="status-badge status-done">Complete ✓</span>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(40)
        
        # Writer done
        writer_status.markdown("""
        <div class="agent-card agent-writer">
            <strong>✍️ Writer</strong><br/>
            <span class="status-badge status-done">Complete ✓</span>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(70)
        
        # Editor done
        editor_status.markdown("""
        <div class="agent-card agent-editor">
            <strong>📝 Editor</strong><br/>
            <span class="status-badge status-done">Complete ✓</span>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(100)
        
        status_text.text("✅ All agents complete!")
        
        # Display output
        with output_container:
            st.markdown("### 📄 Final Output")
            st.markdown(f"""
            <div class="output-box">
            {result.output}
            </div>
            """, unsafe_allow_html=True)
            
            # Metadata
            st.markdown("---")
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Provider", result.provider.title())
            with meta_cols[1]:
                st.metric("Model", result.model)
            with meta_cols[2]:
                cost = "Free" if result.provider == "ollama" else "~$0.01"
                st.metric("Estimated Cost", cost)
            
            # Show individual agent outputs if available
            if result.task_outputs:
                with st.expander("🔍 View Individual Agent Outputs"):
                    for agent, output in result.task_outputs.items():
                        st.markdown(f"**{agent}:**")
                        st.markdown(output)
                        st.markdown("---")
    
    else:
        # Error occurred
        researcher_status.markdown("""
        <div class="agent-card agent-researcher">
            <strong>🔍 Researcher</strong><br/>
            <span class="status-badge" style="background:#f8d7da;color:#721c24;">Error</span>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar.progress(0)
        status_text.text("❌ Error occurred")
        
        st.error(f"**Error:** {result.error}")
        
        # Helpful troubleshooting
        if "ollama" in result.error.lower() or "connection" in result.error.lower():
            st.info("""
            **Troubleshooting Ollama:**
            1. Make sure Ollama is running: `ollama serve`
            2. Make sure the model is installed: `ollama pull llama3.2`
            3. Check if Ollama is accessible at http://localhost:11434
            """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

# Educational connection
st.markdown("### 📚 Module 2 Connection")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Concepts Demonstrated:**
    - Multi-agent collaboration (CrewAI)
    - Role-based agent design
    - Sequential task execution
    - Open source (Ollama) vs Closed source (OpenAI)
    """)

with col2:
    st.markdown("""
    **Try This:**
    - Run the same topic with Ollama and OpenAI
    - Compare output quality vs cost
    - Notice how agents hand off work
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
<p>MIT Professional Education: Agentic AI | Module 2</p>
<p>Multi-Agent Systems Demo</p>
</div>
""", unsafe_allow_html=True)

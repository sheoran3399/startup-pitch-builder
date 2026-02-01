"""
CrewAI Multi-Agent Demos
MIT Professional Education: Agentic AI

This package contains standalone crew definitions that can be run via CLI
or imported into Streamlit apps.

Note: CrewAI dependencies are optional. Install with:
    pip install -r requirements-crewai.txt
"""

try:
    from .research_crew import (
        run_research_crew,
        get_available_providers,
        check_ollama_running,
        check_ollama_model,
        PROVIDER_CONFIGS,
        CrewResult
    )
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    run_research_crew = None
    get_available_providers = None
    check_ollama_running = None
    check_ollama_model = None
    PROVIDER_CONFIGS = {}
    CrewResult = None

__all__ = [
    "run_research_crew",
    "get_available_providers", 
    "check_ollama_running",
    "check_ollama_model",
    "PROVIDER_CONFIGS",
    "CrewResult",
    "CREWAI_AVAILABLE"
]

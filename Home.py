"""
LLM Cost Explorer - Home Page
MIT Professional Education: Agentic AI
"""

import streamlit as st

st.set_page_config(
    page_title="LLM Cost Explorer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">💰 LLM Cost Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Understanding the Economics of Large Language Models at Scale</p>', unsafe_allow_html=True)

# Key insight box
st.markdown("""
<div class="highlight-box">
<h3>🎯 The Key Insight</h3>
<p style="font-size: 1.3rem; margin-bottom: 0;">
<strong>The same AI transaction can cost between $1 and $230</strong> depending on model choice — a 200x variance!
</p>
<p style="margin-top: 0.5rem;">
Understanding these economics is essential for any business considering AI implementation.
</p>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("### 📊 Why Pricing Matters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;">
    <p class="stat-number">4x</p>
    <p><strong>Output vs Input</strong><br/>Output tokens typically cost 4x more than input tokens</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;">
    <p class="stat-number">200x</p>
    <p><strong>Model Variance</strong><br/>Cost difference between cheapest and most expensive models</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f7ff; border-radius: 10px;">
    <p class="stat-number">90%</p>
    <p><strong>Potential Savings</strong><br/>By choosing GPT-4o-mini over GPT-4o for suitable tasks</p>
    </div>
    """, unsafe_allow_html=True)

# What you'll learn
st.markdown("---")
st.markdown("### 🎓 What You'll Explore")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
    <h4>📝 Token Counter</h4>
    <ul>
    <li>See how text converts to tokens in real-time</li>
    <li>Understand why "tokens ≠ words"</li>
    <li>Experiment with different prompt styles</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h4>📈 Scale Analysis</h4>
    <ul>
    <li>Project costs from 1K to 1M API calls</li>
    <li>See the compounding effect at scale</li>
    <li>Identify cost crossover points</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h4>💰 Model Comparison</h4>
    <ul>
    <li>Compare 10+ models from OpenAI, Anthropic, Google</li>
    <li>See input vs output cost breakdown</li>
    <li>Find the best model for your budget</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h4>🔬 Scenario Testing</h4>
    <ul>
    <li>What if prompts are shorter?</li>
    <li>What if responses are longer?</li>
    <li>Export data for your assignment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Getting started
st.markdown("---")
st.markdown("### 🚀 Get Started")
st.info("""
👈 **Click "LLM Cost Calculator" in the sidebar** to begin exploring.

You'll enter a prompt, set expected response length, choose your scale, and see instant cost comparisons across all major models.
""")

# Assignment connection
st.markdown("---")
st.markdown("### 📝 Assignment Connection")
st.success("""
This tool directly supports your assignment:

1. **Enter your business question** → Get real token counts
2. **Compare models** → See the $1 to $230 variance
3. **Scale to 10K and 1M calls** → Understand enterprise costs
4. **Export results** → Download CSV/JSON for your write-up
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
<p>MIT Professional Education: Agentic AI</p>
<p>Assignment Support Tool | No API Key Required</p>
</div>
""", unsafe_allow_html=True)

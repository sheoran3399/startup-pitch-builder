"""
LLM Cost Calculator
Interactive tool for understanding LLM API pricing at scale
Enhanced with additional visualizations
"""

import streamlit as st
import tiktoken
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="LLM Cost Calculator",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# PRICING DATA (Per Million Tokens) - Updated January 2025
# ============================================================================

PRICING_DATA = {
    # OpenAI Models
    "GPT-4o": {"input": 2.50, "output": 10.00, "provider": "OpenAI", "tier": "Premium", "context": "128K"},
    "GPT-4o-mini": {"input": 0.15, "output": 0.60, "provider": "OpenAI", "tier": "Economy", "context": "128K"},
    "GPT-4 Turbo": {"input": 10.00, "output": 30.00, "provider": "OpenAI", "tier": "Premium", "context": "128K"},
    "GPT-3.5 Turbo": {"input": 0.50, "output": 1.50, "provider": "OpenAI", "tier": "Legacy", "context": "16K"},
    
    # Anthropic Models
    "Claude Opus 4": {"input": 15.00, "output": 75.00, "provider": "Anthropic", "tier": "Premium", "context": "200K"},
    "Claude Sonnet 4": {"input": 3.00, "output": 15.00, "provider": "Anthropic", "tier": "Standard", "context": "200K"},
    "Claude Haiku 4.5": {"input": 1.00, "output": 5.00, "provider": "Anthropic", "tier": "Economy", "context": "200K"},
    
    # Google Models
    "Gemini 1.5 Pro": {"input": 1.25, "output": 5.00, "provider": "Google", "tier": "Standard", "context": "1M"},
    "Gemini 1.5 Flash": {"input": 0.075, "output": 0.30, "provider": "Google", "tier": "Economy", "context": "1M"},
    "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40, "provider": "Google", "tier": "Economy", "context": "1M"},
}

# Sample prompts for different scenarios
SAMPLE_PROMPTS = {
    "Customer Service (Short)": "What are your business hours?",
    "Customer Service (Medium)": "What is the process I need to go through to rent a car: prerequisites? documents to present? information to have on hand?",
    "Technical Support": "I'm getting an error message when I try to log into my account. The error says 'Authentication failed: invalid credentials'. I've tried resetting my password twice but keep getting the same error. Can you help me troubleshoot this issue?",
    "HR Query": "I'd like to understand our company's parental leave policy. What are the eligibility requirements, how much time off is provided, and what documentation do I need to submit?",
    "Sales Inquiry": "We're a mid-size manufacturing company looking to implement AI-powered quality control. Can you explain your product offerings, pricing tiers, implementation timeline, and what kind of ROI other manufacturers have seen?",
    "Complex Analysis": "Please analyze the following quarterly sales data and provide insights on trends, anomalies, and recommendations for next quarter. Include statistical analysis and actionable suggestions for the sales team.",
    "Custom": ""
}

# Expected response lengths (in tokens)
RESPONSE_LENGTHS = {
    "Brief (50-100 tokens)": 75,
    "Standard (300-500 tokens)": 400,
    "Detailed (800-1200 tokens)": 1000,
    "Comprehensive (1500-2500 tokens)": 2000,
    "Very Long (3000-4000 tokens)": 3500,
    "Custom": None
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_tokenizer():
    """Load the tiktoken tokenizer (cached)"""
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    if not text:
        return 0
    enc = get_tokenizer()
    return len(enc.encode(text))

def calculate_cost(input_tokens: int, output_tokens: int, model: str, num_calls: int) -> float:
    """Calculate total cost for a model"""
    pricing = PRICING_DATA[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"] * num_calls
    output_cost = (output_tokens / 1_000_000) * pricing["output"] * num_calls
    return input_cost + output_cost

def format_currency(amount: float) -> str:
    """Format as currency string"""
    if amount < 0.01:
        return f"${amount:.4f}"
    elif amount < 1:
        return f"${amount:.3f}"
    elif amount < 1000:
        return f"${amount:.2f}"
    else:
        return f"${amount:,.2f}"

def get_color_scale(values):
    """Return colors from green (low) to red (high)"""
    min_val, max_val = min(values), max(values)
    colors = []
    for v in values:
        ratio = (v - min_val) / (max_val - min_val) if max_val > min_val else 0
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        colors.append(f'rgb({r},{g},50)')
    return colors

# ============================================================================
# PAGE CONTENT
# ============================================================================

st.title("📊 LLM Cost Calculator")
st.markdown("""
Calculate and compare the costs of running LLM API calls across different providers and models.
This tool simulates what you would experience using the OpenAI API or similar services.
""")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Volume settings
st.sidebar.subheader("📈 Scale Settings")
volume_presets = {
    "Pilot (1,000/month)": 1000,
    "Small (10,000/month)": 10000,
    "Medium (100,000/month)": 100000,
    "Large (500,000/month)": 500000,
    "Enterprise (1,000,000/month)": 1000000,
    "Custom": None
}

volume_choice = st.sidebar.selectbox("Monthly API Calls", list(volume_presets.keys()), index=1)

if volume_choice == "Custom":
    num_calls = st.sidebar.number_input("Custom monthly calls", min_value=100, max_value=10000000, value=10000, step=1000)
else:
    num_calls = volume_presets[volume_choice]

st.sidebar.metric("Monthly Calls", f"{num_calls:,}")

# Filter options
st.sidebar.subheader("🔍 Filter Models")
all_providers = list(set(p["provider"] for p in PRICING_DATA.values()))
providers = st.sidebar.multiselect(
    "Providers",
    all_providers,
    default=all_providers
)

all_tiers = list(set(p["tier"] for p in PRICING_DATA.values()))
tiers = st.sidebar.multiselect(
    "Tiers",
    all_tiers,
    default=all_tiers
)

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔤 Token Counter", "💰 Cost Comparison", "📈 Scale Analysis", "🔬 Deep Dive"])

# ============================================================================
# TAB 1: TOKEN COUNTER
# ============================================================================

with tab1:
    st.header("Step 1: Measure Your Token Usage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input (Your Prompt)")
        
        prompt_choice = st.selectbox(
            "Select a sample prompt or write your own:",
            list(SAMPLE_PROMPTS.keys())
        )
        
        if prompt_choice == "Custom":
            input_text = st.text_area(
                "Enter your prompt:",
                height=150,
                placeholder="Type a business-related question here..."
            )
        else:
            input_text = st.text_area(
                "Prompt text:",
                value=SAMPLE_PROMPTS[prompt_choice],
                height=150
            )
        
        input_tokens = count_tokens(input_text)
        
        # Visual token meter
        st.metric("Input Tokens", f"{input_tokens:,}")
        
        # Show token breakdown for short texts
        if input_text and input_tokens <= 30:
            enc = get_tokenizer()
            tokens = enc.encode(input_text)
            token_words = [enc.decode([t]) for t in tokens]
            st.caption(f"**Token breakdown:** {' | '.join(token_words)}")
        
        # Character to token ratio
        if input_text:
            char_count = len(input_text)
            word_count = len(input_text.split())
            st.caption(f"📏 {char_count} characters | {word_count} words | {input_tokens} tokens")
            st.caption(f"📊 Ratio: ~{char_count/input_tokens:.1f} chars/token, ~{word_count/input_tokens:.2f} words/token")
    
    with col2:
        st.subheader("📤 Output (Expected Response)")
        
        response_choice = st.selectbox(
            "Expected response length:",
            list(RESPONSE_LENGTHS.keys())
        )
        
        if response_choice == "Custom":
            output_tokens = st.number_input(
                "Custom output tokens:",
                min_value=10,
                max_value=10000,
                value=400
            )
        else:
            output_tokens = RESPONSE_LENGTHS[response_choice]
        
        st.metric("Output Tokens", f"{output_tokens:,}")
        
        # Visual guide
        st.info("""
        **Token Estimation Guide:**
        - 1 token ≈ 4 characters in English
        - 1 token ≈ 0.75 words
        - 100 tokens ≈ 75 words ≈ 1 short paragraph
        - 500 tokens ≈ 375 words ≈ 1 page
        """)
        
        # Output length visualization
        output_examples = {
            "Tweet": 50,
            "Short answer": 100,
            "Paragraph": 200,
            "Email": 400,
            "Article": 1000,
            "Report": 2500
        }
        
        fig_output = go.Figure(go.Bar(
            x=list(output_examples.values()),
            y=list(output_examples.keys()),
            orientation='h',
            marker_color=['#2ecc71' if v <= output_tokens else '#ecf0f1' for v in output_examples.values()]
        ))
        fig_output.add_vline(x=output_tokens, line_dash="dash", line_color="red", 
                            annotation_text=f"Your selection: {output_tokens}")
        fig_output.update_layout(
            title="Output Length Comparison",
            xaxis_title="Tokens",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_output, use_container_width=True)
    
    # Summary metrics
    st.markdown("---")
    total_tokens = input_tokens + output_tokens
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Input Tokens", f"{input_tokens:,}")
    with col_b:
        st.metric("Output Tokens", f"{output_tokens:,}")
    with col_c:
        st.metric("Total per Call", f"{total_tokens:,}")
    with col_d:
        st.metric("Total Monthly", f"{total_tokens * num_calls:,.0f}")

# ============================================================================
# TAB 2: COST COMPARISON
# ============================================================================

with tab2:
    st.header("Step 2: Compare Costs Across Models")
    
    # Build comparison dataframe
    results = []
    for model, pricing in PRICING_DATA.items():
        if pricing["provider"] not in providers or pricing["tier"] not in tiers:
            continue
            
        input_cost = (input_tokens / 1_000_000) * pricing["input"] * num_calls
        output_cost = (output_tokens / 1_000_000) * pricing["output"] * num_calls
        total_cost = input_cost + output_cost
        cost_per_call = total_cost / num_calls if num_calls > 0 else 0
        
        results.append({
            "Model": model,
            "Provider": pricing["provider"],
            "Tier": pricing["tier"],
            "Context": pricing["context"],
            "Input Cost": input_cost,
            "Output Cost": output_cost,
            "Total Cost": total_cost,
            "Cost per Call": cost_per_call,
            "Input Rate": pricing["input"],
            "Output Rate": pricing["output"]
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("Total Cost")
    
    if len(df) == 0:
        st.warning("No models match your filter criteria. Please adjust the filters in the sidebar.")
    else:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        min_cost = df["Total Cost"].min()
        max_cost = df["Total Cost"].max()
        cheapest_model = df.loc[df["Total Cost"].idxmin(), "Model"]
        most_expensive = df.loc[df["Total Cost"].idxmax(), "Model"]
        
        with col1:
            st.metric("💚 Cheapest", format_currency(min_cost), cheapest_model)
        with col2:
            st.metric("💸 Most Expensive", format_currency(max_cost), most_expensive)
        with col3:
            if min_cost > 0:
                variance = max_cost / min_cost
                st.metric("📊 Cost Variance", f"{variance:.0f}x")
            else:
                st.metric("📊 Cost Variance", "N/A")
        with col4:
            savings = max_cost - min_cost
            st.metric("💰 Max Savings", format_currency(savings))
        
        st.markdown("---")
        
        # Main comparison chart
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            # Stacked bar chart showing input vs output
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Input Cost',
                x=df["Model"],
                y=df["Input Cost"],
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Bar(
                name='Output Cost',
                x=df["Model"],
                y=df["Output Cost"],
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                title=f"Monthly Cost Comparison ({num_calls:,} API calls)",
                barmode='stack',
                xaxis_tickangle=-45,
                height=450,
                yaxis_title="Monthly Cost (USD)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_table:
            st.subheader("Quick Reference")
            
            quick_df = df[["Model", "Total Cost"]].copy()
            quick_df["Total Cost"] = quick_df["Total Cost"].apply(format_currency)
            quick_df = quick_df.rename(columns={"Total Cost": "Monthly Cost"})
            
            st.dataframe(quick_df, use_container_width=True, hide_index=True)
        
        # Cost per call visualization
        st.subheader("💵 Cost Per API Call")
        
        fig_per_call = px.bar(
            df,
            x="Model",
            y="Cost per Call",
            color="Provider",
            title="Cost Per Single API Call",
            labels={"Cost per Call": "Cost ($)"},
            text=df["Cost per Call"].apply(lambda x: f"${x:.4f}")
        )
        fig_per_call.update_traces(textposition='outside')
        fig_per_call.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_per_call, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Cost Breakdown")
        
        display_df = df.copy()
        display_df["Input Cost"] = display_df["Input Cost"].apply(format_currency)
        display_df["Output Cost"] = display_df["Output Cost"].apply(format_currency)
        display_df["Total Cost"] = display_df["Total Cost"].apply(format_currency)
        display_df["Cost per Call"] = display_df["Cost per Call"].apply(lambda x: f"${x:.5f}")
        display_df["Input Rate"] = display_df["Input Rate"].apply(lambda x: f"${x:.3f}/1M")
        display_df["Output Rate"] = display_df["Output Rate"].apply(lambda x: f"${x:.2f}/1M")
        
        st.dataframe(
            display_df[["Model", "Provider", "Tier", "Context", "Input Rate", "Output Rate", "Input Cost", "Output Cost", "Total Cost"]],
            use_container_width=True,
            hide_index=True
        )

# ============================================================================
# TAB 3: SCALE ANALYSIS
# ============================================================================

with tab3:
    st.header("Step 3: Understand How Costs Scale")
    
    # Model selection for scaling
    models_to_compare = st.multiselect(
        "Select models to visualize:",
        list(PRICING_DATA.keys()),
        default=["GPT-4o", "GPT-4o-mini", "Claude Sonnet 4", "Gemini 1.5 Flash"]
    )
    
    if models_to_compare:
        volumes = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
        
        # Scaling line chart
        scale_data = []
        for vol in volumes:
            for model in models_to_compare:
                cost = calculate_cost(input_tokens, output_tokens, model, vol)
                scale_data.append({
                    "Volume": vol,
                    "Model": model,
                    "Cost": cost
                })
        
        scale_df = pd.DataFrame(scale_data)
        
        fig_scale = px.line(
            scale_df,
            x="Volume",
            y="Cost",
            color="Model",
            title="How Costs Scale with Monthly API Volume",
            labels={"Volume": "Monthly API Calls", "Cost": "Monthly Cost (USD)"},
            log_x=True,
            markers=True
        )
        fig_scale.add_vline(x=num_calls, line_dash="dash", line_color="gray",
                          annotation_text=f"Your selection: {num_calls:,}")
        fig_scale.update_layout(height=500)
        st.plotly_chart(fig_scale, use_container_width=True)
        
        # Cost multiplier table
        st.subheader("📊 Cost Multiplier Table")
        st.markdown("How much more expensive is each model compared to the cheapest option?")
        
        multiplier_data = []
        for vol in [10000, 100000, 1000000]:
            row = {"Volume": f"{vol:,}"}
            costs = {model: calculate_cost(input_tokens, output_tokens, model, vol) for model in models_to_compare}
            min_cost = min(costs.values())
            for model, cost in costs.items():
                multiplier = cost / min_cost if min_cost > 0 else 0
                row[model] = f"{multiplier:.1f}x ({format_currency(cost)})"
            multiplier_data.append(row)
        
        multiplier_df = pd.DataFrame(multiplier_data)
        st.dataframe(multiplier_df, use_container_width=True, hide_index=True)
        
        # Savings potential
        st.subheader("💰 Savings Potential")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # If using most expensive vs cheapest
            costs_at_scale = {model: calculate_cost(input_tokens, output_tokens, model, num_calls) 
                            for model in models_to_compare}
            max_model = max(costs_at_scale, key=costs_at_scale.get)
            min_model = min(costs_at_scale, key=costs_at_scale.get)
            
            savings = costs_at_scale[max_model] - costs_at_scale[min_model]
            annual_savings = savings * 12
            
            st.info(f"""
            **Model Switch Savings**
            
            Switching from **{max_model}** to **{min_model}**:
            - Monthly: **{format_currency(savings)}**
            - Annual: **{format_currency(annual_savings)}**
            """)
        
        with col2:
            # Prompt optimization savings
            if len(models_to_compare) > 0:
                sample_model = models_to_compare[0]
                current = calculate_cost(input_tokens, output_tokens, sample_model, num_calls)
                optimized = calculate_cost(int(input_tokens * 0.7), int(output_tokens * 0.7), sample_model, num_calls)
                prompt_savings = current - optimized
                
                st.info(f"""
                **Prompt Optimization Savings**
                
                30% reduction in tokens with **{sample_model}**:
                - Monthly: **{format_currency(prompt_savings)}**
                - Annual: **{format_currency(prompt_savings * 12)}**
                """)
        
        # Break-even analysis
        st.subheader("📈 When Does Model Choice Matter Most?")
        
        # Calculate at what volume the cost difference exceeds certain thresholds
        thresholds = [100, 500, 1000, 5000]
        
        if len(models_to_compare) >= 2:
            model_a = models_to_compare[0]
            model_b = models_to_compare[-1]  # Compare first and last selected
            
            cost_diff_per_call = abs(
                calculate_cost(input_tokens, output_tokens, model_a, 1) - 
                calculate_cost(input_tokens, output_tokens, model_b, 1)
            )
            
            if cost_diff_per_call > 0:
                st.markdown(f"**Comparing {model_a} vs {model_b}:**")
                
                threshold_data = []
                for threshold in thresholds:
                    calls_needed = threshold / cost_diff_per_call
                    threshold_data.append({
                        "Savings Target": f"${threshold}/month",
                        "API Calls Needed": f"{int(calls_needed):,}"
                    })
                
                st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4: DEEP DIVE
# ============================================================================

with tab4:
    st.header("Step 4: Deep Dive Analysis")
    
    # Input vs Output cost analysis
    st.subheader("⚖️ Why Output Tokens Cost More")
    
    st.markdown("""
    LLM providers charge more for output tokens because:
    1. **Generation is computationally expensive** - Each output token requires a forward pass through the model
    2. **Output tokens are sequential** - They can't be parallelized like input processing
    3. **Quality matters more** - Users judge the AI by its responses
    """)
    
    # Show the ratio for each model
    ratio_data = []
    for model, pricing in PRICING_DATA.items():
        ratio = pricing["output"] / pricing["input"]
        ratio_data.append({
            "Model": model,
            "Provider": pricing["provider"],
            "Input ($/1M)": pricing["input"],
            "Output ($/1M)": pricing["output"],
            "Output/Input Ratio": ratio
        })
    
    ratio_df = pd.DataFrame(ratio_data).sort_values("Output/Input Ratio", ascending=False)
    
    fig_ratio = px.bar(
        ratio_df,
        x="Model",
        y="Output/Input Ratio",
        color="Provider",
        title="Output-to-Input Price Ratio by Model",
        labels={"Output/Input Ratio": "Output ÷ Input Price"},
        text=ratio_df["Output/Input Ratio"].apply(lambda x: f"{x:.1f}x")
    )
    fig_ratio.update_traces(textposition='outside')
    fig_ratio.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Provider comparison
    st.subheader("🏢 Provider Comparison")
    
    provider_summary = []
    for provider in set(p["provider"] for p in PRICING_DATA.values()):
        provider_models = {k: v for k, v in PRICING_DATA.items() if v["provider"] == provider}
        min_input = min(v["input"] for v in provider_models.values())
        max_input = max(v["input"] for v in provider_models.values())
        min_output = min(v["output"] for v in provider_models.values())
        max_output = max(v["output"] for v in provider_models.values())
        
        provider_summary.append({
            "Provider": provider,
            "Models": len(provider_models),
            "Input Range": f"${min_input:.3f} - ${max_input:.2f}",
            "Output Range": f"${min_output:.2f} - ${max_output:.2f}",
            "Cheapest Model": min(provider_models.keys(), key=lambda k: provider_models[k]["input"] + provider_models[k]["output"])
        })
    
    st.dataframe(pd.DataFrame(provider_summary), use_container_width=True, hide_index=True)
    
    # Heatmap: Cost by prompt length and response length
    st.subheader("🗺️ Cost Heatmap: Prompt vs Response Length")
    
    selected_model = st.selectbox("Select model for heatmap:", list(PRICING_DATA.keys()), index=1)
    
    prompt_lengths = [25, 50, 100, 200, 500]
    response_lengths = [100, 250, 500, 1000, 2000]
    
    heatmap_data = []
    for p_len in prompt_lengths:
        row = []
        for r_len in response_lengths:
            cost = calculate_cost(p_len, r_len, selected_model, num_calls)
            row.append(cost)
        heatmap_data.append(row)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{r} tokens" for r in response_lengths],
        y=[f"{p} tokens" for p in prompt_lengths],
        colorscale='RdYlGn_r',
        text=[[format_currency(c) for c in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Prompt: %{y}<br>Response: %{x}<br>Cost: %{text}<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        title=f"Monthly Cost by Token Combination ({selected_model}, {num_calls:,} calls)",
        xaxis_title="Response Length",
        yaxis_title="Prompt Length",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.caption("💡 Notice how response length (x-axis) has a bigger impact on cost than prompt length (y-axis)")

# ============================================================================
# EXPORT SECTION
# ============================================================================

st.markdown("---")
st.subheader("📥 Export Your Analysis")

# Build export dataframe
if len(df) > 0:
    export_config = {
        "Configuration": {
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Monthly API Calls": num_calls,
            "Prompt Type": prompt_choice,
            "Response Length": response_choice
        },
        "Results": df.to_dict(orient="records")
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download CSV",
            data=df.to_csv(index=False),
            file_name="llm_cost_comparison.csv",
            mime="text/csv"
        )
    
    with col2:
        import json
        st.download_button(
            label="📋 Download JSON",
            data=json.dumps(export_config, indent=2),
            file_name="llm_cost_analysis.json",
            mime="application/json"
        )
    
    with col3:
        # Create a summary text
        summary = f"""LLM Cost Analysis Summary
========================
Generated by LLM Cost Explorer

Configuration:
- Input Tokens: {input_tokens}
- Output Tokens: {output_tokens}  
- Monthly API Calls: {num_calls:,}

Top Results:
- Cheapest: {cheapest_model} at {format_currency(min_cost)}/month
- Most Expensive: {most_expensive} at {format_currency(max_cost)}/month
- Variance: {max_cost/min_cost:.0f}x
- Potential Savings: {format_currency(max_cost - min_cost)}/month

Full Results:
"""
        for _, row in df.iterrows():
            summary += f"- {row['Model']}: {format_currency(row['Total Cost'])}/month\n"
        
        st.download_button(
            label="📝 Download Summary",
            data=summary,
            file_name="llm_cost_summary.txt",
            mime="text/plain"
        )

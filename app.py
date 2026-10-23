import streamlit as st
import time
import os
from pathlib import Path
import sys

# Ensure src is in python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "src"))

from src.rag.cloud_rag import CloudCyberRAG

# Page Configuration
st.set_page_config(
    page_title="CyberLLM RAG — AI Cybersecurity Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk / Modern Dark Styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #07090e 90%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    
    /* Header Styling */
    .cyber-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .cyber-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    
    /* Metrics & Badges */
    .cyber-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .badge-attack { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-cwe { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-cve { background-color: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-sigma { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Source Cards */
    .source-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .source-card:hover {
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Greetings! I am **CyberLLM**, your Cybersecurity AI Analyst powered by Retrieval-Augmented Generation (RAG).\n\nAsk me about **MITRE ATT&CK techniques, CVE vulnerabilities, CWE security flaws, or Sigma detection rules**.",
            "sources": []
        }
    ]

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Inference & Model Settings")
    
    provider_choice = st.selectbox(
        "LLM Provider",
        options=["Groq (Recommended - Free)", "OpenAI", "Demo Mode (Offline)"],
        index=0,
        help="Groq offers ultra-fast Llama-3.3-70B inference on a 100% free tier."
    )
    
    api_key = None
    if "Groq" in provider_choice:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get a free key in 30 seconds at console.groq.com"
        )
        if not api_key:
            st.info("💡 Tip: Enter a free key from [console.groq.com](https://console.groq.com) for real-time dynamic Llama-3.3-70B generation, or test with Demo mode.")
        provider_name = "groq"
        model_name = "llama-3.3-70b-versatile"
    elif "OpenAI" in provider_choice:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        provider_name = "openai"
        model_name = "gpt-4o-mini"
    else:
        provider_name = "demo"
        model_name = "offline-knowledge-engine"
    
    st.divider()
    st.markdown("### 📊 Knowledge Base Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Threat Docs", value="100K+ / Curated")
    with col2:
        st.metric(label="Embedding", value="MiniLM-L6")
    
    st.markdown("""
    **Indexed Intelligence Sources:**
    - 🛡️ **MITRE ATT&CK** (Enterprise Tactics & Techniques)
    - ⚠️ **CWE** (Common Weakness Enumerations)
    - 🚨 **NVD CVEs** (Critical & High Vulnerabilities)
    - 🔎 **Sigma Rules** (Log & Process Detection Signatures)
    """)
    
    st.divider()
    st.markdown("""
    👨‍💻 **Developer:** [Suyash Dhage](https://github.com/suyasshx)  
    📂 **Source Code:** [suyasshx/cyber-llm-rag](https://github.com/suyasshx/cyber-llm-rag)
    """)

# Main Layout
st.markdown('<div class="cyber-title">🛡️ CyberLLM RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-subtitle">Production-Ready Cybersecurity AI Assistant & Threat Intelligence Engine</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_explorer = st.tabs(["💬 Incident Response Q&A", "🔍 Threat Intelligence Explorer"])

@st.cache_resource
def get_rag_engine(provider, key, model):
    return CloudCyberRAG(
        llm_provider=provider,
        api_key=key,
        model_name=model
    )

rag_engine = get_rag_engine(provider_name, api_key, model_name)

# TAB 1: Chat Assistant
with tab_chat:
    # Quick Prompt Chips
    st.markdown("**⚡ Quick Threat Scenarios:**")
    cols = st.columns(4)
    quick_query = None
    if cols[0].button("Log4Shell Attack Analysis", use_container_width=True):
        quick_query = "Explain the Apache Log4j2 JNDI Remote Code Execution vulnerability (Log4Shell CVE-2021-44228) and how to mitigate it."
    if cols[1].button("Detect Mimikatz (Sigma)", use_container_width=True):
        quick_query = "How can security analysts detect Mimikatz OS credential dumping using Sigma rules and Windows event logs?"
    if cols[2].button("ATT&CK T1059 Scripting", use_container_width=True):
        quick_query = "What is MITRE ATT&CK Technique T1059 Command and Scripting Interpreter, and what are key defenses?"
    if cols[3].button("Remediate SQL Injection", use_container_width=True):
        quick_query = "What is CWE-89 SQL Injection, its security impact, and recommended remediation practices?"

    # Display Previous Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📑 Verified Threat Intelligence Sources ({len(msg['sources'])} retrieved)"):
                    for src in msg["sources"]:
                        badge_class = f"badge-{src.get('source_type', 'attack')}"
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="cyber-badge {badge_class}">{src.get("source_type", "Intel").upper()}</span>'
                            f'<b>{src.get("technique_id", "")} - {src.get("name", "")}</b> '
                            f'<span style="color:#58a6ff; float:right;">Similarity: {int(src.get("similarity", 0.8) * 100)}%</span>'
                            f'<p style="color:#8b949e; font-size:0.88rem; margin-top:6px;">{src.get("content", "")[:320]}...</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Handle Input (Text input or quick button)
    user_input = st.chat_input("Ask a cybersecurity question (e.g., How does T1003 credential dumping work?)...")
    prompt = quick_query or user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving threat intelligence and analyzing vectors..."):
                result = rag_engine.query(prompt, top_k=3)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                latency = result.get("latency_seconds", 0)

            st.markdown(answer)
            st.caption(f"⚡ Analyzed in {latency}s via {result.get('model', 'CyberLLM')}")

            if sources:
                with st.expander(f"📑 Verified Threat Intelligence Sources ({len(sources)} retrieved)"):
                    for src in sources:
                        badge_class = f"badge-{src.get('source_type', 'attack')}"
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="cyber-badge {badge_class}">{src.get("source_type", "Intel").upper()}</span>'
                            f'<b>{src.get("technique_id", "")} - {src.get("name", "")}</b> '
                            f'<span style="color:#58a6ff; float:right;">Similarity: {int(src.get("similarity", 0.8) * 100)}%</span>'
                            f'<p style="color:#8b949e; font-size:0.88rem; margin-top:6px;">{src.get("content", "")[:320]}...</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# TAB 2: Threat Knowledge Explorer
with tab_explorer:
    st.markdown("### 📚 Browse Indexed Threat Intelligence")
    st.write("Search directly across indexed MITRE ATT&CK techniques, CWE definitions, CVE advisories, and Sigma rules.")
    
    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        source_filter = st.selectbox(
            "Filter by Source Type",
            options=["All Sources", "MITRE ATT&CK", "CWE", "CVE", "Sigma Rules"]
        )
    with col_search:
        search_term = st.text_input("Search threat catalog", placeholder="Search by name, ID (e.g., T1059, CWE-89), or keyword...")
    
    type_map = {
        "MITRE ATT&CK": "attack",
        "CWE": "cwe",
        "CVE": "cve",
        "Sigma Rules": "sigma"
    }
    
    docs_to_show = rag_engine.documents
    if source_filter != "All Sources":
        target_type = type_map.get(source_filter)
        docs_to_show = [d for d in docs_to_show if d.get("source_type") == target_type]
        
    if search_term:
        term = search_term.lower()
        docs_to_show = [
            d for d in docs_to_show
            if term in d.get("name", "").lower() 
            or term in d.get("technique_id", "").lower() 
            or term in d.get("content", "").lower()
        ]

    st.markdown(f"**Found {len(docs_to_show)} entries:**")
    for doc in docs_to_show:
        badge_class = f"badge-{doc.get('source_type', 'attack')}"
        with st.expander(f"[{doc.get('source_type', '').upper()}] {doc.get('technique_id', '')} - {doc.get('name', '')}"):
            st.markdown(
                f'<span class="cyber-badge {badge_class}">{doc.get("source_type", "").upper()}</span> '
                f'<b>ID:</b> {doc.get("technique_id", "")} | <b>Reference:</b> <a href="{doc.get("url", "#")}" target="_blank">View Official Advisory ↗</a>',
                unsafe_allow_html=True
            )
            st.markdown(doc.get("content", ""))

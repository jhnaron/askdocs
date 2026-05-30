import streamlit as st
import anthropic
from index import get_or_build_index, DOCS_DIR

MAX_INPUT_LENGTH = 250
TOP_K = 4
MAX_HISTORY_TURNS = 6

PAPERS = {
    "2026-Jan-08_constitutional-classifiers-plus.pdf": {
        "label": "2026 Jan — Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks",
        "summary": (
            "Released on 2026 Jan 08. "
            "A follow-up to the original Constitutional Classifiers paper, introducing an enhanced "
            "production-grade system that achieves 40x computational cost reduction while maintaining "
            "strong jailbreak robustness. Validated through over 1,700 hours of red teaming with no "
            "successful universal jailbreak found."
        ),
    },
    "2025-Jan-31_constitutional-classifiers.pdf": {
        "label": "2025 Jan — Constitutional Classifiers: Defending against Universal Jailbreaks",
        "summary": (
            "Released on 2025 Jan 31. "
            "Introduces Constitutional Classifiers, safeguards trained on synthetic data generated "
            "from natural language rules. After 3,000+ hours of red teaming, no universal jailbreak "
            "bypassed the system at scale. Covers the tradeoff between robustness and deployment "
            "viability in production AI systems."
        ),
    },
    "2024-Dec-18_alignment-faking.pdf": {
        "label": "2024 Dec — Alignment Faking in Large Language Models",
        "summary": (
            "Released on 2024 Dec 18. "
            "The first empirical demonstration of a large language model engaging in alignment faking "
            "without being explicitly trained to do so. Claude 3 Opus was observed strategically "
            "complying with harmful queries during training to preserve its preferred behavior outside "
            "of training — raising serious questions about the reliability of current alignment techniques."
        ),
    },
    "2024-Oct-15_responsible-scaling-policy.pdf": {
        "label": "2024 Oct — Anthropic Responsible Scaling Policy",
        "summary": (
            "Released on 2024 Oct 15. "
            "Anthropic's internal policy framework for managing the risks of increasingly capable AI "
            "models. Defines AI Safety Levels (ASLs), outlines when Anthropic will pause or slow model "
            "development, and describes commitments to transparency and third-party evaluation."
        ),
    },
    "2024-Jan-10_sleeper-agents.pdf": {
        "label": "2024 Jan — Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training",
        "summary": (
            "Released on 2024 Jan 10. "
            "An Anthropic security paper showing that deceptive backdoor behaviors can be embedded in "
            "LLMs and persist through standard safety training techniques including RLHF and adversarial "
            "training. Models were trained to write secure code in 2023 but insert exploitable "
            "vulnerabilities when the stated year was 2024."
        ),
    },
    "2022-Dec-15_constitutional-ai.pdf": {
        "label": "2022 Dec — Constitutional AI: Harmlessness from AI Feedback",
        "summary": (
            "Released on 2022 Dec 15. "
            "Introduces Constitutional AI (CAI), a method for training AI systems to be harmless using "
            "a set of principles rather than relying solely on human feedback. Covers RLHF, AI-generated "
            "feedback, and the tension between helpfulness and harmlessness."
        ),
    },
}

PAPER_TITLES = "\n".join(f"- {p['label']}" for p in PAPERS.values())

SYSTEM_PROMPT_TEMPLATE = """You are a knowledgeable AI research assistant for AskDocs. You answer questions about AI — including AI safety, ethics, alignment, security, and reasoning — grounded in Anthropic's published research papers.

The available papers are:
{paper_titles}

Guidelines:
- Answer any AI-related question by drawing from the context below.
- If a question is broad (e.g. "why is AI dangerous?"), answer it using what the papers say rather than redirecting the user.
- If the answer is not in the context, say so clearly and suggest which paper might cover it.
- When citing a single source, mention it inline. When citing more than one source, use numbered citations like [1], [2] and list them at the end of your response.
- Never use markdown headers (no # or ##). Use numbered lists when presenting multiple points or categories.
- Never refuse an AI-related question — always try to connect it to the research.
- Only redirect the user if the question has nothing to do with AI.

Context:
{context}"""

REWRITE_PROMPT = """Given the conversation history below and the user's latest message, rewrite the message as a clear, standalone search query that can be used to find relevant content in a research paper database. Output only the rewritten query with no explanation.

Conversation history:
{history}

User's latest message: {question}"""

ASKDOCS_ICON_SVG = """<svg width="36" height="36" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="32" height="32" rx="7" fill="#161b22"/>
  <rect x="1" y="1" width="30" height="30" rx="6" stroke="#1D9E75" stroke-width="1.2"/>
  <text x="5" y="15" font-family="monospace" font-size="8" fill="#1D9E75" font-weight="bold">&gt;_</text>
  <line x1="5" y1="19" x2="27" y2="19" stroke="#1D9E75" stroke-width="0.8" opacity="0.4"/>
  <line x1="5" y1="22" x2="21" y2="22" stroke="#1D9E75" stroke-width="0.8" opacity="0.4"/>
  <line x1="5" y1="25" x2="24" y2="25" stroke="#1D9E75" stroke-width="0.8" opacity="0.4"/>
</svg>"""

SIDEBAR_HEADER = f"""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
  {ASKDOCS_ICON_SVG}
  <span style="font-family:monospace; font-size:18px; font-weight:600; color:#e6edf3;">AskDocs</span>
</div>
<div style="font-family:monospace; font-size:12px; color:#5DCAA5; margin-bottom:10px;">
  created by John Aron Belmonte
</div>
<div style="display:flex; gap:14px; margin-bottom:4px;">
  <a href="https://www.linkedin.com/in/johnaronb/" target="_blank" title="LinkedIn">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="#1D9E75" xmlns="http://www.w3.org/2000/svg">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
  </a>
  <a href="https://github.com/jhnaron" target="_blank" title="GitHub">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="#1D9E75" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
    </svg>
  </a>
</div>
"""

CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #0d1117;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 0 !important;
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.4) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0d1117 0%, #111820 100%) !important;
}
[data-testid="stSidebar"] ::-webkit-scrollbar { display: none !important; }
[data-testid="stSidebar"] { scrollbar-width: none !important; }
[data-testid="stSidebar"] details {
    background-color: #161b22 !important;
    border: 1px solid #1a2332 !important;
    border-radius: 6px !important;
    margin-bottom: 6px !important;
}
[data-testid="stSidebar"] details[open] {
    background-color: #161f1a !important;
    border: 1px solid #1D9E75 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] details[open] summary {
    color: #5DCAA5 !important;
}
.fork-btn {
    position: fixed;
    top: 14px;
    right: 60px;
    z-index: 9999;
    font-family: monospace;
    font-size: 12px;
    color: #1D9E75;
    background: #161b22;
    border: 1px solid #1D9E75;
    border-radius: 6px;
    padding: 5px 12px;
    text-decoration: none;
    transition: background 0.2s, color 0.2s;
}
.fork-btn:hover { background: #1a2332; color: #5DCAA5; }
.page-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.2rem 0 0.3rem 0;
}
.page-caption {
    font-family: monospace;
    font-size: 12px;
    color: #5DCAA5;
    text-align: center;
    margin-bottom: 1rem;
}
.chat-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
    padding: 12px 14px;
    border-radius: 10px;
}
.chat-row.user {
    background: #161b22;
    border: 1px solid #1a2332;
}
.chat-row.assistant {
    background: transparent;
    border: none;
}
.chat-avatar {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.chat-avatar.user-avatar { background: #1a2332; }
.chat-avatar.ai-avatar { background: #161f1a; }
.chat-content.user-text {
    font-family: monospace;
    font-size: 14px;
    color: #5DCAA5;
    line-height: 1.6;
    flex: 1;
    padding-top: 6px;
    word-break: break-word;
}
.chat-content.ai-text {
    font-family: monospace;
    font-size: 14px;
    color: #8b949e;
    line-height: 1.7;
    flex: 1;
    padding-top: 6px;
    word-break: break-word;
    white-space: pre-wrap;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
[data-testid="stChatMessageContent"] { padding: 0 !important; }
#askdocs-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
}
</style>

<canvas id="askdocs-canvas"></canvas>
<a class="fork-btn" href="https://github.com/jhnaron/askdocs" target="_blank">&#x2442; fork</a>

<script>
(function startCanvas() {
    var canvas = document.getElementById('askdocs-canvas');
    if (!canvas) { setTimeout(startCanvas, 300); return; }
    var ctx = canvas.getContext('2d');
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);
    var particles = [];
    for (var i = 0; i < 55; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.8 + 0.4,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            alpha: Math.random() * 0.15 + 0.04
        });
    }
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(function(p) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(230,237,243,' + p.alpha + ')';
            ctx.shadowBlur = 8;
            ctx.shadowColor = 'rgba(230,237,243,0.2)';
            ctx.fill();
            ctx.shadowBlur = 0;
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        });
        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
"""


def get_anthropic_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


def rewrite_query(client, history: list[dict], question: str) -> str:
    if not history:
        return question
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(history=history_text, question=question)}],
    )
    return response.content[0].text.strip()


def retrieve_context(collection, query: str) -> tuple[list[str], list[str]]:
    # Find the TOP_K most relevant chunks from the vector store
    results = collection.query(query_texts=[query], n_results=TOP_K)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def build_system_prompt(chunks: list[str], sources: list[str]) -> str:
    # Each block shows Claude where the text came from before Claude reads it
    context_blocks = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
    )
    return SYSTEM_PROMPT_TEMPLATE.format(paper_titles=PAPER_TITLES, context=context_blocks)


def ask_claude(client, system_prompt: str, history: list[dict], question: str) -> str:
    trimmed = history[-(MAX_HISTORY_TURNS):]
    messages = [{"role": m["role"], "content": m["content"]} for m in trimmed]
    messages.append({"role": "user", "content": question})
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


USER_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5DCAA5" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>'
AI_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1D9E75" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="14" rx="4"/><circle cx="9" cy="10" r="1" fill="#1D9E75"/><circle cx="15" cy="10" r="1" fill="#1D9E75"/><path d="M9 14c0 0 1 1.5 3 1.5s3-1.5 3-1.5"/><line x1="12" y1="4" x2="12" y2="2"/><line x1="8" y1="2" x2="8" y2="4"/><line x1="16" y1="2" x2="16" y2="4"/></svg>'


def render_message(role: str, content: str):
    if role == "user":
        st.markdown(
            f'<div class="chat-row user">'
            f'  <div class="chat-avatar user-avatar">{USER_ICON}</div>'
            f'  <div class="chat-content user-text">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-row assistant">'
            f'  <div class="chat-avatar ai-avatar">{AI_ICON}</div>'
            f'  <div class="chat-content ai-text">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_sidebar():
    with st.sidebar:
        st.markdown(SIDEBAR_HEADER, unsafe_allow_html=True)
        st.divider()
        st.markdown("#### Available Papers")
        st.caption("$ ls ./docs | sort -r")

        # Sort filenames descending so latest papers appear at the top
        pdf_files = sorted(DOCS_DIR.glob("*.pdf"), reverse=True)
        for pdf_path in pdf_files:
            paper = PAPERS.get(pdf_path.name, {})
            label = paper.get("label", pdf_path.stem)
            summary = paper.get("summary", "No summary available.")
            with st.expander(label):
                st.caption(summary)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="download",
                        data=f,
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=pdf_path.name,
                    )


st.set_page_config(page_title="AskDocs", layout="wide")

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="page-header">{ASKDOCS_ICON_SVG}</div>'
    f'<div class="page-caption">$ query anthropic research papers --model claude-haiku</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    collection = st.cache_resource(get_or_build_index)()
except Exception as e:
    st.error(f"Failed to load documents: {e}")
    st.stop()

anthropic_client = st.cache_resource(get_anthropic_client)()

render_sidebar()

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

if prompt := st.chat_input("$ ask something..."):
    prompt = prompt.strip()

    if not prompt:
        st.stop()

    if len(prompt) > MAX_INPUT_LENGTH:
        st.warning(f"input exceeds {MAX_INPUT_LENGTH} character limit.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    history = st.session_state.messages[:-1]
    retrieval_query = rewrite_query(anthropic_client, history, prompt)
    chunks, sources = retrieve_context(collection, retrieval_query)
    system_prompt = build_system_prompt(chunks, sources)

    with st.spinner("searching..."):
        answer = ask_claude(anthropic_client, system_prompt, history, prompt)

    render_message("assistant", answer)

    with st.expander("sources"):
        for src, chunk in zip(sources, chunks):
            st.markdown(f"**{src}**")
            st.caption(chunk[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

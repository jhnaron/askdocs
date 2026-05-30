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


def get_anthropic_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


def rewrite_query(client, history: list[dict], question: str) -> str:
    # If there's no history, the question is already standalone
    if not history:
        return question

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": REWRITE_PROMPT.format(history=history_text, question=question),
        }],
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
    return SYSTEM_PROMPT_TEMPLATE.format(
        paper_titles=PAPER_TITLES,
        context=context_blocks,
    )


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


def render_sidebar():
    with st.sidebar:
        st.header("Available Papers")
        st.caption("The app answers questions from these documents.")

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
                        label="Download",
                        data=f,
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=pdf_path.name,
                    )


st.set_page_config(page_title="AskDocs", layout="wide")
st.title("AskDocs")
st.caption("Ask questions about Anthropic research papers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Both resources are cached so they're only initialized once per session
try:
    collection = st.cache_resource(get_or_build_index)()
except Exception as e:
    st.error(f"Failed to load documents: {e}")
    st.stop()

anthropic_client = st.cache_resource(get_anthropic_client)()

render_sidebar()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    prompt = prompt.strip()

    if not prompt:
        st.stop()

    if len(prompt) > MAX_INPUT_LENGTH:
        st.warning(f"Please keep your question under {MAX_INPUT_LENGTH} characters.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[:-1]

    # Rewrite the query using conversation context before searching
    retrieval_query = rewrite_query(anthropic_client, history, prompt)
    chunks, sources = retrieve_context(collection, retrieval_query)
    system_prompt = build_system_prompt(chunks, sources)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_claude(anthropic_client, system_prompt, history, prompt)
            st.markdown(answer)

            with st.expander("Sources used"):
                for src, chunk in zip(sources, chunks):
                    st.markdown(f"**{src}**")
                    st.caption(chunk[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
import anthropic
from index import get_or_build_index, DOCS_DIR

MAX_INPUT_LENGTH = 250
TOP_K = 4

PAPERS = {
    "constitutional_ai_2212.08073v1.pdf": {
        "label": "Constitutional AI: Harmlessness from AI Feedback",
        "summary": (
            "A 2022 Anthropic paper introducing Constitutional AI (CAI), a method for training "
            "AI systems to be harmless using a set of principles rather than relying solely on "
            "human feedback. It covers RLHF, AI-generated feedback, and the tension between "
            "helpfulness and harmlessness."
        ),
    },
    "responsible-scaling-policy.pdf": {
        "label": "Anthropic Responsible Scaling Policy (2024)",
        "summary": (
            "Anthropic's internal policy framework for managing the risks of increasingly "
            "capable AI models. It defines AI Safety Levels (ASLs), outlines when Anthropic "
            "will pause or slow model development, and describes commitments to transparency "
            "and third-party evaluation."
        ),
    },
}

PAPER_TITLES = "\n".join(f"- {p['label']}" for p in PAPERS.values())

SYSTEM_PROMPT_TEMPLATE = """You are a research assistant for AskDocs, an app that answers questions about Anthropic research papers.

The available papers are:
{paper_titles}

Answer the user's question using only the context below. If the question is not related to the papers, politely let them know what you can help with. If the answer is not in the context, say so clearly. Always mention which source you used.

Context:
{context}"""


def get_anthropic_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


def retrieve_context(collection, question: str) -> tuple[list[str], list[str]]:
    # Find the TOP_K most relevant chunks from the vector store
    results = collection.query(query_texts=[question], n_results=TOP_K)
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


def ask_claude(client, system_prompt: str, question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def render_sidebar():
    with st.sidebar:
        st.header("Available Papers")
        st.caption("The app answers questions from these documents.")

        pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
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

    chunks, sources = retrieve_context(collection, prompt)
    system_prompt = build_system_prompt(chunks, sources)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_claude(anthropic_client, system_prompt, prompt)
            st.markdown(answer)

            with st.expander("Sources used"):
                for src, chunk in zip(sources, chunks):
                    st.markdown(f"**{src}**")
                    st.caption(chunk[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

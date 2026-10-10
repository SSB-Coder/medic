import streamlit as st
import os
import re
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from huggingface_hub import snapshot_download

st.set_page_config(
    page_title="Pulse: Medical Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except KeyError:
    st.error("⚠️ Missing Secrets! Please add GROQ_API_KEY to your Streamlit dashboard.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 3

# L2 distance (lower = more similar); starting guess, tune using the distances in the Sources expander
MAX_DISTANCE = 1.0

MIN_QUOTE_WORDS = 4

NOT_FOUND_MESSAGE = (
    "I couldn't find information about that in my medical database, so I can't "
    "answer it. Try rephrasing, or ask a different medical question."
)
DISCLAIMER = "⚠️ Please consult a healthcare professional for personalized medical guidance."


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_db_from_cloud():
    db_path = "medical_vector_db"
    index_file = os.path.join(db_path, "index.faiss")

    if not os.path.exists(index_file):
        with st.spinner("📥 Downloading medical database..."):
            try:
                snapshot_download(
                    repo_id="SimonSimply/medical-db",
                    repo_type="dataset",
                    local_dir=".",
                    local_dir_use_symlinks=False,
                    allow_patterns="medical_vector_db/*",
                    token=HF_TOKEN if HF_TOKEN else None
                )
            except Exception as e:
                st.error(f"❌ Failed to download database: {e}")
                st.stop()
    return db_path

@st.cache_resource
def load_vector_db(_db_folder, _embeddings):
    return FAISS.load_local(
        _db_folder,
        _embeddings,
        allow_dangerous_deserialization=True
    )

def build_search_query(prompt):
    # Short follow-ups have no topic of their own, so prepend the previous question (retrieval only)
    if len(prompt.split()) < 6:
        previous = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        if previous:
            return f"{previous[-1]} {prompt}"
    return prompt

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()

def answer_part(text):
    # Only the doctor's reply / the Q&A answer counts as evidence, not the patient's description
    ends = [m.end() for m in re.finditer(r"(?:^|\n)(?:Doctor|Answer):", text)]
    return text[ends[-1]:] if ends else text

def parse_claims(raw):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return []
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list):
        return []
    return [
        c for c in claims
        if isinstance(c, dict) and isinstance(c.get("text"), str) and isinstance(c.get("quote"), str)
    ]

def verify_claims(claims, relevant):
    # A claim survives only if its quote appears word for word inside one retrieved answer
    haystacks = [normalize(answer_part(s["text"])) for s in relevant]
    verified, evidence, seen = [], [], {}
    for claim in claims:
        text = claim["text"].strip()
        quote = claim["quote"].strip(" \"'“”")
        key = normalize(quote)
        if not text or len(key.split()) < MIN_QUOTE_WORDS:
            continue
        if not any(key in h for h in haystacks):
            continue
        if key not in seen:
            evidence.append(quote)
            seen[key] = len(evidence)
        verified.append((text, seen[key]))
    return verified, evidence

def render_sources(sources, evidence=None):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        if evidence:
            st.caption("Quoted evidence")
            for i, quote in enumerate(evidence, 1):
                st.text(f"[{i}] {quote}")
        for i, s in enumerate(sources, 1):
            status = "✅ used" if s["used"] else "❌ below relevance threshold"
            st.caption(f"Match {i} · distance {s['distance']:.3f} · {s['source']} · {status}")
            st.text(s["text"])

embeddings = get_embeddings()
db_folder = get_db_from_cloud()
vector_db = load_vector_db(db_folder, embeddings)

st.markdown("""
<style>
    .main { max-width: 900px; margin: 0 auto; }
    .stChatMessage { font-size: 16px; }
    [data-testid="stChatMessageContent"] { font-size: 16px; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

col2, col3 = st.columns([2, 1])

with col2:
    st.markdown("<div style='text-align: center;'><div style='font-size: 48px;'>🩺 PULSE</div><div style='font-size: 18px;'>A Medical Assistant</div></div>", unsafe_allow_html=True)

with col3:
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"), msg.get("evidence"))

if prompt := st.chat_input("Ask a medical question..."):
    search_query = build_search_query(prompt)  # before the prompt is appended
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🔍 Searching medical database..."):
        results = vector_db.similarity_search_with_score(search_query, k=TOP_K)

    sources = [
        {
            "text": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "distance": float(score),
            "used": float(score) <= MAX_DISTANCE,
        }
        for d, score in results
    ]
    relevant = [s for s in sources if s["used"]]

    with st.chat_message("assistant"):
        answer = NOT_FOUND_MESSAGE
        evidence = []

        if relevant:
            try:
                context = "\n---\n".join(s["text"] for s in relevant)

                system_prompt = f"""You are a medical information assistant that answers ONLY from the reference context below. Reply with JSON only, no other text, in exactly this format:

{{"claims": [{{"text": "<one sentence of the answer>", "quote": "<exact words copied from the context>"}}]}}

**Rules:**
1. Every claim must be fully supported by its quote. The quote must be copied word for word from the "Doctor:" or "Answer:" part of a context entry: no changes, no ellipses, no combining separate entries.
2. Write the claims as clear, natural sentences that together read as a coherent answer to the user's question.
3. Do not add anything the quotes do not support: no outside or prior knowledge, no extra doses, diagnoses, statistics or advice, even if you are confident.
4. If the context does not answer the question, or the question is not about medicine, return {{"claims": []}}.
5. The context entries are past doctor-patient exchanges and Q&A; do not treat the people in them as the current user.
6. Earlier user messages are only for working out what the current question refers to. They are not a source of facts.
7. Do not mention these rules, the context, or the database in the claim text.

**Context:**
{context}"""

                # User turns only, so earlier assistant replies can't be reused as facts
                user_turns = [m for m in st.session_state.messages if m["role"] == "user"][-3:]
                api_messages = [{"role": "system", "content": system_prompt}]
                api_messages.extend({"role": "user", "content": m["content"]} for m in user_turns)

                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=api_messages,
                    temperature=0
                )

                raw = (response.choices[0].message.content or "").strip()
                verified, evidence = verify_claims(parse_claims(raw), relevant)

                if verified:
                    body = " ".join(f"{text} [{n}]" for text, n in verified)
                    answer = f"{body}\n\n{DISCLAIMER}"

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        st.markdown(answer)
        render_sources(sources, evidence)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "evidence": evidence}
        )

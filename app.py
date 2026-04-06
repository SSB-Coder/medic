import streamlit as st
import os
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

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🔍 Searching medical database..."):
        docs = vector_db.similarity_search(prompt, k=3)
        context = "\n---\n".join([d.page_content for d in docs])

    with st.chat_message("assistant"):
        try:
            system_prompt = f"""You are a professional medical assistant who provides accurate, evidence-based medical information.

**Instructions:**
- Answer based on the provided context
- Give detailed, correct medical advice
- If the question is unrelated to medicine, politely explain you can only answer medical questions
- If the answer is outside your knowledge, say "I don't know" rather than making up information
- Use context and conversation history to stay relevant
- Do NOT make up answers or fabricate context

**IMPORTANT**: After any medical advice and only medical advice, include a disclaimer:
"⚠️ Please consult a healthcare professional for personalized medical guidance."

**Context:**
{context}"""

            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(st.session_state.messages[-6:])

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=api_messages,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

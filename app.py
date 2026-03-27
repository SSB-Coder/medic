import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from huggingface_hub import snapshot_download

# --- 1. CONFIG & SECRETS ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    # Provide a default empty string if HF_TOKEN isn't set
    HF_TOKEN = st.secrets.get("HF_TOKEN", "") 
except KeyError:
    st.error("Missing Secrets! Please add GROQ_API_KEY to your Streamlit dashboard.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺", layout="wide")

# --- 2. EMBEDDINGS SETUP ---
@st.cache_resource
def get_embeddings():
    # This downloads ~90MB. If it hangs, the Streamlit server is likely low on RAM.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 3. CLOUD DOWNLOADER ---
@st.cache_resource
def get_db_from_cloud():
    db_path = "medical_vector_db"
    # Look for the specific index file to verify the download
    index_file = os.path.join(db_path, "index.faiss")
    
    if not os.path.exists(index_file):
        with st.spinner("📥 Downloading medical database from Hugging Face..."):
            try:
                snapshot_download(
                    repo_id="SimonSimply/medical-db", # <--- DOUBLE CHECK THIS NAME
                    repo_type="dataset",
                    local_dir=".",
                    local_dir_use_symlinks=False,
                    allow_patterns="medical_vector_db/*",
                    token=HF_TOKEN if HF_TOKEN else None
                )
            except Exception as e:
                st.error(f"Failed to download database: {e}")
                st.stop()
    return db_path

# --- 4. DATA INITIALIZATION ---
# Load embeddings first, then download/load the DB
embeddings = get_embeddings()
db_folder = get_db_from_cloud()

@st.cache_resource
def load_memory(_db_folder, _embeddings):
    # Fixed: Removed 'use_avx2' which caused the previous TypeError
    return FAISS.load_local(
        _db_folder, 
        _embeddings, 
        allow_dangerous_deserialization=True
    )

vector_db = load_memory(db_folder, embeddings)

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Chat Settings")
    st.info("Managing your session history.")
    
    if st.button("🗑️ Clear Chat & Start New", use_container_width=True):
        st.session_state.messages = []
        st.rerun() 

    st.divider()
    st.markdown("### System Status")
    st.success("Database Connected")
    st.success("Groq LPU Active")

# --- 6. USER INTERFACE ---
st.title("🩺 Professional Medical Assistant")
st.markdown("Querying 256,000+ records via high-speed RAG.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat interaction
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # SEARCH
    with st.spinner("Searching medical database..."):
        # Retrieve top 3 relevant chunks
        docs = vector_db.similarity_search(prompt, k=3)
        context = "\n---\n".join([d.page_content for d in docs])

    # GENERATE
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are a professional medical assistant. Answer based on: {context}"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                answer = completion.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Groq API Error: {e}")

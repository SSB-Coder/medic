import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from huggingface_hub import snapshot_download

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HF_TOKEN = st.secrets.get("HF_TOKEN", "") 
except KeyError:
    st.error("Missing Secrets! Please add GROQ_API_KEY to your Streamlit dashboard.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Medic Chatbot", page_icon="🩺", layout="wide")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_db_from_cloud():
    db_path = "medical_vector_db"
    index_file = os.path.join(db_path, "index.faiss")
    
    if not os.path.exists(index_file):
        with st.spinner("📥 Downloading medical database from Hugging Face..."):
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
                st.error(f"Failed to download database: {e}")
                st.stop()
    return db_path

embeddings = get_embeddings()
db_folder = get_db_from_cloud()

@st.cache_resource
def load_memory(_db_folder, _embeddings):
    return FAISS.load_local(
        _db_folder, 
        _embeddings, 
        allow_dangerous_deserialization=True
    )

vector_db = load_memory(db_folder, embeddings)

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

st.title("🩺 Professional Medical Assistant")
st.markdown("Querying 256,000+ records via high-speed RAG.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching medical database..."):
        
        docs = vector_db.similarity_search(prompt, k=3)
        context = "\n---\n".join([d.page_content for d in docs])

   with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                api_messages = [
                    {
                        "role": "system", 
                        "content": f"You are a professional medical assistant who gives only medical answers in detail. Answer based on: {context} and give correct medical advice. If the question is not related to medicine, say you can only answer medical questions. If it's out of your knowledge, say you don't know. Do not make up answers. Use the context and the conversation history to stay relevant. After any and all medical advice and only medical advice, include a note that the user should consult a healthcare professional for personalized guidance instead of relying solely on the information provided here."
                    }
                ]

                for msg in st.session_state.messages[-6:]:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})

                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=api_messages,
                    temperature=0.1
                )
                
                answer = completion.choices[0].message.content
                st.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Groq API Error: {e}")

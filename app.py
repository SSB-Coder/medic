import streamlit as st
import streamlit.components.v1 as components
import os
import re
import html as html_lib
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

LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 3

# L2 distance (lower = closer). Starting guess: tune it using the distances shown in the matches panel.
WEAK_MATCH_DISTANCE = 1.0
MATCH_CHARS = 400

KIND_LABELS = {"medical": "Doctor–patient chat", "convo": "Q&A entry"}

SUGGESTIONS = [
    "What are the common symptoms of diabetes?",
    "How can I manage high blood pressure?",
    "What causes frequent headaches?",
]

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:wght@500;600&display=swap');

:root {
    --gauze: #F6F8FA;
    --ink: #10243A;
    --slate: #5B6B7B;
    --line: #DCE3EA;
    --pulse: #C8283F;
    --tint: #E8EEF5;
    --serif: 'Newsreader', Georgia, serif;
    --sans: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --nc-w: 6.4rem;
    --nc-gap: 0.75rem;
    color-scheme: light;
}

#MainMenu, footer,
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

.stApp { background: var(--gauze); font-family: var(--sans); }
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--ink); }
.stApp p, .stApp li, .stApp td, .stApp th, .stApp button, .stApp textarea { font-family: var(--sans); }
.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 7rem; }
[data-testid="stHorizontalBlock"] { align-items: center; gap: 0.75rem; }

.brand { display: flex; flex-direction: column; gap: 0.15rem; }
.wordmark { font-family: var(--serif); font-weight: 600; font-size: 2.1rem; line-height: 1.05; letter-spacing: -0.01em; color: var(--ink); }
.tagline { font-family: var(--sans); font-size: 0.92rem; color: var(--slate); }

.ecg { margin: 0.4rem 0 1.4rem; }
.ecg svg { display: block; width: 100%; height: auto; }
.ecg path {
    fill: none; stroke: var(--pulse); stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round;
    stroke-dasharray: 1200; stroke-dashoffset: 0;
    animation: trace 1.8s cubic-bezier(0.4, 0, 0.2, 1) 0.15s both;
}
@keyframes trace { from { stroke-dashoffset: 1200; } to { stroke-dashoffset: 0; } }
@media (prefers-reduced-motion: reduce) { .ecg path { animation: none; } }

.welcome { padding: 1.2rem 0 1rem; }
.welcome-title { font-family: var(--serif); font-weight: 600; font-size: 1.9rem; line-height: 1.15; letter-spacing: -0.01em; color: var(--ink); margin-bottom: 0.5rem; }
.stApp p.welcome-copy { color: var(--slate); max-width: 52ch; margin: 0 0 1.2rem; line-height: 1.6; }

.stButton > button {
    width: 100%; min-height: 3rem; padding: 0.55rem 0.85rem;
    border-radius: 10px; border: 1px solid #C9D3DD; background: #fff;
    color: var(--ink); font-weight: 500; line-height: 1.35;
    transition: border-color 0.15s, color 0.15s;
}
.stButton > button { height: auto; }
.stButton > button, .stButton > button * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
.stButton > button p { color: inherit; }
.stButton > button:hover { border-color: var(--pulse); color: var(--pulse); background: #fff; }
.stButton > button:focus-visible { outline: 2px solid var(--pulse); outline-offset: 2px; }

[data-testid="stChatMessage"] {
    background: #fff; border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.15rem; margin-bottom: 0.8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--tint); border-color: transparent;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] { display: none; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { overflow-x: auto; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 1rem; line-height: 1.65; max-width: 70ch; }
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 {
    font-family: var(--serif); font-weight: 600; line-height: 1.25; margin: 1.1rem 0 0.4rem; padding: 0;
}
[data-testid="stChatMessage"] h1 { font-size: 1.4rem; }
[data-testid="stChatMessage"] h2 { font-size: 1.25rem; }
[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 { font-size: 1.1rem; }
[data-testid="stChatMessage"] table { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 0.6rem 0; }
[data-testid="stChatMessage"] th { background: var(--gauze); text-align: left; font-weight: 600; }
[data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {
    border: 0 !important; border-bottom: 1px solid var(--line) !important;
    padding: 0.6rem 0.75rem; vertical-align: top;
}
[data-testid="stChatMessage"] hr { border: 0; border-top: 1px solid var(--line); margin: 1rem 0; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color: var(--ink); }
[data-testid="stChatMessage"] code, [data-testid="stChatMessage"] pre { background: #EEF2F6; }

[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--gauze); }
[data-testid="stChatInput"] { width: calc(100% - var(--nc-w) - var(--nc-gap)); background: #fff !important; border: 1px solid #C9D3DD !important; border-radius: 999px !important; overflow: hidden; }
[data-testid="stChatInput"] div { background: transparent !important; border-color: transparent !important; box-shadow: none !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--ink) !important; box-shadow: 0 0 0 3px rgba(200, 40, 63, 0.18) !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--ink) !important; font-family: var(--sans); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--slate); }
[data-testid="stChatInput"] button { background: var(--ink); color: #fff; border-radius: 999px; }

.st-key-new_chat { position: fixed; right: 1rem; bottom: 1.15rem; z-index: 1000; width: var(--nc-w) !important; }
.st-key-new_chat button { min-height: 3rem; height: 3rem; padding: 0 0.5rem; border-radius: 999px; }
.st-key-new_chat button, .st-key-new_chat button * { white-space: nowrap !important; }

[data-testid="stElementContainer"]:has(iframe[height="0"]) { position: absolute; width: 0; height: 0; margin: 0; overflow: hidden; }

[data-testid="stChatMessage"] .weak-note {
    margin-top: 0.8rem; padding: 0.55rem 0.8rem; border: 1px solid #EBD9A4; border-radius: 10px;
    background: #FFF7E0; color: #7A5200; font-size: 0.88rem; line-height: 1.5;
}
[data-testid="stExpander"] { margin-top: 0.8rem; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
[data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: transparent; border: 0; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--slate); font-size: 0.9rem; }
[data-testid="stChatMessage"] .match { margin: 0 0 1rem; }
[data-testid="stChatMessage"] .match:last-child { margin-bottom: 0.2rem; }
[data-testid="stChatMessage"] .match-meta { display: flex; justify-content: space-between; gap: 1rem; margin-bottom: 0.25rem; font-size: 0.8rem; color: var(--slate); }
[data-testid="stChatMessage"] .match-text { font-size: 0.92rem; line-height: 1.55; color: var(--ink); }

@media (max-width: 640px) {
    :root { --nc-w: 5.4rem; }
    .wordmark { font-size: 1.75rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .st-key-new_chat button { font-size: 0.85rem; }
}
</style>
"""

HEADER_HTML = (
    '<div class="brand"><span class="wordmark">Pulse</span>'
    '<span class="tagline">Medical information assistant</span></div>'
)

ECG_HTML = (
    '<div class="ecg"><svg viewBox="0 0 740 48" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M0 26H214Q226 12 238 26H262L269 32L281 4L293 44L301 26H336Q360 6 384 26H740"/>'
    '</svg></div>'
)

WELCOME_HTML = (
    '<div class="welcome"><div class="welcome-title">Ask a health question</div>'
    '<p class="welcome-copy">Pulse searches a database of doctor–patient conversations to help answer. '
    'General information only, not a diagnosis.</p></div>'
)

CONTROLS_HTML = """
<script>
(function () {
  var win = window.parent;
  var doc = win.document;

  if (win.__pulseCleanup) {
    try { win.__pulseCleanup(); } catch (err) {}
  }

  function onKey(e) {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
    var t = e.target;
    var tag = t && t.tagName ? t.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || (t && t.isContentEditable)) return;
    var box = doc.querySelector('[data-testid="stChatInput"] textarea');
    if (!box) return;
    e.preventDefault();
    box.focus();
    var main = doc.querySelector('[data-testid="stMain"]') ||
               doc.querySelector('section.main') ||
               doc.querySelector('[data-testid="stAppViewContainer"]');
    if (main && main.scrollTo) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
  }

  function place() {
    var input = doc.querySelector('[data-testid="stChatInput"]');
    if (!input) return;
    var btn = null;
    var all = doc.querySelectorAll('[data-testid="stButton"] button');
    for (var i = 0; i < all.length; i++) {
      if (all[i].textContent.trim() === 'New chat') { btn = all[i]; break; }
    }
    if (!btn) return;
    var r = input.getBoundingClientRect();
    if (!r.width || !r.height) return;
    var holder = btn.closest('[data-testid="stElementContainer"]') ||
                 btn.closest('.element-container') ||
                 btn.closest('[data-testid="stButton"]');
    var w = win.getComputedStyle(doc.documentElement).getPropertyValue('--nc-w').trim() || '6.4rem';
    holder.style.position = 'fixed';
    holder.style.zIndex = '1000';
    holder.style.width = w;
    holder.style.left = (r.right + 12) + 'px';
    holder.style.top = r.top + 'px';
    holder.style.right = 'auto';
    holder.style.bottom = 'auto';
    btn.style.height = r.height + 'px';
    btn.style.minHeight = r.height + 'px';
    btn.style.whiteSpace = 'nowrap';
    btn.style.borderRadius = '999px';
  }

  var timer = setInterval(place, 300);
  doc.addEventListener('keydown', onKey, true);
  win.addEventListener('resize', place);
  win.__pulseCleanup = function () {
    doc.removeEventListener('keydown', onKey, true);
    win.removeEventListener('resize', place);
    clearInterval(timer);
  };
  place();
})();
</script>
"""

def format_answer(text):
    # Models use <br> for line breaks inside table cells. Keep those, neutralize every other tag.
    text = re.sub(r"<br\s*/?>", "\x00", str(text), flags=re.IGNORECASE)
    text = re.sub(r"<(?=[A-Za-z/!?])", "&lt;", text)
    return text.replace("\x00", "<br>")

def show_message(role, text):
    if role == "assistant":
        st.markdown(format_answer(text), unsafe_allow_html=True)
    else:
        st.markdown(text)

def make_excerpt(text, source):
    marker = {"medical": "Doctor:", "convo": "Answer:"}.get(source)
    body = str(text).rsplit(marker, 1)[-1] if marker else str(text)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > MATCH_CHARS:
        body = body[:MATCH_CHARS].rsplit(" ", 1)[0].rstrip(",;:.") + "…"
    return body

def build_matches(results):
    return [
        {
            "kind": KIND_LABELS.get(d.metadata.get("source"), "Database entry"),
            "excerpt": make_excerpt(d.page_content, d.metadata.get("source")),
            "distance": float(score),
        }
        for d, score in results
    ]

def is_weak(matches):
    return not matches or min(m["distance"] for m in matches) > WEAK_MATCH_DISTANCE

def match_html(m):
    text = html_lib.escape(m["excerpt"]).replace("$", "&#36;")
    return (
        '<div class="match"><div class="match-meta">'
        f'<span>{html_lib.escape(m["kind"])}</span><span>distance {m["distance"]:.3f}</span></div>'
        f'<div class="match-text">{text}</div></div>'
    )

WEAK_NOTE_HTML = (
    '<div class="weak-note">No close match found in the database, '
    'so this answer may rely on general knowledge.</div>'
)

def render_matches(matches, weak):
    if weak:
        st.markdown(WEAK_NOTE_HTML, unsafe_allow_html=True)
    if not matches:
        return
    with st.expander(f"Closest matches in the database ({len(matches)})"):
        st.markdown("".join(match_html(m) for m in matches), unsafe_allow_html=True)

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

st.markdown(STYLE, unsafe_allow_html=True)

try:
    components.html(CONTROLS_HTML, height=0)
except Exception:
    pass

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(HEADER_HTML, unsafe_allow_html=True)

if st.button("New chat", key="new_chat"):
    st.session_state.messages = []
    st.rerun()

st.markdown(ECG_HTML, unsafe_allow_html=True)

welcome = st.empty()
if not st.session_state.messages:
    with welcome.container():
        st.markdown(WELCOME_HTML, unsafe_allow_html=True)
        chip_cols = st.columns(len(SUGGESTIONS))
        for i, (chip_col, question) in enumerate(zip(chip_cols, SUGGESTIONS)):
            if chip_col.button(question, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show_message(msg["role"], msg["content"])
        if msg["role"] == "assistant":
            render_matches(msg.get("matches"), msg.get("weak", False))

typed_prompt = st.chat_input("Ask a medical question...")
prompt = typed_prompt or st.session_state.pop("pending_prompt", None)

if prompt:
    welcome.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching the medical database..."):
        results = vector_db.similarity_search_with_score(prompt, k=TOP_K)
        context = "\n---\n".join([d.page_content for d, _ in results])
    matches = build_matches(results)

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
            api_messages.extend(
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]
            )

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=api_messages,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            weak = is_weak(matches)
            show_message("assistant", answer)
            render_matches(matches, weak)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "matches": matches, "weak": weak}
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

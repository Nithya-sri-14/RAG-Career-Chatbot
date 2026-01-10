import os
import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import docx2txt
from pptx import Presentation
import fitz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit.components.v1 as components

# ----------------------
# 1️⃣ GROQ API Key
# ----------------------
def get_groq_api_key():
    if "GROQ_API_KEY" in os.environ:
        return os.environ["GROQ_API_KEY"]
    return st.secrets["GROQ_API_KEY"]

GROQ_API_KEY = get_groq_api_key()

# ----------------------
# 2️⃣ Assets
# ----------------------
RESUME_MATCHER_IMAGE_URL = "https://recruitryte.com/wp-content/uploads/2023/04/AI-Resume-Matching-Tool-for-Job-Descriptions-recruitRyte.jpg"
CHATBOT_IMAGE_URL = "https://neo4j.com/wp-content/uploads/2023/04/knowledge-graph-based-chatbot-scaled.jpg"
ASSETS_PATH = "assets"
SINGLE_BANNER_PATH = os.path.join(ASSETS_PATH, "banner7.png")

def get_image_as_bytes(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

# ----------------------
# 3️⃣ Streamlit Page Config
# ----------------------
st.set_page_config(page_title="AI Career Assistant", layout="wide", page_icon="💼")
st.markdown("""
<style>
body { background-color: #0e1117; color: #ffffff; }
.stButton>button { background-color: #262730; color: white; border-radius: 10px; padding: 0.6em 1em; border: none; }
.stButton>button:hover { background-color: #333642; }
.stCard { border: 1px solid #333642; border-radius: 12px; padding: 10px; background-color: #1e1e2d; margin-top: 15px; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem; }
.main-centered-title { text-align: center; margin-top: 0 !important; margin-bottom: 1.5rem; font-size: 2.5em; color: #f0f0f5; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# 4️⃣ Load Models
# ----------------------
@st.cache_resource(show_spinner=False)
def load_resume_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"❌ Resume model load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_flan_llm():
    model_name = "google/flan-t5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1, max_new_tokens=200)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"❌ LLM load failed: {e}")
        return None

# ----------------------
# 5️⃣ Load Embeddings and FAISS (Auto-build if missing)
# ----------------------
FAISS_PATH = "faiss_index"
@st.cache_resource(show_spinner=False)
def load_embeddings_and_faiss():
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    faiss_path = os.path.join(BASE_PATH, "faiss_index")
    
    if os.path.exists(faiss_path):
        vect = FAISS.load_local(faiss_path, embeddings=emb_model)
        return emb_model, vect
    
    # Build FAISS index silently (no print or st.* calls)
    df = load_data()
    if df.empty:
        return emb_model, None
    
    vect = FAISS.from_texts(df['combined'].tolist(), embedding=emb_model)
    vect.save_local(faiss_path)
    
    return emb_model, vect



# ----------------------
# 6️⃣ Build RAG Chain
# ----------------------
@st.cache_resource(show_spinner=False)
def build_rag_chain():
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    emb_model, vectorstore = load_embeddings_and_faiss()
    
    if vectorstore is None:
        # Show a clean error inside the Chatbot page only
        st.warning("❌ FAISS vector store not loaded. The chatbot will not work until the index is built.")
        return None

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are an expert IT career assistant.
Answer clearly and professionally using ONLY the context.
If context does not contain the answer, say "I don't have that exact information."

Context:
{context}

Question:
{question}

Final Answer:
""")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ----------------------
# 7️⃣ Load dataset
# ----------------------
@st.cache_data
def load_data():
    dataset = load_dataset("DevilsLord/It_job_roles_skills_certifications")
    df = dataset['train'].to_pandas()
    df['combined'] = df['Job Description'].fillna('') + " " + df['Skills'].fillna('') + " " + df['Certifications'].fillna('')
    return df

# ----------------------
# 8️⃣ Text Extractor
# ----------------------
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'txt':
        return file.read().decode('utf-8', errors='ignore')
    elif ext == 'pdf':
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    elif ext == 'docx':
        return docx2txt.process(file)
    elif ext == 'pptx':
        text = ""
        presentation = Presentation(file)
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    return ""

# ----------------------
# 9️⃣ Initialize
# ----------------------
with st.spinner("Initializing AI models and data..."):
    df = load_data()
    resume_model = load_resume_model()
    rag_chain = build_rag_chain()

# ----------------------
# 10️⃣ Streamlit UI
# ----------------------
st.sidebar.title("🌙 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📄 Resume Matcher", "💬 Chatbot"], label_visibility="collapsed")

if page == "🏠 Home":
    st.markdown('<h1 class="main-centered-title">🧠 AI-Powered Career Assistant</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Welcome to your <b>AI-driven Resume–Job Matcher & Chatbot</b>!</p>", unsafe_allow_html=True)
    st.image(get_image_as_bytes(SINGLE_BANNER_PATH), use_container_width=True)

elif page == "📄 Resume Matcher":
    st.markdown('<h2 class="main-centered-title">📄 AI Resume Matcher</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF, DOCX, PPTX, TXT", type=["txt", "pdf", "docx", "pptx"])
    
    if uploaded_file and resume_model:
        resume_text = extract_text(uploaded_file).replace('\n',' ').strip()
        resume_emb = resume_model.encode(resume_text, convert_to_tensor=True)
        job_embs = resume_model.encode(df['combined'].tolist(), convert_to_tensor=True)
        sims = util.cos_sim(resume_emb, job_embs)[0].cpu().numpy()
        df['Similarity'] = sims
        top_matches = df.sort_values('Similarity', ascending=False).head(10).reset_index(drop=True)
        
        for idx, row in top_matches.iterrows():
            st.markdown(f"**{row['Job Title']}** — Match Score: {row['Similarity']:.2f}")

elif page == "💬 Chatbot":
    st.markdown('<h2 class="main-centered-title">💬 Custom IT Career Chatbot</h2>', unsafe_allow_html=True)
    if rag_chain is None:
        st.error("❌ Chatbot model (RAG chain) is not loaded.")
        st.stop()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Ask me about IT careers, skills, or certifications."}]
    
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            assistant_response = rag_chain.invoke(prompt)
            st.write(assistant_response)
            st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/69e7db22-83a5-4c14-a4f7-a75a8b14e024" />
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/978fc868-91f6-402f-8e42-7d38f5eb569b" />
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/7b12e425-f66e-43cd-bbfa-dee5a24ea77f" />

# 🤖 AI-Powered Career Assistant

An end-to-end **AI-driven Resume–Job Matching & IT Career Guidance Platform** that helps job seekers analyze their resumes, discover suitable IT job roles, and get instant career-related answers using **NLP, Semantic Search, and RAG-based Chatbots**.

---

## 🚀 Project Overview

The **AI-Powered Career Assistant** is designed to bridge the gap between **job seekers** and **industry requirements**. It intelligently parses resumes, semantically matches them against hundreds of IT job roles, and provides an interactive chatbot for career guidance.

This project demonstrates practical usage of **Machine Learning, Deep Learning, NLP, and Retrieval-Augmented Generation (RAG)** in a real-world career assistance system.

---

## ✨ Key Features

### 📄 AI Resume Parser & Matcher

* Upload resumes in **PDF, DOCX, PPTX, or TXT** formats
* Extracts key skills, experience, and keywords
* Matches resumes with **200+ IT job roles**
* Provides a **Match Score (0.0 – 1.0)** using semantic similarity

### 🧠 Semantic Job Matching

* Uses **Sentence-BERT** for deep semantic understanding
* Matches skills beyond keyword-based comparison
* Identifies best-fit job roles based on resume content

### 💬 Custom IT Career Chatbot (RAG)

* Ask questions about:

  * IT job roles
  * Required skills
  * Certifications
  * Career paths
* Powered by **FLAN-T5** with **FAISS Vector Search**
* Retrieves accurate answers from curated IT knowledge base

### 🌐 Interactive Web UI

* Built with **Streamlit**
* Clean dark-themed UI
* Easy navigation between:

  * Home
  * Resume Matcher
  * Chatbot

---

## 🏗️ System Architecture

```
User Resume
    ↓
Text Extraction (PDF/DOCX/TXT)
    ↓
Sentence-BERT Embeddings
    ↓
Semantic Similarity Matching
    ↓
Top Job Matches + Match Scores

User Question
    ↓
RAG Pipeline (FAISS + FLAN-T5)
    ↓
Context-aware Career Answer
```

---

## 🧪 Core Technologies Used

### 🔧 Backend & AI

* **Python**
* **Sentence-BERT (SBERT)** – Resume–Job semantic matching
* **FLAN-T5 Small** – Career chatbot responses
* **FAISS** – Vector database for fast retrieval
* **Transformers (Hugging Face)**

### 🎨 Frontend

* **Streamlit** – Interactive UI

### 📚 Data Sources

* IT Job Descriptions
* Required Skills & Certifications Database

---

## 📊 Match Score Explanation

* **0.0 – 0.4** → Low alignment
* **0.5 – 0.7** → Moderate alignment
* **0.8 – 1.0** → Strong job fit

Higher score = better alignment between resume and job role requirements.

---

## 🖥️ Screenshots

* AI Resume Parser Dashboard
* Resume Upload & Job Matching
* Custom IT Career Chatbot

*(Screenshots included in the repository)*

---

## 🛠️ How to Run the Project Locally

```bash
# Clone the repository
git clone https://github.com/your-username/ai-career-assistant.git

# Navigate to project directory
cd ai-career-assistant

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 🎯 Use Cases

* Students exploring IT career paths
* Freshers matching resumes to job roles
* Professionals identifying skill gaps
* Career guidance through AI chatbot

---

## 🌟 Future Enhancements

* Resume improvement suggestions
* Skill gap analysis
* Job recommendation filtering by location
* Multi-language resume support
* PDF career report generation

---

## 📌 Project Highlights

* Real-world NLP application
* Semantic search over keyword matching
* Practical implementation of RAG
* Strong portfolio-ready AI project

---

## 👩‍💻 Author

**Nithya Sri A**
AI & Data Enthusiast | Python Developer | NLP Explorer

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ *If you like this project, don’t forget to star the repo

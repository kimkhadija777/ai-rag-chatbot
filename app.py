import os
import torch

# ------------------------------------------------------------
# ⚡ CRITICAL OPTIMIZATIONS FOR STREAMLIT CLOUD
# ------------------------------------------------------------
# Limit PyTorch to 1 thread to prevent free-tier CPU lockups
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set Page Configuration
st.set_page_config(page_title="AI RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🔥 AI RAG Chatbot")
st.caption("FAISS + Sentence Transformers + Groq GPT-OSS 120B")

# Sidebar
with st.sidebar:
    st.header("📚 Knowledge Areas")
    st.markdown("""
    - 🌍 General Knowledge
    - 🌳 Environment
    - 🤖 Artificial Intelligence
    - 💻 Computer Science
    - 🔐 Cybersecurity
    - 🌐 Networking
    - 📚 Education
    - 💡 Technology
    - 🧠 Productivity
    - 🎵 BTS
    """)
    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------
# 1️⃣ LOAD EMBEDDINGS & FAISS INDEX (CACHED)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    documents = [
        # General
        "Food, water, shelter, and health are basic needs for human living.",
        "Education helps people develop knowledge, skills, critical thinking, and improve their opportunities in life.",
        "Clean air and clean water are essential for human health and survival.",
        "Communication is the process of sharing information, ideas, thoughts, or feelings between people.",
        "Critical thinking means carefully analyzing information, evaluating evidence, and making reasoned conclusions.",

        # Environment
        "A tree is a perennial plant with a woody stem or trunk. Trees generally have roots, a trunk, branches, and leaves.",
        "Trees absorb carbon dioxide and release oxygen through photosynthesis.",
        "Trees provide shade and habitats for many organisms.",
        "Forests provide habitats for animals and help protect soil and ecosystems.",
        "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce food and release oxygen.",
        "Recycling helps reduce waste by processing materials so they can be used again.",
        "Climate change refers to long-term changes in Earth's temperature and weather patterns.",

        # AI
        "Artificial intelligence, or AI, is a field of computer science focused on creating systems that can perform tasks that normally require human intelligence.",
        "Machine learning is a branch of AI in which computer systems learn patterns from data and use those patterns to make predictions or decisions.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "Generative AI can generate content such as text, images, audio, video, or code.",
        "A large language model, or LLM, is an AI model trained on large amounts of text to understand and generate language.",
        "A prompt is an instruction or question given to an AI system.",
        "RAG stands for Retrieval-Augmented Generation. A RAG system retrieves relevant information from a knowledge source and provides it to a language model to help generate an answer.",
        "An embedding is a numerical representation of information such as text that captures semantic meaning.",
        "FAISS is a library developed by Meta for efficient similarity search of dense vectors.",
        "A chatbot is a software application that allows users to communicate with a computer system using natural language.",

        # Computer Science
        "Computer science is the study of computation, algorithms, software, hardware, data, and information processing.",
        "An algorithm is a step-by-step procedure used to solve a problem or perform a task.",
        "A data structure is a way of organizing and storing data so that it can be accessed and modified efficiently.",
        "An array stores elements in a sequence and usually allows access using an index.",
        "A linked list is a data structure made of nodes where each node contains data and a reference to another node.",
        "A stack follows the Last In, First Out principle, commonly called LIFO.",
        "A queue follows the First In, First Out principle, commonly called FIFO.",
        "A tree is a hierarchical data structure consisting of nodes connected by edges.",
        "A binary tree is a tree in which each node can have at most two children.",
        "A database is an organized collection of data that can be stored, managed, and retrieved by computer systems.",
        "SQL stands for Structured Query Language and is commonly used with relational databases.",
        "C++ is a general-purpose programming language commonly used for software development and computer science education.",
        "Python is a high-level programming language commonly used for software development, data science, and artificial intelligence.",
        "Big O notation describes how the time or space requirements of an algorithm grow as input size increases.",

        # Cybersecurity
        "Cybersecurity is the practice of protecting computers, networks, applications, systems, and data from unauthorized access and malicious activity.",
        "The CIA triad stands for Confidentiality, Integrity, and Availability.",
        "Confidentiality means protecting information from unauthorized access.",
        "Integrity means maintaining the accuracy and trustworthiness of data.",
        "Availability means ensuring authorized users can access systems and information when needed.",
        "Authentication is the process of verifying the identity of a user or system.",
        "Authorization determines what an authenticated user or system is allowed to access.",
        "Encryption transforms readable information into an encoded form to help protect it from unauthorized access.",
        "Malware is malicious software designed to damage systems, steal data, disrupt operations, or gain unauthorized access.",
        "Phishing is a social engineering technique that uses deceptive messages or websites to trick people into revealing information.",
        "A firewall monitors and controls network traffic according to security rules.",
        "An intrusion detection system, or IDS, monitors activity and can generate alerts when suspicious activity is detected.",
        "A network packet sniffer captures and analyzes network traffic.",
        "A log is a record of events that occur within a computer system or application.",

        # Networking
        "A computer network is a group of connected devices that can communicate and share resources.",
        "The Internet is a global network of interconnected computer networks.",
        "An IP address is a numerical address used to identify a device or network interface.",
        "HTTP is a protocol used for communication between web browsers and web servers.",
        "HTTPS is HTTP protected using Transport Layer Security, commonly called TLS.",
        "DNS stands for Domain Name System. It translates domain names into IP addresses.",
        "A router forwards data packets between different networks.",
        "Wi-Fi allows compatible devices to connect to networks wirelessly.",

        # Education
        "Active learning involves practicing, solving problems, explaining concepts, and applying knowledge.",
        "Spaced repetition is a learning technique in which information is reviewed at increasing intervals.",
        "Taking notes can help organize important information and make later revision easier.",
        "Practice is important in programming because writing and debugging code develops problem-solving skills.",
        "Breaking a difficult topic into smaller concepts can make learning easier for beginners.",
        "Regular revision can help students remember concepts for longer periods.",

        # Technology
        "Cloud computing provides resources such as storage, processing power, and software services over a network.",
        "An API, or Application Programming Interface, allows different software systems to communicate with each other.",
        "An API key is commonly used as a credential that allows an application to access a service. API keys should be kept private.",
        "Git is a distributed version control system used to track changes in files.",
        "GitHub is a platform commonly used to host Git repositories and collaborate on software projects.",
        "A software library is a collection of reusable code that developers can use in applications.",
        "Google Colab is a cloud-based environment that allows users to run Python code in notebooks.",
        "Gradio is a Python library used to create web interfaces for machine learning and AI applications.",

        # Productivity
        "Setting clear goals can help people organize their time and focus on important tasks.",
        "Breaking a large task into smaller steps can make the task easier to understand and complete.",
        "A consistent study routine can help learners make steady progress over time.",
        "Good communication involves listening carefully, speaking clearly, respecting others, and considering the situation.",

        # BTS
        "BTS is a South Korean music group formed by Big Hit Entertainment and officially debuted on June 13, 2013.",
        "BTS has seven members: RM, Jin, SUGA, j-hope, Jimin, V, and Jung Kook.",
        "RM is a member of BTS and serves as the group's leader.",
        "RM's birth name is Kim Nam-joon.",
        "Jin is a member of BTS. His full name is Kim Seok-jin.",
        "SUGA is a member of BTS. His full name is Min Yoon-gi.",
        "j-hope is a member of BTS. His full name is Jung Ho-seok.",
        "Jimin is a member of BTS. His full name is Park Ji-min.",
        "V is a member of BTS. His full name is Kim Tae-hyung.",
        "Jung Kook is a member of BTS. His full name is Jeon Jung-kook.",
        "The official BTS fandom is called ARMY.",
        "BTS has released music in Korean, Japanese, and English.",
        "BTS has explored themes including youth, friendship, self-love, personal growth, and social issues.",
        "BTS members have participated in individual music projects and solo activities.",
        "BTS is known for combining music with performances, choreography, and visual storytelling."
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode(documents, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, documents

# Visual loading feedback for initial spin-up
with st.spinner("🚀 Loading AI models and indexing knowledge base... (First launch takes ~10-15s)"):
    embedding_model, index, documents = load_rag_pipeline()

# ------------------------------------------------------------
# 2️⃣ RETRIEVAL & CHAT LOGIC
# ------------------------------------------------------------
def retrieve(query, k=4):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.asarray(query_embedding, dtype="float32")
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0] if 0 <= i < len(documents)]

groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ `GROQ_API_KEY` is missing from Streamlit Secrets. Please configure it in your app settings.")
    st.stop()

client = Groq(api_key=groq_api_key)
MODEL_NAME = "openai/gpt-oss-120b"

# Chat State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if user_query := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                context = retrieve(user_query, k=4)
                context_text = "\n\n".join(context)

                prompt = f"""
You are a helpful AI RAG assistant.

Use the following knowledge base to answer the user's question.

KNOWLEDGE BASE:
{context_text}

USER QUESTION:
{user_query}

RULES:
1. Answer clearly and simply.
2. Use the retrieved information.
3. Do not invent facts that are not supported by the knowledge.
4. If there is not enough relevant information, say:
"I don't have enough information about this in my knowledge base."

ANSWER:
"""
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )

                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
      

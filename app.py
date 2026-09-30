import streamlit as st
import os
import math
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Smart PDF Assistant", page_icon="📄", layout="centered")

st.title("📄 AI-Powered Multi-Document Assistant")
st.write("Upload one or more PDFs to summarize, analyze metrics, or ask questions with citations.")

# Retrieve API Key
api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    st.error("Missing Gemini API Key. Please ensure `GEMINI_API_KEY` is set in your `.env` file.")
    st.stop()

# Multi-PDF file uploader
uploaded_files = st.file_uploader(
    "Upload PDF document(s)", 
    type="pdf", 
    accept_multiple_files=True
)

def extract_pdf_data(pdf_files):
    """
    Extracts text while maintaining metadata (source file and page numbers)
    and computes document metrics.
    """
    documents = []
    total_pages = 0
    total_words = 0
    full_text_by_file = {}

    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        file_text = ""
        total_pages += len(pdf_reader.pages)
        
        for i, page in enumerate(pdf_reader.pages):
            extracted = page.extract_text()
            if extracted:
                file_text += extracted + "\n"
                # Store text chunk with metadata for citations
                documents.append(
                    Document(
                        page_content=extracted,
                        metadata={"source": pdf_file.name, "page": i + 1}
                    )
                )
        
        total_words += len(file_text.split())
        full_text_by_file[pdf_file.name] = file_text

    metrics = {
        "files_count": len(pdf_files),
        "total_pages": total_pages,
        "total_words": total_words,
        "reading_time": math.ceil(total_words / 200)  # Average reading speed ~200 wpm
    }
    
    return documents, full_text_by_file, metrics

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def get_vector_store(doc_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=api_key
    )
    return FAISS.from_documents(doc_chunks, embedding=embeddings)

def extract_text_content(response):
    """Parses LLM output cleanly across plain strings and structured content block lists."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            else:
                text_parts.append(str(block))
        return "".join(text_parts)
    return str(content)

def generate_summary(text_map, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
    combined_text = ""
    for filename, text in text_map.items():
        combined_text += f"\n--- Document: {filename} ---\n{text[:3000]}\n"
        
    prompt = f"Provide a clear, comprehensive bullet-point summary of the following document(s):\n\n{combined_text[:6000]}"
    response = llm.invoke(prompt)
    return extract_text_content(response)

def ask_question_with_citations(user_question, vector_store, chat_history, api_key):
    # Retrieve top relevant passages
    docs = vector_store.similarity_search(user_question, k=4)
    
    # Build context with clear metadata markers
    context_blocks = []
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        citations.append(f"• **{source}** (Page {page})")
        context_blocks.append(f"[{source} - Page {page}]: {doc.page_content}")
    
    context = "\n\n".join(context_blocks)
    
    # Format chat history for context
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]])
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question accurately using ONLY the provided document context and prior chat conversation.
    If the answer is not in the context, say "The requested information is not mentioned in the provided documents."

    Recent Conversation:
    {chat_history}

    Document Context:
    {context}

    Question: {question}
    """)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
    chain = prompt | llm
    response = chain.invoke({"chat_history": history_str, "context": context, "question": user_question})
    
    answer_text = extract_text_content(response)
    
    # Deduplicate citations
    unique_citations = list(dict.fromkeys(citations))
    
    return answer_text, unique_citations

if uploaded_files:
    file_identifiers = [f.name for f in uploaded_files]
    
    # Reprocess files if selection changes
    if "current_files" not in st.session_state or st.session_state.current_files != file_identifiers:
        with st.spinner("Processing document(s) and creating embeddings..."):
            raw_docs, text_map, metrics = extract_pdf_data(uploaded_files)
            chunks = split_documents(raw_docs)
            
            st.session_state.vector_store = get_vector_store(chunks, api_key)
            st.session_state.text_map = text_map
            st.session_state.metrics = metrics
            st.session_state.current_files = file_identifiers
            st.session_state.messages = []  # Reset chat history on new files
            st.success("Documents Processed Successfully!")

    # Display Document Metrics Dashboard
    m = st.session_state.metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files", m["files_count"])
    col2.metric("Total Pages", m["total_pages"])
    col3.metric("Total Words", f"{m['total_words']:,}")
    col4.metric("Est. Read Time", f"~{m['reading_time']} min")

    st.divider()

    # App Tabs
    tab1, tab2 = st.tabs(["📝 Document Summary", "💬 Conversational Q&A"])

    # --- TAB 1: SUMMARY & EXPORT ---
    with tab1:
        if st.button("Generate Document Summary"):
            with st.spinner("Analyzing and summarizing..."):
                summary = generate_summary(st.session_state.text_map, api_key)
                st.session_state.summary = summary

        if "summary" in st.session_state:
            st.markdown(st.session_state.summary)
            
            # Export Summary Button
            st.download_button(
                label="📥 Download Summary (.md)",
                data=st.session_state.summary,
                file_name="document_summary.md",
                mime="text/markdown"
            )

    # --- TAB 2: MULTI-TURN CHAT WITH CITATIONS ---
    with tab2:
        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "citations" in message and message["citations"]:
                    with st.expander("📌 Source Citations"):
                        for cite in message["citations"]:
                            st.markdown(cite)

        # Chat Input Box
        if user_query := st.chat_input("Ask a question about your uploaded document(s)..."):
            # Render user query immediately
            st.chat_message("user").markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    answer, citations = ask_question_with_citations(
                        user_query, 
                        st.session_state.vector_store, 
                        st.session_state.messages,
                        api_key
                    )
                    st.markdown(answer)
                    if citations:
                        with st.expander("📌 Source Citations"):
                            for cite in citations:
                                st.markdown(cite)

            # Store assistant response in history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "citations": citations
            })

        # Export Chat History Button
        if st.session_state.messages:
            st.divider()
            formatted_history = "\n\n".join([
                f"**{m['role'].upper()}**: {m['content']}" for m in st.session_state.messages
            ])
            st.download_button(
                label="📥 Download Chat Transcript",
                data=formatted_history,
                file_name="chat_transcript.txt",
                mime="text/plain"
            )

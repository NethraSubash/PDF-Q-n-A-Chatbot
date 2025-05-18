import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 2. Split text into chunks
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# 3. Embed text chunks using MiniLM
def embed_chunks(chunks):
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embed_model)
    return vectorstore

# 4. Load Mistral model for answer generation
def load_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct",
        device_map="auto",  # Use GPU if available
        torch_dtype="auto"
    )
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return llm_pipeline

# 5. Retrieve relevant context from PDF
def get_context(query, vectorstore, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# 6. Generate answer using context
def generate_answer(llm_pipeline, context, question):
    prompt = f"""Answer the question based on the context below:

Context:
{context}

Question:
{question}

Answer:"""
    response = llm_pipeline(prompt, max_new_tokens=200, do_sample=True)
    return response[0]['generated_text'].split("Answer:")[-1].strip()

# 7. Final chatbot function
def answer_question_about_pdf(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    vectorstore = embed_chunks(chunks)
    context = get_context(question, vectorstore)
    llm_pipeline = load_llm_pipeline()
    return generate_answer(llm_pipeline, context, question)

# Example usage
if __name__ == "__main__":
    pdf_path = "example.pdf"  # Replace with your PDF path
    question = "What is the main topic discussed in the document?"
    answer = answer_question_about_pdf(pdf_path, question)
    print(f"Answer: {answer}")

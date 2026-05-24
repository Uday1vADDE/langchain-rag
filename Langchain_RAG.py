from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
# from langchain_community.retrievers import EnsembleRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from sentence_transformers import CrossEncoder
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 — Load PDF
loader = PyPDFLoader("mcp_guide.pdf")
pages = loader.load()
print(f"Pages loaded: {len(pages)}")

# Step 2 — Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"Total chunks: {len(chunks)}")

# Step 3 — Create vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("Vector store created!")

# Step 4 — Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker loaded!")

# Step 5 — Create hybrid retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
print("Hybrid retriever created!")

# Step 6 — Search with hybrid + reranking
def search_with_hybrid_reranking(query, k_final=3):
    print(f"\nQuery: {query}")

    # Hybrid search
    print("Step 1: Hybrid search...")
    candidates = hybrid_retriever.invoke(query)
    print(f"Retrieved {len(candidates)} candidates")

    # Rerank
    print("Step 2: Reranking...")
    pairs = [[query, chunk.page_content] for chunk in candidates]
    scores = reranker.predict(pairs)

    # Sort and pick top k
    ranked = sorted(zip(scores, candidates), reverse=True)
    top_chunks = [chunk for score, chunk in ranked[:k_final]]

    print(f"Step 3: Top {k_final} after reranking:")
    for i, (score, chunk) in enumerate(ranked[:k_final]):
        print(f"Chunk {i+1} — Score: {score:.4f}")
        print(f"Content: {chunk.page_content[:100]}...")

    return top_chunks

# Step 7 — Get answer from LLM


def rewrite_query(query,chat_history,client):
    if not chat_history:
        return query
    messages=[
         {"role": "system", "content": """You are a query rewriter.
        Given chat history and a follow-up question,
        rewrite the question to be standalone and clear.
        Return ONLY the rewritten question, nothing else."""}
    ]

    for msg in chat_history:
        messages.append(msg)
    messages.append(
        {"role": "user", "content": f"Rewrite this question to be standalone: {query}"}
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )

    rewritten=response.choices[0].message.content
    print(f"rewritten query:{rewritten}")
    return rewritten
def get_answer(query,chat_history):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages=[
            {"role": "system", "content": """You are a helpful assistant.
            Answer questions using ONLY the context provided.
            If answer not in context say I don't know."""}]

    search_query=rewrite_query(query, chat_history, client)

    results = search_with_hybrid_reranking(search_query)

    context = "\n\n".join([chunk.page_content for chunk in results])

    
    for msg in chat_history:
        messages.append(msg)

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
        
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return response.choices[0].message.content

#Step 8: Chat loop
# this replaces the single test query
chat_history = []  # empty list — grows with every Q&A

print("\n" + "="*50)
print("RAG Chatbot with Memory ready!")
print("Type 'quit' to exit")
print("="*50 + "\n")

while True:
    # get user input
    query = input("You: ").strip()

    # exit condition
    if query.lower() == "quit":
        print("Goodbye!")
        break

    if query == "":
        continue

    # get answer from RAG
    answer = get_answer(query, chat_history)
    print(f"\nBot: {answer}\n")

    # save Q&A to history
    # next question will see this!
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

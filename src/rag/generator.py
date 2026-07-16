from langchain.chat_models import init_chat_model
from src.rag.vector_store import load_embedding_model, load_vectorstore



embedding = load_embedding_model(model_name= "gemini-embedding-001")
vectorstore= load_vectorstore("data/vectorstore/faiss_index", embedding_model=embedding)

def load_llm(model_name, temperature=0):
    llm = init_chat_model(
        model=model_name,  
        model_provider="google_genai",      
        temperature=temperature,
        max_tokens=1024
        )
    return llm

def run(query, k=2, model_name="gemini-2.5-flash"):     
            
    retrieved_docs = vectorstore.similarity_search(query, k=k)
                    
    context = "\n".join([doc.page_content for doc in retrieved_docs])

            
    prompt = f"""
        Use o contexto abaixo para responder a pergunta.
            
        Contexto:
        {context}

        Pergunta:
        {query}
        """            
    llm = load_llm(model_name= model_name)

    response = llm.invoke(prompt)            

    return response, retrieved_docs
        
# --- BLOCO DE TESTE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔥 SISTEMA RAG PRONTO PARA USO!")
    print("="*50)
    
    pergunta = "Qual o tema principal tratado no documento?"
    print(f"\n🙋 Pergunta: {pergunta}")
    
    # Executa o seu pipeline generator
    response, retrieved= run(query=pergunta)
    
    print("\n🤖 Resposta da LLM:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    print(retrieved)
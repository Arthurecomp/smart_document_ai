#Embeddings, FAISS e Busca

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from src.rag.doc_processor import load_pdf, split_documents
from langchain_community.vectorstores import FAISS


load_dotenv()

# Garante que a API Key do Google está configurada
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Atenção: A variável GOOGLE_API_KEY não foi encontrada no seu arquivo .env!")

def load_embedding_model(model_name= "gemini-embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    return embeddings


def create_vectorstore(documents, embedding_model):   
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore
   
def save_vectorstore(vectorstore, file_path):
    return vectorstore.save_local(file_path)
    

def load_vectorstore(file_path, embedding_model):
    vectorstore = FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization= True)
    return vectorstore
    

# --- BLOCO DE TESTE PRÁTICO ---
if __name__ == "__main__":
    # Vamos criar uma pasta para salvar o nosso banco vetorial se ela não existir
    os.makedirs("data/vectorstore", exist_ok=True)
    
    # IMPORTANTE: Coloque um arquivo PDF qualquer na pasta 'data/' e mude o nome abaixo para testar!
    PDF_TESTE_PATH = "data/pdf1.pdf" 
    FAISS_PATH = "data/vectorstore/faiss_index"
    
    if os.path.exists(PDF_TESTE_PATH):
        print("🚀 Iniciando teste do pipeline RAG (Etapas 1 a 3)...")
        
        # 1. Carrega e divide
        documentos_brutos = load_pdf(PDF_TESTE_PATH)
        chunks = split_documents(documentos_brutos)
        print(f"📝 PDF carregado. Gerados {len(chunks)} chunks de texto.")
        
        # 2. Inicializa o modelo de embeddings do Google
        print("🧠 Inicializando modelo de embeddings do Gemini...")
        embedding_model = load_embedding_model()
        
        # 3. Cria o banco FAISS e salva localmente
        print("💾 Gerando vetores e criando banco FAISS (pode demorar alguns segundos)...")
        db = create_vectorstore(chunks, embedding_model)
        save_vectorstore(db, FAISS_PATH)
        
        # 4. Teste rápido de busca semântica (Retrieval)
        print("\n🔍 Testando busca semântica...")
        query = "Whats is the subject of the document?"
        resultados = db.similarity_search(query, k=2) # Busca os 2 trechos mais parecidos
        
        print(f"\nResultados encontrados para a pergunta: '{query}':")
        for i, doc in enumerate(resultados):
            print(f"\n--- Trecho {i+1} ---")
            print(doc.page_content[:300] + "...") # Mostra as primeiras letras do trecho
            
    else:
        print(f"⚠️ Para testar este script, salve um PDF qualquer em '{PDF_TESTE_PATH}' para rodarmos o pipeline!")

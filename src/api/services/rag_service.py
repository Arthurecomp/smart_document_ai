import os
import tempfile
from langchain_community.vectorstores import FAISS

# 1. Importa os utilitários que VOCÊ já criou no seu src/rag
from src.rag.doc_processor import load_pdf, split_documents
from src.rag.vector_store import load_embedding_model
# Supondo que suas funções load_llm e run/generator estejam em src.rag.generator
from src.rag.generator import load_llm 


class RagService:
    def __init__(self, vectorstore_dir: str = "data/vectorstore"):
        self.vectorstore_dir = vectorstore_dir        
        self.embeddings = load_embedding_model(model_name="gemini-embedding-001")
        os.makedirs(self.vectorstore_dir, exist_ok=True)

    def _get_pdf_vectorstore_path(self, filename: str) -> str:
        """Retorna o caminho único do FAISS para este documento."""
        clean_filename = os.path.splitext(filename)[0]
        return os.path.join(self.vectorstore_dir, clean_filename)

    def index_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> int:
        """Processa os bytes do PDF e salva o vetor isolado."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            documents = load_pdf(tmp_path)
            chunks = split_documents(documents)

            # Salva o FAISS na pasta individual do arquivo
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            save_path = self._get_pdf_vectorstore_path(filename)
            vectorstore.save_local(save_path)

            return len(chunks)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def answer_question(
        self, 
        filename: str, 
        question: str, 
        k: int = 2, 
        model_name: str = "gemini-2.5-flash"
    ) -> str:
        """Executa o SEU fluxo RAG usando o índice isolado do arquivo."""
        save_path = self._get_pdf_vectorstore_path(filename)

        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"O índice do arquivo '{filename}' não foi encontrado. Faça o upload do documento primeiro."
            )

        # 1. Carrega o FAISS exclusivo deste PDF
        vectorstore = FAISS.load_local(
            save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # 2. Busca de similaridade (a mesma do seu generator)
        retrieved_docs = vectorstore.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # 3. Seu Prompt original
        prompt = f"""
        Use o contexto abaixo para responder a pergunta.
            
        Contexto:
        {context}

        Pergunta:
        {question}
        """

        # 4. Carrega o seu LLM via a sua função load_llm
        llm = load_llm(model_name=model_name)

        # 5. Executa e extrai o texto da resposta
        response = llm.invoke(prompt)
        
        # Garante o retorno como string (independente de ser AIMessage ou str)
        return response.content if hasattr(response, 'content') else str(response)

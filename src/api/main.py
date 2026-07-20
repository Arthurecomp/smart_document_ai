from fastapi import FastAPI
from src.api.routers import documents, classification, qa

app = FastAPI(
    title="Smart Document AI",
    description="Backend modular para armazenamento, classificação via BERT e RAG isolado por PDF.",
    version="1.0.0"
)

# Registro das rotas
app.include_router(documents.router)
app.include_router(classification.router)
app.include_router(qa.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Smart Document AI Backend está online!"}

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    message: str = Field(..., example="Arquivo enviado e indexado com sucesso!")
    filename: str = Field(..., example="relatorio.pdf")
    minio_url: str = Field(..., example="minio://documents/relatorio.pdf")
    chunks_indexed: int = Field(..., example="12")


class ClassificationResponse(BaseModel):
    filename: str = Field(..., example="relatorio.pdf")
    category: str = Field(..., example="Financeiro")
    confidence: float = Field(..., example=0.98)

class QueryRequest(BaseModel):
    filename: str = Field(..., description="Nome do PDF indexado para consultar", example="relatorio.pdf")
    question: str = Field(..., description="Sua pergunta sobre o documento", example="Qual foi o faturamento total?")

class QueryResponse(BaseModel):
    filename: str = Field(..., example="relatorio.pdf")
    question: str = Field(..., example="Qual foi o faturamento total?")
    answer: str = Field(..., example="O faturamento total foi de R$ 1.5 milhão...")

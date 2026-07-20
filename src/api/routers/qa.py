from fastapi import APIRouter, HTTPException, status
from src.api.services.rag_service import RagService
from src.api.schemas.requests import QueryRequest, QueryResponse

router = APIRouter(prefix="/rag", tags=["RAG"])
rag_service = RagService()

@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    try:
        # Consulta o FAISS isolado do documento informado via Gemini
        answer = rag_service.answer_question(
            filename=request.filename,
            question=request.question
        )

        return QueryResponse(
            filename=request.filename,
            question=request.question,
            answer=answer
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar a consulta RAG: {str(e)}"
        )

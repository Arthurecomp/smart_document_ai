from fastapi import APIRouter, UploadFile, File, HTTPException, status
from src.api.services.storage_service import StorageService
from src.api.services.rag_service import RagService
from src.api.schemas.requests import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])
storage_service = StorageService()
rag_service = RagService()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Apenas arquivos PDF são permitidos."
        )

    try:
        content = await file.read()

        # 1. Envia os bytes para o MinIO
        minio_url = storage_service.upload_bytes(content, file.filename)

        # 2. Cria e salva o índice FAISS isolado para este PDF
        chunks_count = rag_service.index_pdf_bytes(content, file.filename)

        return DocumentUploadResponse(
            message="Arquivo enviado para o MinIO e indexado no FAISS com sucesso!",
            filename=file.filename,
            minio_url=minio_url,
            chunks_indexed=chunks_count
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar o documento: {str(e)}"
        )

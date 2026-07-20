from fastapi import APIRouter, HTTPException, status
from src.api.services.storage_service import StorageService
from src.api.services.bert_service import BertService
from src.api.schemas.requests import ClassificationResponse

router = APIRouter(prefix="/classification", tags=["Classification"])
storage_service = StorageService()
bert_service = BertService()

@router.post("/classify/{filename}", response_model=ClassificationResponse)
async def classify_document(filename: str):
    try:
        # 1. Baixa os bytes do PDF diretamente do MinIO
        pdf_bytes = storage_service.get_file_bytes(filename)

        # 2. Extrai o texto das primeiras páginas
        extracted_text = bert_service.extract_text_from_bytes(pdf_bytes)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Não foi possível extrair texto do PDF para classificação."
            )

        # 3. Classifica com o seu BERT (arthurmarceldw3/meu-bert-agnews)
        category, confidence = bert_service.classify_text(extracted_text)

        return ClassificationResponse(
            filename=filename,
            category=category,
            confidence=confidence
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Arquivo '{filename}' não foi encontrado no MinIO."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao classificar o documento: {str(e)}"
        )

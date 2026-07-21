import io
import os
from minio import Minio

class StorageService:
    def __init__(self):
        self.endpoint = "localhost:9000"
        self.access_key = "admin"
        self.secret_key = "password12345"
        self.bucket_name = "documentos-brutos"
        self.secure = False

        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Cria o bucket no MinIO caso ainda não exista."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_path: str, object_name: str) -> str:
        """Envia um arquivo físico no disco para o MinIO."""
        self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=file_path
        )
        return f"minio://{self.bucket_name}/{object_name}"

    def upload_bytes(self, content: bytes, object_name: str) -> str:
        """Envia bytes em memória diretamente para o MinIO."""
        data_stream = io.BytesIO(content)
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=data_stream,
            length=len(content),
            content_type="application/pdf"
        )
        return f"minio://{self.bucket_name}/{object_name}"

    def get_file_bytes(self, object_name: str) -> bytes:
        """Baixa o arquivo do MinIO e retorna seus bytes."""
        response = self.client.get_object(
            bucket_name=self.bucket_name,
            object_name=object_name
        )
        content = response.read()
        response.close()
        response.release_conn()
        return content

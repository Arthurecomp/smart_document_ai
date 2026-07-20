import io
import os
from minio import Minio
import pypdf

class MinIOManager:
    def __init__(self, endpoint="localhost:9000", access_key="admin", secret_key="password123"):        
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

    def garantir_bucket(self, bucket_name: str):        
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print(f"🪣 Bucket '{bucket_name}' criado com sucesso!")

    def upload_arquivo(self, bucket_name: str, caminho_local: str, nome_destino: str):        
        self.garantir_bucket(bucket_name)
        self.client.fput_object(bucket_name, nome_destino, caminho_local)
        print(f"📤 Arquivo '{nome_destino}' enviado para o bucket '{bucket_name}'.")

    def ler_texto_pdf(self, bucket_name: str, nome_arquivo: str) -> str:       
        try:
            response = self.client.get_object(bucket_name, nome_arquivo)
            pdf_bytes = response.read()
            response.close()
            response.release_conn()

            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            texto_completo = ""
            for page in reader.pages:
                texto = page.extract_text()
                if texto:
                    texto_completo += texto + "\n"
            return texto_completo
        except Exception as e:
            print(f"❌ Erro ao ler '{nome_arquivo}' do MinIO: {e}")
            return ""

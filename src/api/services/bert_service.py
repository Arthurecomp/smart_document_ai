import io
import pypdf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertService:
    CLASS_MAP = {
        0: "World (Mundo)",
        1: "Sports (Esportes)",
        2: "Business (Negócios)",
        3: "Sci/Tech (Ciência/Tecnologia)"
    }

    def __init__(self, model_name: str = "arthurmarceldw3/meu-bert-agnews"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Carrega o seu modelo do Hugging Face Hub de forma lazy."""
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar o modelo '{self.model_name}' do Hugging Face: {str(e)}")

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extrai o texto das primeiras páginas do PDF para classificação."""
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        # Pega as primeiras 3 páginas para classificar
        for page in reader.pages[:3]:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def classify_text(self, text: str) -> tuple[str, float]:
        """Classifica o texto usando o seu BERT fine-tuned."""
        self._load_model()

        if not text.strip():
            return "Desconhecido", 0.0

        # Tokeniza diretamente para o dispositivo ativo (cuda/cpu)
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)

        # Inferência rápida com torch.inference_mode()
        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()

        category = self.CLASS_MAP.get(predicted_class_id, "Desconhecido")
        return category, round(confidence, 4)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Mapeamento oficial das categorias do AG News
CLASS_MAP = {
    0: "World (Mundo)",
    1: "Sports (Esportes)",
    2: "Business (Negócios)",
    3: "Sci/Tech (Ciência/Tecnologia)"
}

# 1. Aponta diretamente para o SEU repositório na nuvem do Hugging Face!
MODEL_NAME = "arthurmarceldw3/meu-bert-agnews"

print("📥 Carregando o seu modelo e tokenizador do Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def predict(text: str):
    # 2. Tokeniza a entrada diretamente para o dispositivo ativo
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # 3. Roda a inferência rápida (sem gradientes)
    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_id].item()
        
    print("\n" + "="*60)
    print(f"📝 Texto analisado: '{text}'")
    print(f"🏷️ Classificação: {CLASS_MAP[predicted_class_id]}")
    print(f"🎯 Confiança: {confidence:.2%}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("✅ Sistema pronto! Vamos testar:")
    
    # Teste 1: Tecnologia pura
    predict("AMD announced its new line of processors using advanced Zen 5 architecture to boost AI capabilities.")
    
    # Teste 2: Negócios misturado com tecnologia
    predict("Wall Street closed higher today as shares of major tech companies rallied after Nvidia's earnings report.")
import torch
import numpy as np
from src.models.mlp_classifier import MLPClassifier
from src.preprocessing.vectorizer import DocumentVectorizer

# Configura o dispositivo agnóstico (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    modelo = MLPClassifier(input_dim=5000, hidden_dim=128, output_dim=4)    
    
    modelo.load_state_dict(torch.load(path, map_location=device))    
    
    modelo.to(device)
    modelo.eval()
    return modelo


def predict_text(text_input):
    
    document_vectorizer = DocumentVectorizer(max_features=5000)    

    document_vectorizer.load_vectorizer("outputs/meu_vetorizador.pkl")
    
    modelo = load_model("outputs/model_0.pth")
    
    texto_limpo = document_vectorizer.preprocess_text(text_input)
        
    X_dense = document_vectorizer.transform([texto_limpo]).toarray()    

    X_tensor = torch.from_numpy(X_dense).float().to(device)
 
    with torch.inference_mode():
        
        logits = modelo(X_tensor)      
        
        probabilidades = torch.softmax(logits, dim=1)
        classe_escolhida = probabilidades.argmax(dim=1).item()

        if classe_escolhida==0:
            classe_escolhida = "World"
        elif classe_escolhida==1:
            classe_escolhida = "Sports"
        elif classe_escolhida==2:
            classe_escolhida = "Business "
        elif classe_escolhida==3:
            classe_escolhida = " Sci/Tech "
        
        
    return "A classe escolhida é " + classe_escolhida


if __name__ == "__main__":
    frase = "he British Department for Education and Skills (DfES) recently launched a Music Manifesto campaig..."
    resultado = predict_text(frase)
    print(f"🔮 O modelo analisou a frase e escolheu a Classe: {resultado}")

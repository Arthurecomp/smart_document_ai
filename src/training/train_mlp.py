import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.datasets.dataset_loader import load_raw_data
from src.models.mlp_classifier import MLPClassifier
from src.preprocessing.vectorizer import DocumentVectorizer
import os,sys 

test_path = "data\\raw\\test.csv"
train_path = "data\\raw\\train.csv"



# Isso força o Python a incluir a pasta 'src' no buscador de caminhos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


device = "cuda" if torch.cuda.is_available() else "cpu"


def run_training():
    X_train, y_train, X_test, y_test,_,_ = load_raw_data(train_path, test_path)
    
    document_vetorizador = DocumentVectorizer(max_features=5000)



    df_train_tfid = document_vetorizador.fit_transform(X_train)

    df_test_tfid = document_vetorizador.transform(X_test)

    X_train_dense, X_test_dense = to_array(df_train_tfid, df_test_tfid)

    X_train_tensor, y_train_tensor = to_tensor(X_train_dense, y_train)
    X_test_tensor, y_test_tensor = to_tensor(X_test_dense, y_test)

    train_loader = loader_dataset(X_train_tensor, y_train_tensor, True)    
    
    test_loader = loader_dataset(X_test_tensor, y_test_tensor, False)

 
    model_treinado = train(epochs=8, train_loader=train_loader, test_loader=test_loader, input_dim=5000)
    
    save_model("outputs/model_0.pth", model_treinado)
    document_vetorizador.save_vectorizer("outputs/meu_vetorizador.pkl")


def train(epochs, train_loader, test_loader, input_dim):
    model = MLPClassifier(input_dim=input_dim, hidden_dim=128, output_dim=4).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):    
       
        train_loss = 0.0
        train_acc = 0.0    
        model.train() 

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)   
            train_loss += loss.item() 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.argmax(y_pred, dim=1)        
            acertos_do_batch = (y_pred_class == y).sum().item()
            train_acc += acertos_do_batch / len(y)
            
        total_train_batches = len(train_loader)
        loss_train_medio = train_loss / total_train_batches
        acc_train_media = train_acc / total_train_batches
        
      
        test_loss = 0.0
        test_acc = 0.0
        model.eval() # Congela os pesos criados no passo anterior
        
        with torch.inference_mode():
            for batch, (X_test, y_test) in enumerate(test_loader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                
                test_logits = model(X_test)
                loss_t = loss_fn(test_logits, y_test)
                test_loss += loss_t.item()
                
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                acertos_teste_batch = (test_pred == y_test).sum().item()
                test_acc += acertos_teste_batch / len(y_test)
                
        total_test_batches = len(test_loader)
        loss_test_medio = test_loss / total_test_batches
        acc_test_media = test_acc / total_test_batches

   
        print(f"Época {epoch+1}/{epochs} | "
              f"Treino Loss: {loss_train_medio:.4f} - Acc: {acc_train_media:.4f} | "
              f"Teste Loss: {loss_test_medio:.4f} - Acc: {acc_test_media:.4f}")
    
    return model



def save_model(path, model):
    torch.save(obj=model.state_dict(), f=path)
    print(f"💾 Modelo salvo em: {path}")

def to_array(df_train_tfid, df_test_tfid):
    X_train_dense = df_train_tfid.toarray()
    X_test_dense = df_test_tfid.toarray()
    return X_train_dense, X_test_dense

def to_tensor(X_dense, y):
    X_tensor = torch.from_numpy(X_dense).float()
    y_tensor = torch.from_numpy(y.values).long()
    return X_tensor, y_tensor

def loader_dataset(X_tensor, y_tensor, booleano):
    meu_dataset_tensor = TensorDataset(X_tensor, y_tensor)    
    loader = DataLoader(meu_dataset_tensor, batch_size=32, shuffle=booleano)
    return loader


if __name__ == "__main__":
    run_training()

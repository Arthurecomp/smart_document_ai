import torch
from torch import nn



class MLPClassifier(nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.layer_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.relu = nn.ReLU() 
        self.layer_2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)     
    
    def forward(self, x):        
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x





# class DynamicMLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers):
#         super().__init__()
        
#         # Criamos uma lista de módulos especial do PyTorch
#         self.hidden_layers = nn.ModuleList()
#         self.relu = nn.ReLU()
        
#         # 1. Criamos a PRIMEIRA camada (Entrada -> Primeira Oculta)
#         self.hidden_layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
        
#         # 2. Criamos as camadas ocultas do meio dinamicamente
#         # Usamos n_hidden_layers - 1 porque a primeira já foi criada acima
#         for _ in range(n_hidden_layers - 1):
#             self.hidden_layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            
#         # 3. Criamos a camada final (Última Oculta -> Saída de 4 classes)
#         self.layer_out = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    
#     def forward(self, x):        
#         # Passa o dado por dentro de cada camada criada dinamicamente na lista
#         for layer in self.hidden_layers:
#             x = self.relu(layer(x))
            
#         # Passa pela camada de saída final (retorna os logits puros)
#         x = self.layer_out(x)
#         return x

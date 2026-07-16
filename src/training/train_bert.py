from src.datasets.dataset_loader import load_raw_data

import transformers

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from transformers import DataCollatorWithPadding
import numpy as np
from transformers import TrainingArguments, Trainer

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


test_path = "data\\raw\\test.csv"
train_path = "data\\raw\\train.csv"


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de processamento ativo: {device.upper()}")
if device == "cuda":
    print(f"GPU Detectada: {torch.cuda.get_device_name(0)}")
else:
    print("GPU não detectada! Vá em Ambiente de Execução -> Alterar tipo e ative a T4.")


chekpoint = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(chekpoint)

def modelo (chekpoint: str, label: int):
    model = AutoModelForSequenceClassification.from_pretreined(chekpoint, num_labels= label)
    return model

def tokenize_function(examples):    
    return tokenizer(examples['text'], truncation=True)





def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Coleta precisão, recall e f1 de forma ponderada
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run():   
    _,_,_,_,df_train, df_test = load_raw_data(test_path, train_path )
    df_train = df_train.rename(columns={'class': 'label'})
    df_test = df_test.rename(columns={'class': 'label'})

    df_train = df_train[['text', 'label']] #só mudando posição
    df_test = df_test[['text', 'label']]

    if df_train['label'].min() == 1:
        df_train['label'] = df_train['label'] - 1
        df_test['label'] = df_test['label'] - 1

    # Convertendo os DataFrames do Pandas para o formato Dataset do Hugging Face (mais otimizado para memória)
    train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
    test_dataset = Dataset.from_pandas(df_test[['text', 'label']])
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    return tokenized_train, tokenized_test, data_collator
    
    

tokenized_train, tokenized_test, data_collator = run()


training_args = TrainingArguments(
    output_dir="./resultados_bert",     # Diretório onde os checkpoints serão salvos
    eval_strategy="epoch",                   # Avalia a acurácia a cada final de época
    save_strategy="epoch",                   # Salva o progresso a cada época
    learning_rate=2e-5,                      # Taxa de aprendizado baixa recomendada para Transformer fine-tuning
    per_device_train_batch_size=16,          # Tamanho do lote de dados (reduza para 8 se der erro de falta de memória RAM/OOM)
    per_device_eval_batch_size=16,
    num_train_epochs=2,                      # 2 épocas já são suficientes para superar a Baseline sem sofrer Overfitting
    weight_decay=0.01,                       # Regularização para evitar que o modelo decore os dados
    load_best_model_at_end=True,             # Garante que o melhor modelo final será selecionado
    metric_for_best_model="f1",              # A nossa métrica de decisão será o F1-Score
    fp16=True,                               # ATIVAÇÃO DO FP16 (Usa precisão mista de float16 para acelerar a T4 em até 3x!)
    logging_steps=100                        # Reporta o log a cada 100 passos
)

trainer = Trainer(
    model=modelo(chekpoint=chekpoint, label=4),
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)



trainer.train()


raw_predictions = trainer.predict(tokenized_test)
y_pred = np.argmax(raw_predictions.predictions, axis=-1)
y_true = raw_predictions.label_ids

print(classification_report(y_true, y_pred, target_names=["Classe 1", "Classe 2", "Classe 3", "Classe 4"]))
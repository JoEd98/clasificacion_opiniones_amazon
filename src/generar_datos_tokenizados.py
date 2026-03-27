import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( device )

dataset = load_dataset("SetFit/amazon_reviews_multi_es")

textos = dataset["train"]["text"]
palabras_maximas = 0
palabras_minimas = np.inf
for texto in textos:
    numero_palabras = len( texto.split() )
    if palabras_maximas < numero_palabras:
        palabras_maximas = numero_palabras
    if palabras_minimas > numero_palabras:
        palabras_minimas = numero_palabras

model_name = "xlm-roberta-large"
tokenizador = AutoTokenizer.from_pretrained(model_name)

def tokenizar(ejemplo):
    return tokenizador( ejemplo["text"], truncation=True, padding=True, max_length=512 )

datos_tokenizados = dataset.map(tokenizar, batched=True)

datos_tokenizados = datos_tokenizados.remove_columns(["text", "label_text"])
datos_tokenizados.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizador)

tokenizador.save_pretrained("./tokenizer")
datos_tokenizados.save_to_disk("./dataset")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers import Trainer
import evaluate
from evaluate import load
import numpy as np
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import hydra 
import logging 
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd 
from hydra.core.hydra_config import HydraConfig
import os

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dataset = load_dataset("SetFit/amazon_reviews_multi_es")

    control= "xlm-roberta-large"
    tokenizador= AutoTokenizer.from_pretrained(control)
    modelo= AutoModelForSequenceClassification.from_pretrained(control,num_labels=5)

    for param in modelo.base_model.parameters():
        param.requires_grad = False

    def tokenizar(ejemplo):
        return tokenizador(ejemplo["text"],truncation=True)
    
    columnas = dataset["train"].column_names
    columnas.remove("label")
    datos_tokenizados = dataset.map(tokenizar, batched=True, remove_columns=columnas)
    datos_tokenizados

    recopilador_datos = DataCollatorWithPadding(tokenizer=tokenizador)

    metrica = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predicciones, etiquetas = eval_pred
        predicciones = np.argmax(predicciones, axis=1)
        return metrica.compute(references=etiquetas, predictions=predicciones)

    training_args = TrainingArguments(
        output_dir="modelo-ajustado",

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,

        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=datos_tokenizados["train"].shuffle(seed=42).select(range(2000)),
        eval_dataset=datos_tokenizados["validation"].shuffle(seed=42).select(range(400)),
        data_collator=recopilador_datos,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()

    import matplotlib.pyplot as plt

    logs = pd.DataFrame(trainer.state.log_history)

    train_loss = logs[logs["loss"].notna()].copy()
    train_loss["epoch_int"] = train_loss["epoch"].astype(int)
    train_loss_epoch = train_loss.groupby("epoch_int")["loss"].mean()

    eval_loss = logs[logs["eval_loss"].notna()].copy()
    eval_loss["epoch_int"] = eval_loss["epoch"].astype(int)
    eval_loss_epoch = eval_loss.groupby("epoch_int")["eval_loss"].mean()

    best_epoch = eval_loss_epoch.idxmin()
    best_value = eval_loss_epoch.min()

    ruta = Path("./Graficas/epoca_vs_perdida.png")
    plt.figure(figsize=(8,5))
    plt.plot(train_loss_epoch.index, train_loss_epoch.values, marker="o", label="Train Loss")
    plt.plot(eval_loss_epoch.index, eval_loss_epoch.values, marker="o", label="Validation Loss")
    plt.scatter(best_epoch, best_value)
    plt.text(best_epoch, best_value, "  Mejor modelo", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss (por epoch REAL)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(ruta)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )

        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    best_model_path = trainer.state.best_model_checkpoint
    print("Mejor modelo en:", best_model_path)

    modelo_mejor = AutoModelForSequenceClassification.from_pretrained(best_model_path)

    eval_args = TrainingArguments(
        output_dir="eval",
        per_device_eval_batch_size=8
    )

    trainer_best = Trainer(
        model=modelo_mejor,
        args=eval_args,
        data_collator=recopilador_datos
    )

    resultados_test = trainer_best.predict(datos_tokenizados["test"].shuffle(seed=42).select(range(400)))

    preds = np.argmax(resultados_test.predictions, axis=1)
    labels = resultados_test.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )

    acc = accuracy_score(labels, preds)

    logger.info("\nResultados en Test:")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")

    data = {
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "weight_decay": cfg.weight_decay,
        'accuracy':acc,
        'precision':precision,
        'recall': recall,
        'f1': f1,
        'promedio': np.mean( [ acc , precision , recall , f1 ] )
    }

    job_id = HydraConfig.get().job.num
    data["job_id"] = job_id

    project_root = get_original_cwd()
    data_path = os.path.join(project_root, "resultados/resultados_experimentos.csv")
    ruta = Path(data_path)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    nuevo_df = pd.DataFrame([data])

    if ruta.exists():
        df = pd.read_csv(ruta)
        df = pd.concat([df, nuevo_df], ignore_index=True)
    else:
        df = nuevo_df

    df.to_csv(ruta, index=False)

if __name__ == "__main__":
    main()
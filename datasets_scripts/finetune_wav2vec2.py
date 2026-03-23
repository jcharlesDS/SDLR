"""
Script de fine-tuning de Wav2Vec2 pour classification de langue FR/EN.

Utilise facebook/wav2vec2-large-xlsr-53 pré-entraîné et le fine-tune pour 
distinguer français et anglais.

Usage:
    python finetune_wav2vec2.py --dataset-dir training_data/wav2vec2_dataset
    python finetune_wav2vec2.py --dataset-dir training_data/wav2vec2_dataset --epochs 5 --batch-size 8
"""

import argparse
import evaluate
import librosa
import numpy as np
import sys
import torch
from dataclasses import dataclass
from datasets import load_from_disk
from pathlib import Path

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Optional, Union


@dataclass
class DataCollatorWithPadding:
    """
    Data collator avec padding dynamique pour les inputs audio.
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Séparer les inputs et labels
        input_features = []
        labels = []
        
        for feature in features:
            # Extraire l'audio array
            audio = feature["audio"]["array"]
            sampling_rate = feature["audio"]["sampling_rate"]
            
            # Resample si nécessaire (Wav2Vec2 requiert 16kHz)
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            
            # Limiter la durée (30 secondes max pour éviter OOM)
            max_samples = 16000 * 30
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            input_features.append({"input_values": audio})
            labels.append(feature["label"])
        
        # Padding des inputs avec le feature extractor
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Ajouter les labels
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch


def compute_metrics(eval_pred):
    """
    Calcule les métriques d'évaluation (accuracy, F1).
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


def preprocess_function(examples, feature_extractor):
    """
    Pré-traite les exemples audio.
    """
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    # Extraire les features
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tuner Wav2Vec2 pour classification FR/EN")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Dossier contenant le dataset préparé")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-large-xlsr-53",
                        help="Modèle Wav2Vec2 pré-entraîné à fine-tuner")
    parser.add_argument("--output-dir", type=str, default="../trained_models/wav2vec2-language-id",
                        help="Dossier de sortie pour le modèle fine-tuné")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Taille de batch (réduire si OOM)")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Nombre de warmup steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Utiliser le mixed precision training (nécessite GPU)")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Sauvegarder tous les N steps")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  FINE-TUNING WAV2VEC2 - CLASSIFICATION FR/EN")
    print("=" * 70)
    
    # Vérifier CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Charger le dataset
    print(f"\nChargement du dataset depuis {args.dataset_dir}...")
    dataset = load_from_disk(args.dataset_dir)
    
    print(f"\nStatistiques du dataset:")
    for split_name, split_data in dataset.items():
        n_fr = sum(1 for ex in split_data if ex["label"] == 1)
        n_en = sum(1 for ex in split_data if ex["label"] == 0)
        print(f"   {split_name:12s}: {len(split_data):5d} ({n_fr} FR, {n_en} EN)")
    
    # Charger le feature extractor
    print(f"\nChargement du feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    
    # Charger le modèle avec tête de classification
    print(f"\nChargement du modèle {args.model_name}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # 2 classes: FR et EN
        label2id={"en": 0, "fr": 1},
        id2label={0: "en", 1: "fr"}
    )
    
    model.to(device)
    
    # Préparer le data collator
    data_collator = DataCollatorWithPadding(feature_extractor=feature_extractor)
    
    # Configuration de l'entraînement
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16 and device == "cuda",
        push_to_hub=False,
        save_total_limit=2,  # Garder seulement les 2 meilleurs checkpoints
        report_to=["tensorboard"],
        dataloader_num_workers=4,
    )
    
    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Entraînement
    print(f"\nDébut de l'entraînement...")
    print(f"   Époques: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation}")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Total steps: ~{len(dataset['train']) * args.epochs // (args.batch_size * args.gradient_accumulation)}")
    print()
    
    train_result = trainer.train()
    
    # Sauvegarder le modèle final
    print(f"\nSauvegarde du modèle final...")
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))
    
    # Évaluation finale sur le test set
    print(f"\nÉvaluation finale sur le test set...")
    test_results = trainer.evaluate(dataset["test"])
    
    print(f"\nEntraînement terminé!")
    print(f"\nRésultats finaux:")
    print(f"   Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"   F1-Score: {test_results['eval_f1']:.4f}")
    print(f"   Loss:     {test_results['eval_loss']:.4f}")
    
    print(f"\nModèle sauvegardé dans: {output_dir}")
    print(f"\nPour utiliser le modèle:")
    print(f"   1. Le modèle est automatiquement chargé par l'application")
    print(f"   2. Ou testez avec: python evaluate_language_models.py")
    
    # Sauvegarder les métriques
    metrics_file = output_dir / "final_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Fine-tuning Wav2Vec2 - Classification FR/EN\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Modèle: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset_dir}\n\n")
        f.write(f"Hyperparamètres:\n")
        f.write(f"  Époques: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n\n")
        f.write(f"Résultats test set:\n")
        f.write(f"  Accuracy: {test_results['eval_accuracy']:.4f}\n")
        f.write(f"  F1-Score: {test_results['eval_f1']:.4f}\n")
        f.write(f"  Loss:     {test_results['eval_loss']:.4f}\n")
    
    print(f"   Métriques sauvegardées: {metrics_file}")


if __name__ == "__main__":
    main()

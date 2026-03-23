"""
Script d'évaluation et comparaison des modèles de détection de langue.

Compare les performances de :
- CNN fine-tuné (ancien modèle)
- Wav2Vec2 fine-tuné (nouveau modèle)  
- Whisper seul
- Système de vote (Whisper + Wav2Vec2)

Usage:
    python evaluate_language_models.py --test-dataset training_data/wav2vec2_dataset
    python evaluate_language_models.py --test-audio test_audio/ --ground-truth labels.json
"""

import argparse
import json
import sys
import soundfile as sf
import tempfile
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports des modèles
from models.language_id import LanguageIdentifier
from models.language_classifier import LanguageClassifierInference
from models.wav2vec2_language_id import Wav2Vec2LanguageID


class ModelEvaluator:
    """Classe pour évaluer et comparer les modèles de détection de langue."""
    
    def __init__(self, device="auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        print(f"Device: {self.device}")
        self.results = {}
        
    def load_cnn_model(self, model_path="../trained_models/lang_classifier.pth"):
        """Charge le modèle CNN fine-tuné."""
        print(f"\nChargement CNN fine-tuné...")
        try:
            self.cnn_model = LanguageClassifierInference(model_path=model_path, device=self.device)
            print("CNN chargé")
            return True
        except Exception as e:
            print(f"Erreur: {e}")
            self.cnn_model = None
            return False
    
    def load_wav2vec2_model(self, model_path="../trained_models/wav2vec2-language-id"):
        """Charge le modèle Wav2Vec2 fine-tuné."""
        print(f"\nChargement Wav2Vec2 fine-tuné...")
        try:
            self.wav2vec2_model = Wav2Vec2LanguageID(model_path=model_path, device=self.device)
            print("Wav2Vec2 chargé")
            return True
        except Exception as e:
            print(f"Erreur: {e}")
            self.wav2vec2_model = None
            return False
    
    def load_combined_system(self):
        """Charge le système complet (Whisper + Wav2Vec2)."""
        print(f"\nChargement système complet (Whisper + Wav2Vec2)...")
        try:
            self.combined_system = LanguageIdentifier(device=self.device, use_finetuned=True)
            print("Système complet chargé")
            return True
        except Exception as e:
            print(f"Erreur: {e}")
            self.combined_system = None
            return False
    
    def evaluate_on_dataset(self, dataset_path, max_samples=None):
        """
        Évalue tous les modèles sur un dataset HuggingFace.
        
        Args:
            dataset_path: Chemin vers le dataset préparé
            max_samples: Limite d'exemples à évaluer (None = tous)
        """
        print(f"\nChargement du test set depuis {dataset_path}...")
        dataset = load_from_disk(dataset_path)["test"]
        
        if max_samples and len(dataset) > max_samples:
            print(f"   Limitation à {max_samples} exemples")
            dataset = dataset.select(range(max_samples))
        
        print(f"   {len(dataset)} exemples à évaluer")
        
        # Préparer les structures de résultats
        models = {}
        if self.cnn_model:
            models["CNN"] = {"predictions": [], "confidences": []}
        if self.wav2vec2_model:
            models["Wav2Vec2"] = {"predictions": [], "confidences": []}
        if self.combined_system:
            models["System (Vote)"] = {"predictions": [], "confidences": [], "methods": []}
            models["Whisper Only"] = {"predictions": [], "confidences": []}
        
        true_labels = []
        
        # Évaluer chaque exemple
        print(f"\nÉvaluation en cours...")
        for example in tqdm(dataset, desc="Évaluation"):
            # Récupérer l'audio et le label
            audio = example["audio"]["array"]
            sampling_rate = example["audio"]["sampling_rate"]
            true_label = example["label"]  # 0=EN, 1=FR
            true_labels.append(true_label)
            
            # Sauvegarder temporairement l'audio (les modèles attendent un fichier)
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio, sampling_rate)
            
            # Test CNN
            if self.cnn_model:
                try:
                    result = self.cnn_model.predict(temp_file.name)
                    pred = 0 if result["language"] == "en" else 1
                    models["CNN"]["predictions"].append(pred)
                    models["CNN"]["confidences"].append(result["confidence"])
                except Exception as e:
                    models["CNN"]["predictions"].append(-1)
                    models["CNN"]["confidences"].append(0.0)
            
            # Test Wav2Vec2
            if self.wav2vec2_model:
                try:
                    result = self.wav2vec2_model.predict(temp_file.name)
                    pred = 0 if result["language"] == "en" else 1
                    models["Wav2Vec2"]["predictions"].append(pred)
                    models["Wav2Vec2"]["confidences"].append(result["confidence"])
                except Exception as e:
                    models["Wav2Vec2"]["predictions"].append(-1)
                    models["Wav2Vec2"]["confidences"].append(0.0)
            
            # Test système complet
            if self.combined_system:
                try:
                    result = self.combined_system.detect_language(temp_file.name)
                    pred = 0 if result["language"] == "en" else 1
                    models["System (Vote)"]["predictions"].append(pred)
                    models["System (Vote)"]["confidences"].append(result["confidence"])
                    models["System (Vote)"]["methods"].append(result["method"])
                    
                    # Extraire Whisper seul
                    if "whisper" in result:
                        whisper_pred = 0 if result["whisper"]["language"] == "en" else 1
                        models["Whisper Only"]["predictions"].append(whisper_pred)
                        models["Whisper Only"]["confidences"].append(result["whisper"]["confidence"])
                except Exception as e:
                    models["System (Vote)"]["predictions"].append(-1)
                    models["System (Vote)"]["confidences"].append(0.0)
                    models["System (Vote)"]["methods"].append("error")
            
            # Nettoyer le fichier temporaire
            Path(temp_file.name).unlink()
        
        # Calculer les métriques
        self.results = self._compute_metrics(models, true_labels)
        
        return self.results
    
    def _compute_metrics(self, models, true_labels):
        """Calcule les métriques pour chaque modèle."""
        results = {}
        
        for model_name, data in models.items():
            predictions = np.array(data["predictions"])
            confidences = np.array(data["confidences"])
            true = np.array(true_labels)
            
            # Filtrer les prédictions invalides (-1)
            valid_mask = predictions != -1
            predictions = predictions[valid_mask]
            true_filtered = true[valid_mask]
            confidences = confidences[valid_mask]
            
            if len(predictions) == 0:
                continue
            
            # Accuracy
            accuracy = np.mean(predictions == true_filtered)
            
            # Accuracy par classe
            en_mask = true_filtered == 0
            fr_mask = true_filtered == 1
            
            en_accuracy = np.mean(predictions[en_mask] == true_filtered[en_mask]) if en_mask.sum() > 0 else 0
            fr_accuracy = np.mean(predictions[fr_mask] == true_filtered[fr_mask]) if fr_mask.sum() > 0 else 0
            
            # Confiance moyenne
            avg_confidence = np.mean(confidences)
            
            # Classification report
            report = classification_report(
                true_filtered, predictions,
                target_names=["EN", "FR"],
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(true_filtered, predictions)
            
            results[model_name] = {
                "accuracy": accuracy,
                "en_accuracy": en_accuracy,
                "fr_accuracy": fr_accuracy,
                "avg_confidence": avg_confidence,
                "f1_score": report["weighted avg"]["f1-score"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "confusion_matrix": cm,
                "classification_report": report
            }
            
            # Statistiques spécifiques au système de vote
            if model_name == "System (Vote)" and "methods" in data:
                methods = [m for i, m in enumerate(data["methods"]) if valid_mask[i]]
                methods_count = {m: methods.count(m) for m in set(methods)}
                results[model_name]["vote_methods"] = methods_count
        
        return results
    
    def print_results(self):
        """Affiche les résultats de manière formatée."""
        print("\n" + "=" * 70)
        print("  RÉSULTATS COMPARATIFS")
        print("=" * 70)
        
        # Tableau comparatif
        print(f"\n{'Modèle':<20} {'Accuracy':<12} {'F1-Score':<12} {'Confiance':<12}")
        print("-" * 70)
        
        for model_name, metrics in sorted(self.results.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"{model_name:<20} {metrics['accuracy']:.4f}       "
                f"{metrics['f1_score']:.4f}       {metrics['avg_confidence']:.4f}")
        
        # Détails par modèle
        for model_name, metrics in self.results.items():
            print(f"\n{'='*70}")
            print(f"  {model_name}")
            print(f"{'='*70}")
            print(f"  Accuracy globale:  {metrics['accuracy']:.4f}")
            print(f"  Accuracy EN:       {metrics['en_accuracy']:.4f}")
            print(f"  Accuracy FR:       {metrics['fr_accuracy']:.4f}")
            print(f"  F1-Score:          {metrics['f1_score']:.4f}")
            print(f"  Precision:         {metrics['precision']:.4f}")
            print(f"  Recall:            {metrics['recall']:.4f}")
            print(f"  Confiance moyenne: {metrics['avg_confidence']:.4f}")
            
            # Matrice de confusion
            print(f"\n  Matrice de confusion:")
            cm = metrics['confusion_matrix']
            print(f"              Prédit EN  Prédit FR")
            print(f"  Vrai EN        {cm[0][0]:<10} {cm[0][1]:<10}")
            print(f"  Vrai FR        {cm[1][0]:<10} {cm[1][1]:<10}")
            
            # Statistiques du vote si disponible
            if "vote_methods" in metrics:
                print(f"\n  Méthodes de décision:")
                for method, count in sorted(metrics["vote_methods"].items(), key=lambda x: -x[1]):
                    print(f"    {method:<20}: {count}")
    
    def save_results(self, output_dir="evaluation_results"):
        """Sauvegarde les résultats dans des fichiers."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les métriques en JSON
        metrics_file = output_path / "metrics.json"
        
        # Convertir les numpy arrays en listes pour JSON
        json_results = {}
        for model, metrics in self.results.items():
            json_results[model] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
                if k != "classification_report"  # Trop verbeux
            }
        
        with open(metrics_file, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nRésultats sauvegardés:")
        print(f"   Métriques: {metrics_file}")
        
        # Générer les graphiques
        self._plot_results(output_path)
    
    def _plot_results(self, output_path):
        """Génère des graphiques de comparaison."""
        # Graphique de comparaison des accuracy
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(self.results.keys())
        accuracies = [self.results[m]["accuracy"] for m in models]
        f1_scores = [self.results[m]["f1_score"] for m in models]
        
        # Accuracy
        axes[0].bar(range(len(models)), accuracies, color='steelblue')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy par modèle')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # F1-Score
        axes[1].bar(range(len(models)), f1_scores, color='coral')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score par modèle')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "comparison.png", dpi=300, bbox_inches='tight')
        print(f"   Graphique: {output_path / 'comparison.png'}")
        
        # Matrices de confusion pour chaque modèle
        for model_name, metrics in self.results.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['EN', 'FR'], yticklabels=['EN', 'FR'])
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Vérité')
            ax.set_title(f'Matrice de confusion - {model_name}')
            
            filename = output_path / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"   Matrices de confusion sauvegardées")


def main():
    parser = argparse.ArgumentParser(description="Évaluer et comparer les modèles de détection de langue")
    parser.add_argument("--test-dataset", type=str,
                        help="Chemin vers le dataset de test préparé")
    parser.add_argument("--max-samples", type=int,
                        help="Limite d'exemples à évaluer (pour tests rapides)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Dossier de sortie pour les résultats")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device à utiliser")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ÉVALUATION ET COMPARAISON DES MODÈLES DE DÉTECTION DE LANGUE")
    print("=" * 70)
    
    # Créer l'évaluateur
    evaluator = ModelEvaluator(device=args.device)
    
    # Charger les modèles disponibles
    evaluator.load_cnn_model()
    evaluator.load_wav2vec2_model()
    evaluator.load_combined_system()
    
    # Évaluer sur le dataset
    if args.test_dataset:
        evaluator.evaluate_on_dataset(args.test_dataset, max_samples=args.max_samples)
        
        # Afficher les résultats
        evaluator.print_results()
        
        # Sauvegarder
        evaluator.save_results(args.output_dir)
        
        print(f"\nÉvaluation terminée!")
    else:
        print("\nErreur: Spécifiez --test-dataset")


if __name__ == "__main__":
    main()

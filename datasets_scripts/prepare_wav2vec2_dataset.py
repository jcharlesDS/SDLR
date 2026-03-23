"""
Script de préparation des données VoxPopuli pour fine-tuning Wav2Vec2.

Ce script télécharge et prépare les datasets VoxPopuli français et anglais
au format requis par HuggingFace Datasets pour le fine-tuning de Wav2Vec2.

VoxPopuli contient des discours du Parlement Européen.

Usage:
    python prepare_wav2vec2_dataset.py --download  # Télécharge automatiquement
    python prepare_wav2vec2_dataset.py --data-dir /path/to/voxpopuli  # Utilise des données locales
"""

import argparse
import sys
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from pathlib import Path

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_local_voxpopuli(data_dir):
    """
    Charge les datasets VoxPopuli depuis un dossier local.
    
    Args:
        data_dir: Dossier contenant les datasets (doit contenir fr/ et en/)
    
    Returns:
        Tuple[Dataset, Dataset]: Datasets français et anglais
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dossier introuvable: {data_dir}")
    
    print(f"Chargement depuis {data_dir}...")
    
    # Vérifier la structure
    fr_path = data_path / "fr"
    en_path = data_path / "en"
    
    if not fr_path.exists() or not en_path.exists():
        raise FileNotFoundError(
            f"Structure invalide. Attendu: {data_dir}/fr/ et {data_dir}/en/\n"
            f"Astuce: Utilisez --download pour télécharger automatiquement"
        )
    
    # Charger les datasets
    print("   Chargement français...")
    dataset_fr = load_from_disk(str(fr_path))
    
    print("   Chargement anglais...")
    dataset_en = load_from_disk(str(en_path))
    
    print(f"\nDatasets chargés:")
    print(f"   Français: {len(dataset_fr)} exemples")
    print(f"   Anglais:  {len(dataset_en)} exemples")
    
    return dataset_fr, dataset_en


def download_voxpopuli(output_dir="../training_data/voxpopuli", cache_dir="../.cache"):
    """
    Télécharge automatiquement les datasets VoxPopuli FR et EN depuis HuggingFace.
    
    Args:
        output_dir: Dossier où sauvegarder les datasets
        cache_dir: Dossier cache HuggingFace
    
    Returns:
        Tuple[Dataset, Dataset]: Datasets français et anglais
    """
    print("Téléchargement des datasets VoxPopuli depuis HuggingFace...")
    print("   VoxPopuli: Discours du Parlement Européen")
    print("   Cela peut prendre du temps (plusieurs GB)...")
    
    try:
        # VoxPopuli
        print("\nChargement français...")
        dataset_fr = load_dataset(
            "facebook/voxpopuli",
            "fr",
            split="train",
            cache_dir=cache_dir
        )
        
        print("\nChargement anglais...")
        dataset_en = load_dataset(
            "facebook/voxpopuli",
            "en",
            split="train",
            cache_dir=cache_dir
        )
        
        print(f"\nDatasets téléchargés:")
        print(f"   Français: {len(dataset_fr)} exemples")
        print(f"   Anglais:  {len(dataset_en)} exemples")
        
        return dataset_fr, dataset_en
        
    except Exception as e:
        print(f"\nErreur de téléchargement: {e}")
        print("\nSolutions:")
        print("   1. Vérifiez votre connexion internet")
        print("   2. Libérez de l'espace disque")
        print("   3. Dataset: https://huggingface.co/datasets/facebook/voxpopuli")
        raise


def prepare_dataset(dataset, language_code, max_samples_per_split=None):
    """
    Prépare un dataset pour l'entraînement Wav2Vec2.
    
    Args:
        dataset: Dataset HuggingFace
        language_code: 'fr' ou 'en'
        max_samples_per_split: Limite d'exemples par split (None = tous)
    
    Returns:
        Dataset préparé avec labels de langue
    """
    print(f"\nPréparation du dataset {language_code.upper()}...")
    
    # Limiter le nombre d'exemples si demandé
    if max_samples_per_split and len(dataset) > max_samples_per_split:
        print(f"Limitation à {max_samples_per_split} exemples")
        dataset = dataset.select(range(max_samples_per_split))
    
    # Ajouter le label de langue
    label = 0 if language_code == "en" else 1  # 0=EN, 1=FR
    
    def add_language_label(example):
        example["label"] = label
        example["language"] = language_code
        return example
    
    dataset = dataset.map(add_language_label, desc=f"Ajout labels {language_code}")
    
    # Pour VoxPopuli: on skip le filtrage par durée pour éviter le décodage audio
    # qui nécessite torchcodec/FFmpeg. Le dataset VoxPopuli est déjà de haute qualité.
    # Le filtrage par durée se fera pendant le fine-tuning si nécessaire.
    print(f"   Dataset size: {len(dataset)} examples")
    
    return dataset


def create_train_val_test_splits(dataset_fr, dataset_en, train_size=2000, val_size=500, test_size=300):
    """
    Crée les splits train/validation/test équilibrés FR+EN.
    
    Args:
        dataset_fr: Dataset français
        dataset_en: Dataset anglais
        train_size: Nombre d'exemples par langue pour train
        val_size: Nombre d'exemples par langue pour validation
        test_size: Nombre d'exemples par langue pour test
    
    Returns:
        DatasetDict avec splits train/validation/test
    """
    print(f"\nCréation des splits:")
    print(f"   Train:      {train_size * 2} exemples ({train_size} FR + {train_size} EN)")
    print(f"   Validation: {val_size * 2} exemples ({val_size} FR + {val_size} EN)")
    print(f"   Test:       {test_size * 2} exemples ({test_size} FR + {test_size} EN)")
    
    # Vérifier qu'on a assez de données
    total_needed_fr = train_size + val_size + test_size
    total_needed_en = train_size + val_size + test_size
    
    if len(dataset_fr) < total_needed_fr:
        raise ValueError(f"Dataset FR insuffisant: {len(dataset_fr)} < {total_needed_fr}")
    if len(dataset_en) < total_needed_en:
        raise ValueError(f"Dataset EN insuffisant: {len(dataset_en)} < {total_needed_en}")
    
    # Shuffle pour randomiser
    dataset_fr = dataset_fr.shuffle(seed=42)
    dataset_en = dataset_en.shuffle(seed=42)
    
    # Splits français
    fr_train = dataset_fr.select(range(train_size))
    fr_val = dataset_fr.select(range(train_size, train_size + val_size))
    fr_test = dataset_fr.select(range(train_size + val_size, train_size + val_size + test_size))
    
    # Splits anglais
    en_train = dataset_en.select(range(train_size))
    en_val = dataset_en.select(range(train_size, train_size + val_size))
    en_test = dataset_en.select(range(train_size + val_size, train_size + val_size + test_size))
    
    # Combiner FR + EN pour chaque split
    
    train_dataset = concatenate_datasets([fr_train, en_train]).shuffle(seed=42)
    val_dataset = concatenate_datasets([fr_val, en_val]).shuffle(seed=42)
    test_dataset = concatenate_datasets([fr_test, en_test]).shuffle(seed=42)
    
    # Créer le DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    print(f"\nSplits créés:")
    for split_name, split_data in dataset_dict.items():
        # Compter sans itérer (évite de décoder l'audio)
        labels = split_data["label"]
        n_fr = sum(1 for label in labels if label == 1)
        n_en = sum(1 for label in labels if label == 0)
        print(f"   {split_name:12s}: {len(split_data):5d} total ({n_fr} FR, {n_en} EN)")
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Préparer les données VoxPopuli pour Wav2Vec2")
    parser.add_argument("--download", action="store_true", 
                        help="Télécharger automatiquement depuis HuggingFace")
    parser.add_argument("--data-dir", type=str, 
                        help="Dossier contenant les datasets VoxPopuli locaux")
    parser.add_argument("--output-dir", type=str, default="../training_data/wav2vec2_dataset",
                        help="Dossier de sortie pour le dataset préparé")
    parser.add_argument("--train-size", type=int, default=2000,
                        help="Nombre d'exemples par langue pour train (défaut: 2000)")
    parser.add_argument("--val-size", type=int, default=500,
                        help="Nombre d'exemples par langue pour validation (défaut: 500)")
    parser.add_argument("--test-size", type=int, default=300,
                        help="Nombre d'exemples par langue pour test (défaut: 300)")
    parser.add_argument("--cache-dir", type=str, default=".cache",
                        help="Dossier cache HuggingFace")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  PRÉPARATION DATASET WAV2VEC2 - VOXPOPULI FR/EN")
    print("=" * 70)
    
    # Charger ou télécharger les datasets
    if args.download:
        dataset_fr, dataset_en = download_voxpopuli(args.output_dir, args.cache_dir)
    elif args.data_dir:
        dataset_fr, dataset_en = load_local_voxpopuli(args.data_dir)
    else:
        print("\nErreur: Spécifiez --download ou --data-dir")
        return
    
    # Préparer les datasets
    dataset_fr = prepare_dataset(dataset_fr, "fr", max_samples_per_split=args.train_size + args.val_size + args.test_size)
    dataset_en = prepare_dataset(dataset_en, "en", max_samples_per_split=args.train_size + args.val_size + args.test_size)
    
    # Créer les splits
    dataset_dict = create_train_val_test_splits(
        dataset_fr, dataset_en,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size
    )
    
    # Sauvegarder le dataset
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSauvegarde dans {output_path}...")
    dataset_dict.save_to_disk(str(output_path))
    
    print(f"\nDataset préparé avec succès!")
    print(f"   Localisation: {output_path}")
    print(f"\nProchaine étape:")
    print(f"   python finetune_wav2vec2.py --dataset-dir {output_path}")


if __name__ == "__main__":
    main()

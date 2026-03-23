import pandas as pd
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_common_voice(lang, corpus_path, output_base="../training_data", train_size=300, val_size=100):
    """
    Extrait et organise les fichiers Common Voice pour l'entraînement.
    
    Args:
        lang: Code langue ('fr' ou 'en')
        corpus_path: Chemin vers le dossier cv-corpus/[lang]
        output_base: Dossier de sortie
        train_size: Nombre de fichiers pour l'entraînement (défaut: 300)
        val_size: Nombre de fichiers pour la validation (défaut: 100)
    """
    
    corpus_path = Path(corpus_path)
    clips_dir = corpus_path / "clips"
    
    # Déterminer quel fichier TSV utiliser (Scripted vs Spontaneous)
    tsv_file = corpus_path / "validated.tsv"
    if not tsv_file.exists():
        # Essayer le format Spontaneous Speech
        tsv_file = corpus_path / f"ss-corpus-{lang}.tsv"
        if not tsv_file.exists():
            print(f"Aucun fichier TSV trouvé dans {corpus_path}")
            print(f"Cherché: validated.tsv ou ss-corpus-{lang}.tsv")
            return
        dataset_type = "Spontaneous Speech"
    else:
        dataset_type = "Scripted Speech"
    
    print(f"\nChargement des métadonnées {lang.upper()} ({dataset_type})...")
    print(f"Fichier: {tsv_file.name}")
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Filtrer les fichiers qui existent et ont une durée raisonnable (2-10 secondes)
    print(f"Total dans {tsv_file.name}: {len(df)} fichiers")
    
    # Vérifier si la colonne 'duration' existe
    if 'duration' in df.columns:
        # Garder seulement les fichiers entre 2 et 10 secondes (durée en millisecondes)
        df = df[(df['duration'] >= 2000) & (df['duration'] <= 10000)]
        print(f"   Après filtrage (2-10s): {len(df)} fichiers")
    elif 'duration_ms' in df.columns:
        # Nom alternatif de la colonne durée
        df = df[(df['duration_ms'] >= 2000) & (df['duration_ms'] <= 10000)]
        print(f"   Après filtrage (2-10s): {len(df)} fichiers")
    else:
        print(f"Colonne durée non trouvée, pas de filtrage par durée")
        print(f"Colonnes disponibles: {', '.join(df.columns[:5])}...")
    
    # Mélanger et sélectionner
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_needed = train_size + val_size
    if len(df) < total_needed:
        print(f"Seulement {len(df)} fichiers disponibles, ajustement...")
        train_size = int(len(df) * 0.75)
        val_size = len(df) - train_size
    
    # Créer les dossiers
    train_dir = Path(output_base) / "train" / lang
    val_dir = Path(output_base) / "val" / lang
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Déterminer le nom de la colonne contenant le chemin du fichier
    path_column = 'path' if 'path' in df.columns else 'audio_file' if 'audio_file' in df.columns else df.columns[0]
    print(f"Colonne utilisée pour les fichiers: '{path_column}'")
    
    # Copier les fichiers d'entraînement
    print(f"\nCopie des fichiers d'entraînement {lang.upper()}...")
    train_df = df.head(train_size)
    copied_train = 0
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Train {lang}"):
        source = clips_dir / row[path_column]
        if source.exists():
            # Convertir MP3 en WAV si besoin (ou garder MP3)
            target = train_dir / f"{lang}_{copied_train:04d}.mp3"
            shutil.copy(source, target)
            copied_train += 1
    
    print(f"{copied_train} fichiers d'entraînement copiés")
    
    # Copier les fichiers de validation
    print(f"\nCopie des fichiers de validation {lang.upper()}...")
    val_df = df.iloc[train_size:train_size + val_size]
    copied_val = 0
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Val {lang}"):
        source = clips_dir / row[path_column]
        if source.exists():
            target = val_dir / f"{lang}_{copied_val:04d}.mp3"
            shutil.copy(source, target)
            copied_val += 1
    
    print(f"{copied_val} fichiers de validation copiés")
    
    return copied_train, copied_val

if __name__ == "__main__":
    # Parser d'arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Prépare les données Common Voice pour le fine-tuning')
    parser.add_argument('--train', type=int, default=300, help='Nombre de fichiers d\'entraînement par langue (défaut: 300)')
    parser.add_argument('--val', type=int, default=100, help='Nombre de fichiers de validation par langue (défaut: 100)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Préparation des données Common Voice pour le fine-tuning")
    print("=" * 70)
    print(f"Configuration:")
    print(f"   Train par langue: {args.train} fichiers")
    print(f"   Val par langue:   {args.val} fichiers")
    print(f"   Total:            {(args.train + args.val) * 2} fichiers")
    print("=" * 70)
    
    # MODIFIEZ CES CHEMINS selon où vous avez extrait les archives
    french_corpus = r"E:\Datasets\corpus-2.0-2025-12-05-fr"
    english_corpus = r"E:\Datasets\1764158905630-sps-corpus-1.0-2025-11-25-en\sps-corpus-1.0-2025-11-25-en"
    
    # Vérifier que les chemins existent
    if not Path(french_corpus).exists():
        print(f"Chemin français non trouvé: {french_corpus}")
        print("Modifiez la variable 'french_corpus' dans le script")
        exit(1)
    
    if not Path(english_corpus).exists():
        print(f"Chemin anglais non trouvé: {english_corpus}")
        print("Modifiez la variable 'english_corpus' dans le script")
        exit(1)
    
    # Préparer les données
    try:
        fr_train, fr_val = prepare_common_voice(
            lang="fr",
            corpus_path=french_corpus,
            train_size=args.train,
            val_size=args.val
        )
        
        en_train, en_val = prepare_common_voice(
            lang="en",
            corpus_path=english_corpus,
            train_size=args.train,
            val_size=args.val
        )
        
        print("\n" + "=" * 70)
        print("PRÉPARATION TERMINÉE!")
        print("=" * 70)
        print(f"\nRésumé:")
        print(f"   Français  - Train: {fr_train}, Val: {fr_val}")
        print(f"   Anglais   - Train: {en_train}, Val: {en_val}")
        print(f"   Total     - Train: {fr_train + en_train}, Val: {fr_val + en_val}")
        print(f"\nRépertoire: ../training_data/")
        print(f"   Structure:")
        print(f"     training_data/")
        print(f"       train/")
        print(f"         fr/  ({fr_train} fichiers)")
        print(f"         en/  ({en_train} fichiers)")
        print(f"       val/")
        print(f"         fr/  ({fr_val} fichiers)")
        print(f"         en/  ({en_val} fichiers)")
        print(f"\nPrêt pour le fine-tuning! Lancez: python train_language_classifier.py")
        
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
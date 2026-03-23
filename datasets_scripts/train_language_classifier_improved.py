"""
Script d'entraînement amélioré pour le CNN avec data augmentation
et plus de données pour mieux généraliser aux vrais fichiers audio
"""
import librosa
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.language_classifier import AudioLanguageClassifier


class AugmentedAudioDataset(Dataset):
    """Dataset avec data augmentation pour améliorer la généralisation"""
    
    def __init__(self, audio_files: list, labels: list, max_time_frames=300, augment=True):
        self.audio_files = audio_files
        self.labels = labels
        self.max_time_frames = max_time_frames
        self.augment = augment
    
    def __len__(self):
        return len(self.audio_files)
    
    def add_noise(self, audio, noise_factor=0.005):
        """Ajoute du bruit gaussien"""
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented
    
    def time_stretch(self, audio, rate=None):
        """Accélère ou ralentit l'audio"""
        if rate is None:
            rate = random.uniform(0.85, 1.15)  # ±15%
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, sr, n_steps=None):
        """Change la hauteur tonale"""
        if n_steps is None:
            n_steps = random.randint(-2, 2)  # ±2 demi-tons
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Charger l'audio
        y, sr = librosa.load(audio_path, sr=16000, duration=10.0)
        
        # Appliquer l'augmentation aléatoirement si activé
        if self.augment and random.random() > 0.5:
            aug_type = random.choice(['noise', 'stretch', 'pitch', 'combined'])
            
            if aug_type == 'noise':
                y = self.add_noise(y)
            elif aug_type == 'stretch':
                y = self.time_stretch(y)
            elif aug_type == 'pitch':
                y = self.pitch_shift(y, sr)
            elif aug_type == 'combined':
                y = self.add_noise(y)
                if random.random() > 0.5:
                    y = self.time_stretch(y)
        
        # Extraire le spectrogramme
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        # Padding ou truncate
        current_time = mel_spec_normalized.shape[1]
        if current_time < self.max_time_frames:
            pad_width = self.max_time_frames - current_time
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_normalized = mel_spec_normalized[:, :self.max_time_frames]
        
        # Convertir en tensor
        x = torch.from_numpy(mel_spec_normalized).float().unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y


def train_model(train_loader, val_loader, device, num_epochs=50, learning_rate=0.0005):
    """Entraîne le modèle AudioLanguageClassifier avec les données augmentées"""
    model = AudioLanguageClassifier().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10  # Early stopping après 10 epochs sans amélioration
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
        f"Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, "
        f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "../trained_models/lang_classifier.pth")
            print(f"Meilleur modèle sauvegardé (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping après {epoch+1} epochs (pas d'amélioration depuis {max_patience} epochs)")
            break
    
    # Calculer et afficher le temps total
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Entraînement terminé en {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")
    print(f"Meilleure précision validation: {best_val_acc:.2f}%")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"Entraînement du classificateur de langue FR/EN")
    print("=" * 70)
    print(f"Device: {device}")
    print("=" * 70)
    
    # Vérifier la structure des données
    train_fr_path = Path("../training_data/train/fr")
    train_en_path = Path("../training_data/train/en")
    val_fr_path = Path("../training_data/val/fr")
    val_en_path = Path("../training_data/val/en")
    
    if not train_fr_path.exists() or not train_en_path.exists():
        print(f"Erreur: Dossiers d'entraînement non trouvés!")
        print(f"Lancez d'abord: python prepare_common_voice_data.py")
        exit(1)
    
    # Charger les fichiers
    print("\nChargement des fichiers...")
    train_fr_files = list(train_fr_path.glob("*.mp3")) + list(train_fr_path.glob("*.wav"))
    train_en_files = list(train_en_path.glob("*.mp3")) + list(train_en_path.glob("*.wav"))
    
    val_fr_files = list(val_fr_path.glob("*.mp3")) + list(val_fr_path.glob("*.wav"))
    val_en_files = list(val_en_path.glob("*.mp3")) + list(val_en_path.glob("*.wav"))
    
    print(f"Train FR: {len(train_fr_files)} fichiers")
    print(f"Train EN: {len(train_en_files)} fichiers")
    print(f"Val FR: {len(val_fr_files)} fichiers")
    print(f"Val EN: {len(val_en_files)} fichiers")
    
    if len(train_fr_files) == 0 or len(train_en_files) == 0:
        print(f"Erreur: Aucun fichier audio trouvé!")
        exit(1)
    
    train_files = train_fr_files + train_en_files
    train_labels = [0] * len(train_fr_files) + [1] * len(train_en_files)  # 0=fr, 1=en
    
    val_files = val_fr_files + val_en_files
    val_labels = [0] * len(val_fr_files) + [1] * len(val_en_files)
    
    print(f"\nTotal - Train: {len(train_files)} fichiers, Val: {len(val_files)} fichiers")
    
    # Créer le dossier pour sauvegarder le modèle
    Path("../trained_models").mkdir(exist_ok=True)
    
    # Créer les datasets AVEC augmentation pour le train, SANS pour la validation
    print("\nPréparation des datasets...")
    print("Augmentation activée pour l'entraînement")
    print("Pas d'augmentation pour la validation")
    
    train_dataset = AugmentedAudioDataset(train_files, train_labels, augment=True)
    val_dataset = AugmentedAudioDataset(val_files, val_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Entraîner
    print("\nDébut de l'entraînement...\n")
    model = train_model(train_loader, val_loader, device, num_epochs=50)
    
    print("\n" + "=" * 70)
    print("Fine-tuning amélioré terminé!")
    print("=" * 70)
    print(f"Modèle sauvegardé: ../trained_models/lang_classifier_improved.pth")

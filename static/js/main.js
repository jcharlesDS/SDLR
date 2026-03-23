// État de l'application
let selectedFile = null;
let fileId = null;

// État du mode batch
let batchMode = false;
let selectedFiles = [];
let batchResults = [];
let currentResultIndex = 0;

// Résultats actuels pour export
let currentResults = null;
let currentAnalysisFileName = '';
let currentAudioUrl = null;

// Éditions utilisateur
let speakerNames = {}; // { speakerId: customName }
let transcriptionEdits = {}; // { segmentIndex: editedText }
let batchEdits = {}; // { resultIndex: { speakerNames: {}, transcriptionEdits: {} } }

// Éléments du DOM
let uploadBox, audioFileInput, analyzeBtn, enableTranscription, useCuda, mergeCollar, mergeCollarValue, numSpeakers;
let progressSection, progressFill, progressText, resultsSection;
let clearUploadsBtn, configTokenBtn, tokenModal, hfTokenInput, saveTokenBtn, cancelTokenBtn, tokenError;
let themeToggle, sunIcon, moonIcon;
let batchModeCheckbox, batchNavigation, prevResultBtn, nextResultBtn, resultCounter, currentFileName, fileNameText;
let exportBtn, exportMenu;
let audioPlayer, audioPlayerCard;

// Vérifier si CUDA est disponible au chargement de la page
window.addEventListener('DOMContentLoaded', async () => {
    // Sélection des éléments du DOM après le chargement complet
    uploadBox = document.getElementById('uploadBox');
    audioFileInput = document.getElementById('audioFile');
    analyzeBtn = document.getElementById('analyzeBtn');
    enableTranscription = document.getElementById('enableTranscription');
    useCuda = document.getElementById('useCuda');
    mergeCollar = document.getElementById('mergeCollar');
    mergeCollarValue = document.getElementById('mergeCollarValue');
    numSpeakers = document.getElementById('numSpeakers');
    progressSection = document.getElementById('progressSection');
    progressFill = document.getElementById('progressFill');
    progressText = document.getElementById('progressText');
    resultsSection = document.getElementById('resultsSection');
    clearUploadsBtn = document.getElementById('clearUploadsBtn');
    configTokenBtn = document.getElementById('configTokenBtn');
    tokenModal = document.getElementById('tokenModal');
    hfTokenInput = document.getElementById('hfToken');
    saveTokenBtn = document.getElementById('saveTokenBtn');
    cancelTokenBtn = document.getElementById('cancelTokenBtn');
    tokenError = document.getElementById('tokenError');
    themeToggle = document.getElementById('themeToggle');
    sunIcon = document.getElementById('sunIcon');
    moonIcon = document.getElementById('moonIcon');
    batchModeCheckbox = document.getElementById('batchMode');
    batchNavigation = document.getElementById('batchNavigation');
    prevResultBtn = document.getElementById('prevResultBtn');
    nextResultBtn = document.getElementById('nextResultBtn');
    resultCounter = document.getElementById('resultCounter');
    currentFileName = document.getElementById('currentFileName');
    fileNameText = document.getElementById('fileNameText');
    exportBtn = document.getElementById('exportBtn');
    exportMenu = document.getElementById('exportMenu');
    audioPlayer = document.getElementById('audioPlayer');
    audioPlayerCard = document.getElementById('audioPlayerCard');
    
    // Initialiser le thème
    initializeTheme();
    
    // Vérifier que les éléments sont trouvés
    if (!mergeCollar || !mergeCollarValue) {
        console.error('Elements manquants:', { mergeCollar, mergeCollarValue });
    }
    
    // Mise à jour de l'affichage du slider merge_collar
    if (mergeCollar && mergeCollarValue) {
        // Initialiser l'affichage avec la valeur par défaut du slider
        mergeCollarValue.textContent = parseFloat(mergeCollar.value).toFixed(1);
        
        mergeCollar.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value).toFixed(1);
            mergeCollarValue.textContent = value;
            console.log('Slider value changed to:', value);
        });
        console.log('Slider initialized with value:', mergeCollar.value);
    }
    
    // Vérifier si le token HuggingFace est configuré
    checkHuggingfaceToken();
    
    // Initialiser les event listeners
    initializeEventListeners();
    
    try {
        const response = await fetch('/health');
        if (response.ok) {
            // Serveur disponible
        }
    } catch (error) {
        console.error('Erreur lors de la vérification du serveur:', error);
    }
});

// Initialiser tous les event listeners
function initializeEventListeners() {
    // Gestion du mode batch
    batchModeCheckbox.addEventListener('change', (e) => {
        batchMode = e.target.checked;
        if (batchMode) {
            audioFileInput.setAttribute('multiple', 'multiple');
            uploadBox.querySelector('h3').textContent = 'Glissez vos fichiers audio ici';
            uploadBox.querySelector('p').textContent = 'ou cliquez pour sélectionner plusieurs fichiers';
        } else {
            audioFileInput.removeAttribute('multiple');
            uploadBox.querySelector('h3').textContent = 'Glissez votre fichier audio ici';
            uploadBox.querySelector('p').textContent = 'ou cliquez pour sélectionner un fichier';
            // Réinitialiser si on désactive le batch
            selectedFiles = [];
            batchResults = [];
            currentResultIndex = 0;
        }
    });
    
    // Navigation batch
    prevResultBtn.addEventListener('click', () => {
        if (currentResultIndex > 0) {
            currentResultIndex--;
            displayResultAtIndex(currentResultIndex);
            updateNavigation();
        }
    });
    
    nextResultBtn.addEventListener('click', () => {
        if (currentResultIndex < batchResults.length - 1) {
            currentResultIndex++;
            displayResultAtIndex(currentResultIndex);
            updateNavigation();
        }
    });
    
    // Export button - toggle menu
    exportBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const isVisible = exportMenu.style.display === 'block';
        exportMenu.style.display = isVisible ? 'none' : 'block';
    });
    
    // Export options
    document.querySelectorAll('.export-option').forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const format = option.dataset.format;
            exportData(format);
            exportMenu.style.display = 'none';
        });
    });
    
    // Close export menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!exportBtn.contains(e.target) && !exportMenu.contains(e.target)) {
            exportMenu.style.display = 'none';
        }
    });
    
    //Gestion du glisser-déposer de fichiers
    uploadBox.addEventListener('click', () => audioFileInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            if (batchMode) {
                handleBatchFileSelect(Array.from(files));
            } else {
                handleFileSelect(files[0]);
            }
        }
    });

    // Sélection de fichier
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            if (batchMode) {
                handleBatchFileSelect(Array.from(e.target.files));
            } else {
                handleFileSelect(e.target.files[0]);
            }
        }
    });

    // Bouton d'analyse
    analyzeBtn.addEventListener('click', async () => {
        if (batchMode) {
            await processBatchFiles();
        } else {
            await processSingleFile();
        }
    });

    // Bouton de configuration du token HuggingFace
    configTokenBtn.addEventListener('click', () => {
        tokenModal.style.display = 'flex';
        hfTokenInput.focus();
    });

    // Bouton de nettoyage des uploads
    clearUploadsBtn.addEventListener('click', async () => {
        if (!confirm('Voulez-vous vraiment supprimer tous les fichiers du dossier uploads ?')) {
            return;
        }
        
        try {
            clearUploadsBtn.disabled = true;
            clearUploadsBtn.textContent = 'Suppression...';
            
            const response = await fetch('/api/uploads/clear', {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                alert(result.message);
            } else {
                alert('Erreur lors de la suppression : ' + result.detail);
            }
        } catch (error) {
            console.error('Erreur:', error);
            alert('Erreur lors de la suppression : ' + error.message);
        } finally {
            clearUploadsBtn.disabled = false;
            clearUploadsBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" style="vertical-align: middle; margin-top: -1px;">
                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" fill="none" stroke="white" stroke-width="2"/>
                    <line x1="10" y1="11" x2="10" y2="17" stroke="white" stroke-width="2"/>
                    <line x1="14" y1="11" x2="14" y2="17" stroke="white" stroke-width="2"/>
                </svg>
                Vider
            `;
        }
    });

    // Modal Token HuggingFace
    saveTokenBtn.addEventListener('click', async () => {
        const token = hfTokenInput.value.trim();
        
        if (!token) {
            showTokenError('Veuillez entrer un token');
            return;
        }
        
        if (!token.startsWith('hf_')) {
            showTokenError('Le token doit commencer par "hf_"');
            return;
        }
        
        try {
            saveTokenBtn.disabled = true;
            saveTokenBtn.textContent = 'Enregistrement...';
            
            const response = await fetch('/api/config/huggingface-token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ token: token })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                tokenModal.style.display = 'none';
                hfTokenInput.value = '';
                alert('✓ Token HuggingFace configuré avec succès !');
            } else {
                showTokenError(result.detail || 'Erreur lors de la sauvegarde');
            }
        } catch (error) {
            console.error('Erreur:', error);
            showTokenError('Erreur de connexion au serveur');
        } finally {
            saveTokenBtn.disabled = false;
            saveTokenBtn.textContent = 'Enregistrer';
        }
    });

    cancelTokenBtn.addEventListener('click', () => {
        tokenModal.style.display = 'none';
        hfTokenInput.value = '';
        tokenError.style.display = 'none';
    });

    // Fermer la modal si on clique en dehors
    tokenModal.addEventListener('click', (e) => {
        if (e.target === tokenModal) {
            tokenModal.style.display = 'none';
            hfTokenInput.value = '';
            tokenError.style.display = 'none';
        }
    });
}

// Gestion de la sélection de fichier
function handleFileSelect(file) {
    selectedFile = file;

    // Mettre à jour l'interface utilisateur
    const content = uploadBox.querySelector('.upload-content');
    content.innerHTML = `
        <svg class="upload-icon" viewBox="0 0 24 24" width="48" height="48">
            <path d="M9 2a1 1 0 0 0-.894.553L7.382 4H4a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-3.382l-.724-1.447A1 1 0 0 0 15 2H9z"/>
            <circle cx="12" cy="13" r="3"/>
        </svg>
        <h3>${file.name}</h3>
        <p>Taille: ${(file.size / (1024 * 1024)).toFixed(2)} MB</p>
        <p style="margin-top: 1rem; color: #6b7280;">Cliquez pour changer de fichier</p>
    `;

    analyzeBtn.disabled = false;
}

// Gestion de la sélection de plusieurs fichiers (batch)
function handleBatchFileSelect(files) {
    // Limiter à 10 fichiers maximum
    if (files.length > 10) {
        alert(`Vous avez sélectionné ${files.length} fichiers. Seuls les 10 premiers seront analysés.\n\nLimite: 10 fichiers maximum en mode batch.`);
        files = Array.from(files).slice(0, 10);
    }
    
    selectedFiles = files;

    // Mettre à jour l'interface utilisateur
    const content = uploadBox.querySelector('.upload-content');
    const totalSize = files.reduce((sum, f) => sum + f.size, 0);
    
    const filesList = files.map(f => `<li>${f.name} (${(f.size / (1024 * 1024)).toFixed(2)} MB)</li>`).join('');
    
    content.innerHTML = `
        <svg class="upload-icon" viewBox="0 0 24 24" width="48" height="48">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
        </svg>
        <h3>${files.length} fichier(s) sélectionné(s)</h3>
        <p>Taille totale: ${(totalSize / (1024 * 1024)).toFixed(2)} MB</p>
        <details style="margin-top: 1rem; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;">
            <summary style="cursor: pointer; color: var(--primary-color); font-weight: 500;">Voir la liste</summary>
            <ul style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary); max-height: 200px; overflow-y: auto;">
                ${filesList}
            </ul>
        </details>
        <p style="margin-top: 1rem; color: #6b7280;">Cliquez pour changer les fichiers</p>
    `;

    analyzeBtn.disabled = false;
}

// Traiter un seul fichier (mode normal)
async function processSingleFile() {
    if (!selectedFile) return;

    try {
        // 1. Uploader le fichier
        progressSection.style.display = 'block';
        resultsSection.style.display = 'none';
        updateProgress(2, 'Upload du fichier...');

        const uploadResult = await uploadFile(selectedFile);
        fileId = uploadResult.file_id;
        currentAudioUrl = uploadResult.audio_url;  // Stocker l'URL audio

        updateProgress(5, 'Fichier uploadé avec succès');

        // 2. Lancer l'analyse avec EventSource pour progression en temps réel
        let streamUrl = `/api/analyze/stream?file_id=${fileId}&enable_transcription=${enableTranscription.checked}&use_cuda=${useCuda.checked}&merge_collar=${parseFloat(mergeCollar.value)}`;
        
        // Ajouter num_speakers seulement si une valeur est saisie
        if (numSpeakers.value && numSpeakers.value.trim() !== '') {
            streamUrl += `&num_speakers=${parseInt(numSpeakers.value)}`;
        }
        
        // Créer l'EventSource pour recevoir les mises à jour de progression
        const eventSource = new EventSource(streamUrl);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                console.error('Erreur:', data.error);
                progressText.textContent = `Une erreur est survenue : ${data.error}`;
                progressText.style.color = 'var(--danger-color)';
                eventSource.close();
                return;
            }
            
            if (data.progress !== undefined) {
                updateProgress(data.progress, data.message);
            }
            
            if (data.step === 'complete' && data.results) {
                eventSource.close();
                setTimeout(() => {
                    displayResults(data.results);
                }, 500);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('Erreur EventSource:', error);
            progressText.textContent = 'Une erreur de connexion est survenue';
            progressText.style.color = 'var(--danger-color)';
            eventSource.close();
        };

    } catch (error) {
        console.error('Erreur:', error);
        progressText.textContent = `Une erreur est survenue : ${error.message}`;
        progressText.style.color = 'var(--danger-color)';
    }
}

// Traiter plusieurs fichiers (mode batch)
async function processBatchFiles() {
    if (selectedFiles.length === 0) return;

    // Réinitialiser les résultats
    batchResults = [];
    currentResultIndex = 0;

    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';

    // Traiter chaque fichier séquentiellement
    for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        
        try {
            updateProgress(0, `Traitement du fichier ${i + 1}/${selectedFiles.length}: ${file.name}`);
            
            // Upload
            updateProgress(5, `Upload de ${file.name}...`);
            const uploadResult = await uploadFile(file);
            
            // Analyse
            const result = await analyzeFileWithProgress(uploadResult.file_id, file.name, i + 1, selectedFiles.length);
            
            // Stocker le résultat avec le nom du fichier, l'URL audio et le file_id
            batchResults.push({
                fileName: file.name,
                results: result,
                audioUrl: uploadResult.audio_url,
                fileId: uploadResult.file_id
            });
            
        } catch (error) {
            console.error(`Erreur pour ${file.name}:`, error);
            // Stocker l'erreur
            batchResults.push({
                fileName: file.name,
                error: error.message
            });
        }
    }

    // Afficher le premier résultat
    currentResultIndex = 0;
    displayBatchResults();
}

// Analyser un fichier avec progression (pour le batch)
async function analyzeFileWithProgress(fileId, fileName, currentFile, totalFiles) {
    return new Promise((resolve, reject) => {
        let streamUrl = `/api/analyze/stream?file_id=${fileId}&enable_transcription=${enableTranscription.checked}&use_cuda=${useCuda.checked}&merge_collar=${parseFloat(mergeCollar.value)}`;
        
        if (numSpeakers.value && numSpeakers.value.trim() !== '') {
            streamUrl += `&num_speakers=${parseInt(numSpeakers.value)}`;
        }
        
        const eventSource = new EventSource(streamUrl);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                eventSource.close();
                reject(new Error(data.error));
                return;
            }
            
            if (data.progress !== undefined) {
                updateProgress(data.progress, `[${currentFile}/${totalFiles}] ${fileName}: ${data.message}`);
            }
            
            if (data.step === 'complete' && data.results) {
                eventSource.close();
                resolve(data.results);
            }
        };
        
        eventSource.onerror = (error) => {
            eventSource.close();
            reject(new Error('Erreur de connexion'));
        };
    });
}

// Afficher les résultats en mode batch
function displayBatchResults() {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Afficher la navigation
    batchNavigation.style.display = 'flex';
    currentFileName.style.display = 'block';
    
    // Afficher le résultat courant
    displayResultAtIndex(currentResultIndex);
    updateNavigation();
}

// Afficher un résultat spécifique par index
function displayResultAtIndex(index) {
    const item = batchResults[index];
    
    if (!item) return;
    
    // Sauvegarder les éditions du résultat précédent (en mode batch)
    if (batchMode && currentResultIndex !== index) {
        saveCurrentEdits(currentResultIndex);
    }
    
    // Restaurer les éditions pour ce résultat
    restoreEdits(index);
    
    // Mettre à jour le nom du fichier, l'URL audio et le file_id
    fileNameText.textContent = item.fileName;
    currentAnalysisFileName = item.fileName;
    currentAudioUrl = item.audioUrl;
    fileId = item.fileId || null;
    
    if (item.error) {
        // Cacher toutes les cartes de résultats
        const resultCards = resultsSection.querySelectorAll('.result-card');
        resultCards.forEach(card => card.style.display = 'none');
        
        // Chercher ou créer un conteneur d'erreur
        let errorContainer = document.getElementById('batchErrorContainer');
        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.id = 'batchErrorContainer';
            // L'insérer après currentFileName
            currentFileName.parentNode.insertBefore(errorContainer, currentFileName.nextSibling);
        }
        
        errorContainer.style.display = 'block';
        errorContainer.innerHTML = `
            <div style="padding: 2rem; text-align: center; background: var(--section-bg); border-radius: 8px; color: var(--danger-color); margin-top: 1rem;">
                <svg viewBox="0 0 24 24" width="48" height="48" style="margin-bottom: 1rem;" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <p style="font-size: 1.2rem; font-weight: 600;">Erreur lors de l'analyse</p>
                <p>${item.error}</p>
            </div>
        `;
    } else {
        // Cacher le conteneur d'erreur s'il existe
        const errorContainer = document.getElementById('batchErrorContainer');
        if (errorContainer) {
            errorContainer.style.display = 'none';
        }
        
        // Afficher les résultats normalement
        displayResults(item.results);
    }
}

// Sauvegarder les éditions actuelles pour un index de résultat
function saveCurrentEdits(index) {
    if (Object.keys(speakerNames).length > 0 || Object.keys(transcriptionEdits).length > 0) {
        batchEdits[index] = {
            speakerNames: { ...speakerNames },
            transcriptionEdits: { ...transcriptionEdits }
        };
    }
}

// Restaurer les éditions pour un index de résultat
function restoreEdits(index) {
    if (batchEdits[index]) {
        speakerNames = { ...batchEdits[index].speakerNames };
        transcriptionEdits = { ...batchEdits[index].transcriptionEdits };
    } else {
        // Réinitialiser si pas d'éditions sauvegardées
        speakerNames = {};
        transcriptionEdits = {};
    }
}

// Mettre à jour la navigation (boutons et compteur)
function updateNavigation() {
    resultCounter.textContent = `${currentResultIndex + 1} / ${batchResults.length}`;
    
    prevResultBtn.disabled = currentResultIndex === 0;
    nextResultBtn.disabled = currentResultIndex === batchResults.length - 1;
}

// Upload du fichier
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur lors de l\'upload du fichier');
    }

    return await response.json();
}

// Analyse du fichier audio
async function analyzeAudio(fileId, enableTranscription, useCuda, mergeCollar) {
    const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: fileId,
            enable_transcription: enableTranscription,
            use_cuda: useCuda,
            merge_collar: mergeCollar
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur lors de l\'analyse du fichier');
    }

    return await response.json();
}

// Mise à jour de la barre de progression
function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
    progressText.style.color = ''; // Réinitialiser la couleur
}

// Formater les secondes en format MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Formater une durée en format MM:SS (arrondi à la seconde supérieure)
function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.ceil(seconds % 60); // Arrondir à la seconde supérieure
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Générer une couleur aléatoire pour un locuteur
function generateRandomColor() {
    // Générer des couleurs HSL pour avoir de belles couleurs saturées
    const hue = Math.floor(Math.random() * 360);
    const saturation = 60 + Math.floor(Math.random() * 20); // 60-80%
    const lightness = 45 + Math.floor(Math.random() * 10); // 45-55%
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

// Créer un mapping de couleurs pour les locuteurs
function createSpeakerColorMap(numSpeakers) {
    const colorMap = {};
    for (let i = 1; i <= numSpeakers; i++) {
        colorMap[i] = generateRandomColor();
    }
    return colorMap;
}

// Affichage des résultats
function displayResults(results) {
    // Stocker les résultats pour l'export
    currentResults = results;
    currentAnalysisFileName = selectedFile ? selectedFile.name : 'analyse';
    
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Masquer la navigation batch si on est en mode single file
    if (!batchMode || batchResults.length === 0) {
        batchNavigation.style.display = 'none';
        currentFileName.style.display = 'none';
    }
    
    // Charger l'audio dans le lecteur
    if (currentAudioUrl && audioPlayer && audioPlayerCard) {
        audioPlayer.src = currentAudioUrl;
        audioPlayer.load();
        audioPlayerCard.style.display = 'block';
    } else {
        if (audioPlayerCard) audioPlayerCard.style.display = 'none';
    }
    
    // Créer un mapping de couleurs unique pour chaque locuteur
    const speakerColors = createSpeakerColorMap(results.num_speakers);
    
    // Temps de traitement
    const processingTime = document.getElementById('processingTime');
    const deviceUsed = document.getElementById('deviceUsed');
    
    if (processingTime) {
        processingTime.textContent = `${results.processing_time}s`;
    }
    if (deviceUsed) {
        deviceUsed.textContent = results.device === 'cuda' ? '🖥️ GPU (CUDA)' : '💻 CPU';
    }

    // Langue
    const languageResult = document.getElementById('languageResult');
    
    if (languageResult) {
        languageResult.innerHTML = `
            <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;">
                ${results.language.language_name}
            </div>
            <div style="color: var(--text-secondary);">
                Confiance : ${(results.language.confidence * 100).toFixed(1)}%
            </div>
        `;
    }

    // Locuteurs
    const speakersResult = document.getElementById('speakersResult');
    
    if (speakersResult) {
        speakersResult.innerHTML = `
            <div style="font-size: 1.25rem; margin-bottom: 1rem;">
                ${results.num_speakers} locuteur(s) détecté(s)
            </div>
        `;
    }

    // Timeline - utiliser les couleurs assignées et rendre les temps cliquables
    const timeline = document.getElementById('timeline');

    if (timeline) {
        timeline.innerHTML = results.speakers.map((segment, index) => `
            <div class="timeline-segment" style="border-color: ${speakerColors[segment.speaker_id]};">
                <div class="speaker-label">
                    <span class="speaker-name" data-speaker-id="${segment.speaker_id}">${getSpeakerName(segment.speaker_id)}</span>
                    <button class="edit-speaker-btn" data-speaker-id="${segment.speaker_id}" title="Renommer le locuteur">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                </div>
                <div class="time-label">
                    <span class="clickable-time" data-time="${segment.start_time}">${formatTime(segment.start_time)}</span> - <span class="clickable-time" data-time="${segment.end_time}">${formatTime(segment.end_time)}</span>
                    (durée: ${formatDuration(segment.duration)})
                </div>
                <div class="segment-export-container">
                    <button class="segment-export-btn" data-segment-index="${index}">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Exporter segment
                    </button>
                    <div class="segment-export-menu" id="exportMenu${index}">
                        <div class="export-option" data-format="mp3" data-segment-index="${index}">MP3</div>
                        <div class="export-option" data-format="wav" data-segment-index="${index}">WAV</div>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Ajouter les listeners pour les timestamps cliquables
        const timeElements = timeline.querySelectorAll('.clickable-time');
        timeElements.forEach(timeElement => {
            timeElement.addEventListener('click', () => {
                const time = parseFloat(timeElement.dataset.time);
                seekAudio(time);
            });
        });
        
        // Ajouter les listeners pour les boutons d'export de segment
        const exportBtns = timeline.querySelectorAll('.segment-export-btn');
        exportBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const segmentIndex = btn.dataset.segmentIndex;
                const menu = document.getElementById(`exportMenu${segmentIndex}`);
                
                // Fermer tous les autres menus
                document.querySelectorAll('.segment-export-menu').forEach(m => {
                    if (m !== menu) m.style.display = 'none';
                });
                
                // Toggle ce menu
                menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
            });
        });
        
        // Ajouter les listeners pour les options d'export
        const exportOptions = timeline.querySelectorAll('.segment-export-menu .export-option');
        exportOptions.forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                const segmentIndex = parseInt(option.dataset.segmentIndex);
                const format = option.dataset.format;
                const segment = results.speakers[segmentIndex];
                
                exportSegment(segment, format);
                
                // Fermer le menu
                document.getElementById(`exportMenu${segmentIndex}`).style.display = 'none';
            });
        });
        
        // Fermer les menus en cliquant ailleurs
        document.addEventListener('click', () => {
            document.querySelectorAll('.segment-export-menu').forEach(menu => {
                menu.style.display = 'none';
            });
        });
        
        // Ajouter les listeners pour l'édition des noms de locuteurs
        const editSpeakerBtns = timeline.querySelectorAll('.edit-speaker-btn');
        editSpeakerBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const speakerId = parseInt(btn.dataset.speakerId);
                renameSpeaker(speakerId);
            });
        });
    }

    // Transcription - utiliser les mêmes couleurs et rendre les temps cliquables
    if (results.transcription && results.transcription.length > 0) {
        const transcriptionCard = document.getElementById('transcriptionCard');
        const transcriptionResult = document.getElementById('transcriptionResult');

        transcriptionCard.style.display = 'block';

        transcriptionResult.innerHTML = results.transcription.map((segment, index) => `
            <div class="transcription-item" data-segment-index="${index}" style="border-left: 4px solid ${speakerColors[segment.speaker_id]};">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; align-items: center;">
                    <span class="speaker-label">
                        <span class="speaker-name" data-speaker-id="${segment.speaker_id}">${getSpeakerName(segment.speaker_id)}</span>
                    </span>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <button class="edit-transcript-btn" data-segment-index="${index}" title="Éditer la transcription">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <span class="time-label">
                            <span class="clickable-time" data-time="${segment.start_time}">${formatTime(segment.start_time)}</span> - <span class="clickable-time" data-time="${segment.end_time}">${formatTime(segment.end_time)}</span>
                        </span>
                    </div>
                </div>
                <div class="transcript-text" style="color: var(--text-primary); line-height: 1.6;">
                    ${getTranscriptText(index, segment.text)}
                </div>
            </div>
        `).join('');
        
        // Ajouter les listeners pour l'édition de transcription
        const editTranscriptBtns = transcriptionResult.querySelectorAll('.edit-transcript-btn');
        editTranscriptBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const segmentIndex = parseInt(btn.dataset.segmentIndex);
                editTranscriptionSegment(segmentIndex);
            });
        });
        
        // Ajouter les listeners pour les timestamps cliquables
        const transTimeElements = transcriptionResult.querySelectorAll('.clickable-time');
        console.log(`Attachement de ${transTimeElements.length} listeners sur la transcription`);
        transTimeElements.forEach((timeElement, index) => {
            timeElement.addEventListener('click', (e) => {
                console.log(`CLIC détecté sur transcription élément ${index}`);
                const time = parseFloat(timeElement.dataset.time);
                console.log(`Temps extrait: ${time}s`);
                seekAudio(time);
            });
        });
    }
}

// Export functionality
function exportData(format) {
    if (!currentResults) {
        console.error('No results to export');
        return;
    }
    
    const fileName = currentAnalysisFileName.replace(/\.[^/.]+$/, ''); // Remove extension
    
    switch(format) {
        case 'json':
            exportJSON(currentResults, fileName);
            break;
        case 'txt':
            exportTXT(currentResults, fileName);
            break;
        case 'srt':
            exportSRT(currentResults, fileName);
            break;
    }
}

function exportJSON(results, fileName) {
    // Créer une copie avec les données personnalisées
    const exportData = JSON.parse(JSON.stringify(results));
    
    // Ajouter les noms personnalisés
    exportData.speaker_names = speakerNames;
    
    // Mettre à jour les segments de locuteurs avec noms personnalisés
    exportData.speakers = exportData.speakers.map(segment => ({
        ...segment,
        speaker_name: getSpeakerName(segment.speaker_id)
    }));
    
    // Mettre à jour la transcription avec textes édités et noms personnalisés
    if (exportData.transcription) {
        exportData.transcription = exportData.transcription.map((segment, index) => ({
            ...segment,
            text: getTranscriptText(index, segment.text),
            speaker_name: getSpeakerName(segment.speaker_id)
        }));
    }
    
    const content = JSON.stringify(exportData, null, 2);
    downloadFile(content, `${fileName}_analyse.json`, 'application/json');
}

function exportTXT(results, fileName) {
    let content = '='.repeat(60) + '\n';
    content += 'ANALYSE AUDIO - RÉSUMÉ\n';
    content += '='.repeat(60) + '\n\n';
    
    // Information sur la langue
    content += `Langue détectée: ${results.language.language_name}\n`;
    content += `Confiance: ${(results.language.confidence * 100).toFixed(1)}%\n\n`;
    
    // Nombre de locuteurs
    content += `Nombre de locuteurs: ${results.num_speakers}\n\n`;
    
    // Temps de traitement
    content += `Temps de traitement: ${results.processing_time}s\n`;
    content += `Dispositif utilisé: ${results.device === 'cuda' ? 'GPU (CUDA)' : 'CPU'}\n\n`;
    
    // Timeline
    content += '='.repeat(60) + '\n';
    content += 'TIMELINE DES LOCUTEURS\n';
    content += '='.repeat(60) + '\n\n';
    
    results.speakers.forEach((segment, index) => {
        content += `[${formatTime(segment.start_time)} - ${formatTime(segment.end_time)}] `;
        content += `${getSpeakerName(segment.speaker_id)} (durée: ${formatDuration(segment.duration)})\n`;
    });
    
    // Transcription
    if (results.transcription && results.transcription.length > 0) {
        content += '\n' + '='.repeat(60) + '\n';
        content += 'TRANSCRIPTION\n';
        content += '='.repeat(60) + '\n\n';
        
        results.transcription.forEach((segment, index) => {
            content += `[${formatTime(segment.start_time)} - ${formatTime(segment.end_time)}] `;
            content += `${getSpeakerName(segment.speaker_id)}:\n`;
            content += `${getTranscriptText(index, segment.text)}\n\n`;
        });
    }
    
    downloadFile(content, `${fileName}_analyse.txt`, 'text/plain');
}

function exportSRT(results, fileName) {
    if (!results.transcription || results.transcription.length === 0) {
        alert('Aucune transcription disponible pour l\'export SRT');
        return;
    }
    
    let content = '';
    
    results.transcription.forEach((segment, index) => {
        // Numéro de séquence
        content += `${index + 1}\n`;
        
        // Timestamps au format SRT (HH:MM:SS,mmm --> HH:MM:SS,mmm)
        content += `${formatTimeToSRT(segment.start_time)} --> ${formatTimeToSRT(segment.end_time)}\n`;
        
        // Texte avec identification du locuteur
        content += `${getSpeakerName(segment.speaker_id)}: ${getTranscriptText(index, segment.text)}\n\n`;
    });
    
    downloadFile(content, `${fileName}_sous-titres.srt`, 'text/plain');
}

function formatTimeToSRT(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const milliseconds = Math.floor((seconds % 1) * 1000);
    
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
}

function downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Export d'un segment audio
async function exportSegment(segment, format) {
    if (!fileId) {
        alert('Erreur : ID de fichier non disponible');
        return;
    }
    
    try {
        // Construire l'URL de l'API
        const url = `/api/audio/extract-segment?file_id=${fileId}&start_time=${segment.start_time}&end_time=${segment.end_time}&format=${format}`;
        
        // Afficher un message de chargement (optionnel)
        const exportingMsg = document.createElement('div');
        exportingMsg.style.cssText = 'position: fixed; top: 20px; right: 20px; background: var(--primary-color); color: white; padding: 1rem; border-radius: 8px; z-index: 10000; box-shadow: 0 4px 6px rgba(0,0,0,0.3);';
        exportingMsg.textContent = `Export du segment en cours...`;
        document.body.appendChild(exportingMsg);
        
        // Télécharger le fichier
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error('Erreur lors de l\'export du segment');
        }
        
        // Récupérer le blob
        const blob = await response.blob();
        
        // Créer un lien de téléchargement
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `segment_${formatTime(segment.start_time)}-${formatTime(segment.end_time)}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);
        
        // Retirer le message de chargement et afficher succès
        exportingMsg.textContent = '✓ Segment exporté';
        exportingMsg.style.background = '#10b981';
        setTimeout(() => {
            exportingMsg.remove();
        }, 2000);
        
    } catch (error) {
        console.error('Erreur lors de l\'export du segment:', error);
        alert('Erreur lors de l\'export du segment : ' + error.message);
    }
}

// === Fonctions d'édition ===

// Obtenir le nom d'un locuteur (personnalisé ou par défaut)
function getSpeakerName(speakerId) {
    return speakerNames[speakerId] || `Locuteur ${speakerId}`;
}

// Obtenir le texte de transcription (édité ou original)
function getTranscriptText(segmentIndex, originalText) {
    const editedText = transcriptionEdits[segmentIndex];
    if (editedText && editedText !== originalText) {
        return `${editedText} <span class="edited-badge" title="Transcription modifiée">✏️ Édité</span>`;
    }
    return editedText || originalText;
}

// Renommer un locuteur
function renameSpeaker(speakerId) {
    const currentName = getSpeakerName(speakerId);
    const defaultName = `Locuteur ${speakerId}`;
    const isCustom = speakerNames[speakerId] !== undefined;
    
    // Afficher une modal simple avec input
    const modalHtml = `
        <div class="edit-modal" id="speakerEditModal">
            <div class="edit-modal-content">
                <h3>Renommer le locuteur</h3>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Nom actuel : <strong>${currentName}</strong>
                </p>
                <input 
                    type="text" 
                    id="speakerNameInput" 
                    class="edit-input" 
                    placeholder="Nouveau nom (ex: Jean, Marie...)"
                    value="${isCustom ? currentName : ''}"
                    maxlength="50"
                />
                <div class="edit-modal-actions">
                    <button class="btn-secondary" id="cancelSpeakerEdit">Annuler</button>
                    ${isCustom ? '<button class="btn-warning" id="resetSpeakerName">Réinitialiser</button>' : ''}
                    <button class="btn-primary" id="saveSpeakerName">Enregistrer</button>
                </div>
            </div>
        </div>
    `;
    
    // Ajouter au DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = document.getElementById('speakerEditModal');
    const input = document.getElementById('speakerNameInput');
    
    // Focus sur l'input
    setTimeout(() => input.focus(), 100);
    
    // Event listeners
    document.getElementById('cancelSpeakerEdit').addEventListener('click', () => modal.remove());
    
    if (isCustom) {
        document.getElementById('resetSpeakerName').addEventListener('click', () => {
            delete speakerNames[speakerId];
            updateAllSpeakerNames(speakerId);
            modal.remove();
            showToast(`Nom réinitialisé à "${defaultName}"`, 'success');
        });
    }
    
    document.getElementById('saveSpeakerName').addEventListener('click', () => {
        const newName = input.value.trim();
        if (newName && newName !== defaultName) {
            speakerNames[speakerId] = newName;
            updateAllSpeakerNames(speakerId);
            modal.remove();
            showToast(`Locuteur renommé en "${newName}"`, 'success');
        } else if (!newName) {
            input.focus();
            input.style.borderColor = 'var(--danger-color)';
        } else {
            delete speakerNames[speakerId];
            updateAllSpeakerNames(speakerId);
            modal.remove();
        }
    });
    
    // Enter pour sauvegarder, Escape pour annuler
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            document.getElementById('saveSpeakerName').click();
        } else if (e.key === 'Escape') {
            modal.remove();
        }
    });
    
    // Cliquer en dehors pour fermer
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

// Mettre à jour tous les affichages du nom d'un locuteur
function updateAllSpeakerNames(speakerId) {
    const newName = getSpeakerName(speakerId);
    document.querySelectorAll(`.speaker-name[data-speaker-id="${speakerId}"]`).forEach(el => {
        el.textContent = newName;
    });
}

// Éditer un segment de transcription
function editTranscriptionSegment(segmentIndex) {
    if (!currentResults || !currentResults.transcription) return;
    
    const segment = currentResults.transcription[segmentIndex];
    const originalText = segment.text;
    const currentText = getTranscriptText(segmentIndex, originalText);
    
    // Trouver l'élément de transcription
    const transcriptItem = document.querySelector(`.transcription-item[data-segment-index="${segmentIndex}"]`);
    const textElement = transcriptItem.querySelector('.transcript-text');
    
    // Créer le textarea d'édition
    const editHtml = `
        <div class="transcript-edit-container">
            <textarea 
                class="transcript-edit-textarea" 
                id="transcriptEditArea${segmentIndex}"
                rows="3"
            >${currentText}</textarea>
            <div class="transcript-edit-actions">
                <button class="btn-secondary btn-sm" data-action="cancel">Annuler</button>
                <button class="btn-primary btn-sm" data-action="save">Enregistrer</button>
            </div>
        </div>
    `;
    
    textElement.innerHTML = editHtml;
    const textarea = document.getElementById(`transcriptEditArea${segmentIndex}`);
    
    // Auto-resize du textarea
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    });
    
    // Focus
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    
    // Event listeners
    const cancelBtn = textElement.querySelector('[data-action="cancel"]');
    const saveBtn = textElement.querySelector('[data-action="save"]');
    
    cancelBtn.addEventListener('click', () => {
        textElement.innerHTML = currentText;
    });
    
    saveBtn.addEventListener('click', () => {
        const newText = textarea.value.trim();
        if (newText && newText !== originalText) {
            transcriptionEdits[segmentIndex] = newText;
            textElement.innerHTML = newText;
            showToast('Transcription modifiée', 'success');
        } else if (!newText) {
            textarea.focus();
            textarea.style.borderColor = 'var(--danger-color)';
        } else {
            // Le texte est identique à l'original, supprimer l'édition
            delete transcriptionEdits[segmentIndex];
            textElement.innerHTML = originalText;
        }
    });
    
    // Ctrl+Enter pour sauvegarder, Escape pour annuler
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            saveBtn.click();
        } else if (e.key === 'Escape') {
            cancelBtn.click();
        }
    });
}

// Afficher un toast message
function showToast(message, type = 'info') {
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: 'var(--primary-color)'
    };
    
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Navigation audio
function seekAudio(time) {
    if (!audioPlayer || !currentAudioUrl) {
        return;
    }
    
    try {
        // Sauvegarder si l'audio était en cours de lecture
        const wasPlaying = !audioPlayer.paused;
        
        // Pause AVANT de changer la position (évite les problèmes de navigateur)
        if (wasPlaying) {
            audioPlayer.pause();
        }
        
        // Changer la position (plus fiable sur un audio en pause)
        audioPlayer.currentTime = time;
        
        // Vérifier si le changement a pris effet et reprendre la lecture si nécessaire
        setTimeout(() => {
            // Reprendre la lecture si elle était active avant
            if (wasPlaying) {
                audioPlayer.play().catch(err => {
                    console.error('Erreur lors de la reprise de la lecture:', err);
                });
            }
        }, 50);
        
        // Scroller vers le lecteur audio pour le rendre visible
        audioPlayerCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } catch (err) {
        console.error('Erreur lors du changement de position:', err);
    }
}

// Utilitaire
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Gestion du token HuggingFace
async function checkHuggingfaceToken() {
    try {
        const response = await fetch('/api/config/huggingface-token/status');
        const result = await response.json();
        
        if (!result.configured) {
            // Afficher la modal si le token n'est pas configuré
            tokenModal.style.display = 'flex';
        }
    } catch (error) {
        console.error('Erreur lors de la vérification du token:', error);
    }
}

function showTokenError(message) {
    tokenError.textContent = message;
    tokenError.style.display = 'block';
}

// Gestion du mode sombre
function initializeTheme() {
    // Récupérer la préférence sauvegardée ou utiliser la préférence système
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    setTheme(theme);
    
    // Event listener pour le bouton toggle
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Mettre à jour les icônes
    if (theme === 'dark') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}
"""
config.py — Configuration du pipeline YOLO-gen OBB
Modifiez ce fichier selon votre environnement.
"""
 
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
# ── Modèle Hugging Face ────────────────────────────────────────────────────────
HF_MODEL_ID = "magistermilitum/YOLO_manuscripts"
 
# Token HF : laissez None pour utiliser la variable d'environnement HF_TOKEN
# ou le cache de `huggingface-cli login`.
# Sinon, collez votre token ici : HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
HF_TOKEN = os.getenv("HF_TOKEN")
 
# ── Image à analyser ──────────────────────────────────────────────────────────
# Placez l'image dans le même dossier que ce fichier, ou indiquez un chemin absolu.
IMAGE_PATH = "Le_Roman_de_la_Rose.jpeg"
 
# ── Dossier de sortie ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("yologen_output")
 
# ── Seuil de confiance final ──────────────────────────────────────────────────
# Le pipeline testera des seuils décroissants et s'arrêtera au premier
# seuil ≤ CONF_FINAL ayant produit au moins une détection.
CONF_FINAL = 0.10
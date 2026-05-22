# Analyse de mise en page de manuscrits médiévaux avec YOLO-gen

TP — Computer Vision | Détection par boîtes orientées (OBB) sur le *Roman de la Rose*

> **Référence :** Torres Aguilar, S. (2025). *From Codicology to Code: A Comparative Study of Transformer and YOLO-based Detectors for Layout Analysis in Historical Documents*. arXiv:2506.20326  
> **Modèle :** [`magistermilitum/YOLO_manuscripts`](https://huggingface.co/magistermilitum/YOLO_manuscripts) — Hugging Face, licence MIT

---

## Présentation

Ce projet implémente un pipeline Python qui applique **YOLO-gen 11x-OBB** pour la *Document Layout Analysis* (DLA) : localisation et classification des régions structurelles d'un folio médiéval — blocs de texte, enluminures, lettrines, marginalia — avant toute reconnaissance de caractères.

L'image de démonstration est un folio du *Roman de la Rose* (BnF, Français 25526). Le modèle a été entraîné sur trois corpus de manuscrits occidentaux (XII–XVIIe s.) : **e-NDP**, **CATMuS** et **HORAE**.

> Le corpus HORAE est décrit dans : Boillet et al. (2019). *HORAE: an annotated dataset of books of hours*. HIP'19. arXiv:2012.00351

---

## Structure du projet

```
.
├── main.py                        # Pipeline principal (4 fonctions)
├── config.py                      # Constantes : modèle, chemins, seuils, couleurs
├── Le_Roman_de_la_Rose.jpeg       # Image de démonstration (BnF, Fr. 25526)
├── requirement.txt
├── détection_yologen.md           # Tutoriel détaillé du TP
└── yologen_output/                # Générés à l'exécution
    ├── yologen_detections.jpg     # Visualisation annotée
    └── yologen_detections.json    # Export des détections
```

---

## 1. Installation

### 1.1 Dépendances Python

```bash
pip install -r requirement.txt
```

Dépendances principales : `ultralytics`, `huggingface_hub`, `pillow`, `matplotlib`, `numpy`, `python-dotenv`.

### 1.2 Authentification Hugging Face

Le modèle `magistermilitum/YOLO_manuscripts` est hébergé sur Hugging Face et peut nécessiter une authentification.

**Option A — via la CLI (recommandé) :**
```bash
huggingface-cli login
# Coller un token de lecture généré sur huggingface.co/settings/tokens
```

**Option B — via variable d'environnement :**
```bash
export HF_TOKEN=hf_...
```

**Option C — dans `config.py` :**
```python
HF_TOKEN = "hf_..."
```

Le script résout le token dans cet ordre : `config.py` → variable d'environnement → cache `huggingface-cli login`. En cas d'échec (403/401), un message d'aide s'affiche.

---

## 2. Utilisation

```bash
python main.py
```

Le script enchaîne automatiquement les quatre étapes du pipeline :

```
load_model()  →  diagnose_thresholds()  →  visualize()  →  report_and_export()
```

Les sorties sont écrites dans `yologen_output/`.

---

## 3. Architecture du pipeline

### `load_model()`
Télécharge `best.pt` (~118 Mo) depuis Hugging Face via `hf_hub_download()` et retourne un objet `YOLO` prêt à l'inférence.

### `diagnose_thresholds(model, image_path)`
Teste la détection à cinq seuils décroissants (`0.50 → 0.25 → 0.10 → 0.05 → 0.01`) et retourne les résultats au premier seuil ayant produit au moins une détection, dans la limite de `CONF_FINAL`. Affiche un tableau de diagnostic dans le terminal.

Paramètres d'inférence : `iou=0.45`, `imgsz=1280`.

### `visualize(results, img_pil, conf, save_path)`
Dessine les boîtes OBB (polygones orientés à 4 points) sur l'image avec `matplotlib.patches.Polygon` : remplissage semi-transparent (α=0.15) et contour coloré par classe. Sauvegarde la figure annotée en JPEG.

### `report_and_export(results, conf, save_path)`
Affiche un rapport console groupé par classe (effectif, confiance moyenne et maximale) et exporte les détections en JSON.

**Structure JSON :**
```json
{
  "model":          "magistermilitum/YOLO_manuscripts",
  "paper":          "arXiv:2506.20326",
  "conf_threshold": 0.10,
  "regions": [
    { "class": "Text", "confidence": 0.823,
      "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }
  ]
}
```

---

## 4. Pourquoi des boîtes orientées (OBB) ?

Les modèles YOLO classiques produisent des boîtes droites (AABB) alignées sur les axes. Les manuscrits médiévaux contiennent des annotations obliques et des éléments non rectangulaires : les **OBB** sont des quadrilatères librement orientés, décrits par **quatre points** (8 coordonnées) plutôt que deux coins opposés.

Les détections sont accessibles via `result.obb` (et non `result.boxes`). Les coordonnées sont fournies par `box.xyxyxyxy`, un tenseur de forme `(1, 8)` à remodeler en `(4, 2)`.

---

## 5. Configuration

Les constantes sont centralisées dans `config.py` :

| Variable | Valeur par défaut | Description |
|---|---|---|
| `HF_MODEL_ID` | `magistermilitum/YOLO_manuscripts` | Identifiant du modèle sur HF |
| `HF_TOKEN` | `None` | Token d'authentification HF |
| `IMAGE_PATH` | `Le_Roman_de_la_Rose.jpeg` | Image analysée |
| `OUTPUT_DIR` | `yologen_output/` | Dossier de sortie |
| `CONF_FINAL` | `0.10` | Seuil de confiance maximum retenu |
| `CLASS_COLORS` | 8 couleurs hex | Palette par classe |

---

## 6. Pièges courants

**`result.obb` vaut `None` ou est vide.** Toujours tester avant d'itérer. Si le diagnostic affiche `0` à tous les seuils, vérifier que `best.pt` fait bien ~118 Mo (un téléchargement interrompu produit un fichier invalide).

**Erreur 403/401.** Configurer le token HF selon l'une des trois options décrites en §1.2.

---

## Licence

Voir le fichier `LICENSE`. Le modèle `magistermilitum/YOLO_manuscripts` est distribué sous licence MIT.
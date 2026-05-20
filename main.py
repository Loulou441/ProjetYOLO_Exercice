"""
Pipeline d'analyse de mise en page de manuscrits médiévaux
avec YOLO-gen 11x-OBB (boîtes orientées).

Référence : Torres Aguilar, S. (2025). arXiv:2506.20326
Modèle    : magistermilitum/YOLO_manuscripts (Hugging Face, MIT)
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
from config import HF_MODEL_ID, HF_TOKEN, IMAGE_PATH, OUTPUT_DIR, CONF_FINAL

# Palette de couleurs par classe (extensible)
CLASS_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#8338EC", "#FB5607",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. load_model
# ══════════════════════════════════════════════════════════════════════════════

def load_model() -> YOLO:
    """
    Télécharge best.pt depuis Hugging Face et retourne un objet YOLO
    prêt à l'inférence.

    Résolution du token (ordre de priorité) :
      1. Constante HF_TOKEN dans config.py
      2. Variable d'environnement HF_TOKEN
      3. Cache huggingface-cli login
    """
    token = HF_TOKEN or os.environ.get("HF_TOKEN") or None

    print(f"[load_model] Téléchargement de best.pt depuis {HF_MODEL_ID} …")
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="best.pt",
            token=token,
        )
    except Exception as exc:
        if "403" in str(exc) or "401" in str(exc):
            sys.exit(
                "\n[ERREUR] Accès refusé (403/401).\n"
                "Veuillez :\n"
                "  • définir HF_TOKEN dans config.py, ou\n"
                "  • exporter la variable d'environnement HF_TOKEN, ou\n"
                "  • lancer `huggingface-cli login`\n"
            )
        raise

    print(f"[load_model] Modèle chargé depuis : {model_path}")
    return YOLO(model_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2. diagnose_thresholds
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_thresholds(model: YOLO, image_path: str) -> tuple:
    """
    Teste la détection à plusieurs seuils décroissants.
    Retourne (results, conf_retenu) au premier seuil ≤ CONF_FINAL
    ayant produit au moins une détection.

    Seuils testés : 0.50 → 0.25 → 0.10 → 0.05 → 0.01
    Paramètres d'inférence : iou=0.45, imgsz=1280, verbose=False
    """
    thresholds = [0.50, 0.25, 0.10, 0.05, 0.01]

    print("\n[diagnose_thresholds] Tableau de diagnostic")
    print(f"{'Seuil':>8}  {'Détections':>12}  {'Retenu':>8}")
    print("─" * 36)

    chosen_results = None
    chosen_conf = None

    for conf in thresholds:
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=0.45,
            imgsz=1280,
            verbose=False,
        )

        n_detections = 0
        for r in results:
            if r.obb is not None:
                n_detections += len(r.obb)

        retenu = ""
        if chosen_results is None and conf <= CONF_FINAL and n_detections > 0:
            chosen_results = results
            chosen_conf = conf
            retenu = "✓"

        print(f"{conf:>8.2f}  {n_detections:>12}  {retenu:>8}")

    print("─" * 36)

    if chosen_results is None:
        print("[diagnose_thresholds] Aucune détection à aucun seuil ≤ CONF_FINAL.")
        # On retourne quand même les résultats au seuil minimal pour la suite
        results = model.predict(
            source=image_path,
            conf=thresholds[-1],
            iou=0.45,
            imgsz=1280,
            verbose=False,
        )
        return results, thresholds[-1]

    print(f"\n[diagnose_thresholds] Seuil retenu : {chosen_conf}\n")
    return chosen_results, chosen_conf


# ══════════════════════════════════════════════════════════════════════════════
# 3. visualize
# ══════════════════════════════════════════════════════════════════════════════

def visualize(results, img_pil: Image.Image, conf: float, save_path: Path) -> None:
    """
    Dessine les boîtes OBB (polygones orientés) sur l'image et sauvegarde
    la figure au chemin save_path.

    Chaque région est représentée par deux Polygon superposés :
      • rempli  (alpha=0.15)
      • contour (facecolor="none")

    En cas d'absence de détection, affiche un message centré et sauvegarde.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(np.array(img_pil))
    ax.set_axis_off()
    ax.set_title(
        f"YOLO-gen OBB — {HF_MODEL_ID}\nseuil conf = {conf:.2f}",
        fontsize=12,
        pad=10,
    )

    # Collecte de toutes les détections
    all_detections = []
    for result in results:
        if result.obb is None:
            continue
        for box in result.obb:
            cls_id = int(box.cls[0])
            cls_name = model_names.get(cls_id, f"cls_{cls_id}")
            confidence = float(box.conf[0])
            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
            all_detections.append((cls_id, cls_name, confidence, pts))

    if not all_detections:
        ax.text(
            0.5, 0.5,
            "Aucune détection",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=20, color="red",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Figure sauvegardée (sans détection) : {save_path}")
        return

    # Dessin des polygones
    legend_handles = {}
    for cls_id, cls_name, confidence, pts in all_detections:
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # Polygone rempli (semi-transparent)
        poly_fill = mpatches.Polygon(
            pts,
            closed=True,
            linewidth=0,
            facecolor=color,
            alpha=0.15,
        )
        ax.add_patch(poly_fill)

        # Polygone contour
        poly_edge = mpatches.Polygon(
            pts,
            closed=True,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(poly_edge)

        # Étiquette au coin supérieur du polygone (y minimal)
        top_idx = pts[:, 1].argmin()
        tx, ty = pts[top_idx]
        ax.text(
            tx, ty,
            f"{cls_name} {confidence:.2f}",
            fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.85, lw=0),
        )

        # Légende (une entrée par classe unique)
        if cls_name not in legend_handles:
            legend_handles[cls_name] = mpatches.Patch(color=color, label=cls_name)

    ax.legend(
        handles=list(legend_handles.values()),
        loc="upper right",
        fontsize=9,
        framealpha=0.85,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Figure sauvegardée : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. report_and_export
# ══════════════════════════════════════════════════════════════════════════════

def report_and_export(results, conf: float, save_path: Path) -> None:
    """
    Affiche un rapport console groupé par classe
    (effectif, confiance moyenne, maximum) et exporte les détections en JSON.

    Structure JSON :
    {
      "model":          "magistermilitum/YOLO_manuscripts",
      "paper":          "arXiv:2506.20326",
      "conf_threshold": 0.10,
      "regions": [
        { "class": "Text", "confidence": 0.823,
          "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }
      ]
    }
    """
    # Collecte
    regions = []
    per_class = defaultdict(list)

    for result in results:
        if result.obb is None:
            continue
        for box in result.obb:
            cls_id = int(box.cls[0])
            cls_name = model_names.get(cls_id, f"cls_{cls_id}")
            confidence = float(box.conf[0])
            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).tolist()

            regions.append({
                "class": cls_name,
                "confidence": round(confidence, 4),
                "polygon": [[round(x, 2), round(y, 2)] for x, y in pts],
            })
            per_class[cls_name].append(confidence)

    # Rapport console
    print("\n[report_and_export] ─── Rapport de détection ───")
    print(f"{'Classe':<20} {'N':>5} {'Conf moy':>10} {'Conf max':>10}")
    print("─" * 50)
    for cls_name, confs in sorted(per_class.items()):
        print(
            f"{cls_name:<20} {len(confs):>5}"
            f" {np.mean(confs):>10.3f} {max(confs):>10.3f}"
        )
    print("─" * 50)
    print(f"{'TOTAL':<20} {len(regions):>5}\n")

    # Export JSON
    payload = {
        "model": HF_MODEL_ID,
        "paper": "arXiv:2506.20326",
        "conf_threshold": conf,
        "regions": regions,
    }

    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"[report_and_export] JSON sauvegardé : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

# Variable globale pour la palette de noms (remplie après chargement du modèle)
model_names: dict[int, str] = {}


def main():
    global model_names

    OUTPUT_DIR.mkdir(exist_ok=True)

    model = load_model()
    model_names = model.names  # {0: "Text", 1: "Decoration", …}

    results, conf = diagnose_thresholds(model, IMAGE_PATH)

    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    visualize(results, img_pil, conf, OUTPUT_DIR / "yologen_detections.jpg")
    report_and_export(results, conf, OUTPUT_DIR / "yologen_detections.json")


if __name__ == "__main__":
    main()
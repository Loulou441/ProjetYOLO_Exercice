from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

HF_MODEL_ID = "magistermilitum/YOLO_manuscripts"
 
HF_TOKEN = os.getenv("HF_TOKEN")
 
IMAGE_PATH = "Le_Roman_de_la_Rose.jpeg"
 
OUTPUT_DIR = Path("yologen_output")

CONF_FINAL = 0.10

CLASS_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#8338EC", "#FB5607",
]
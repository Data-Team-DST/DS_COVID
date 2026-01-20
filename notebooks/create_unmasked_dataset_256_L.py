from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "raw_dataset_without_masks"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "unmasked_full_dataset_256_256_L"

# Paramètres
TARGET_SIZE = (256, 256)
CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

print(f"Dataset source: {SOURCE_PATH}")
print(f"Dataset de sortie: {OUTPUT_PATH}")
print(f"Taille cible: {TARGET_SIZE}")
print(f"Classes: {CLASSES}")
print(f"\n{'='*60}")

# Statistiques
total_processed = 0
errors = []

# Traiter chaque classe
for class_name in CLASSES:
    print(f"\nTraitement de la classe: {class_name}")
    
    # Chemins source et sortie
    source_dir = SOURCE_PATH / class_name
    output_dir = OUTPUT_PATH / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les images
    image_files = sorted(source_dir.glob("*.png"))
    print(f"Nombre d'images trouvées: {len(image_files)}")
    
    # Traiter chaque image
    for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
        try:
            # 1. Charger et convertir en grayscale
            img = Image.open(img_path).convert('L')
            
            # 2. Redimensionner
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # 3. Sauvegarder
            output_file = output_dir / img_path.name
            img.save(output_file)
            
            total_processed += 1
            
        except Exception as e:
            errors.append((img_path, str(e)))
    
    print(f"✓ {class_name}: {len(list(output_dir.glob('*.png')))} images créées")

print(f"\n{'='*60}")
print(f"RÉSUMÉ")
print(f"{'='*60}")
print(f"Total d'images traitées: {total_processed}")
print(f"Erreurs: {len(errors)}")

if errors:
    print("\nPremières erreurs:")
    for img, err in errors[:5]:
        print(f"  - {img.name}: {err}")

# Vérification finale
print(f"\n{'='*60}")
print("Nombre d'images par classe dans le nouveau dataset:")
for class_name in CLASSES:
    class_dir = OUTPUT_PATH / class_name
    n_images = len(list(class_dir.glob("*.png")))
    print(f"  {class_name}: {n_images}")

print(f"\n✓ Dataset non-masqué 256×256 grayscale créé avec succès dans: {OUTPUT_PATH}")

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from typing import Literal, Tuple
from rich.console import Console
from rich.table import Table

class ImagePreprocessor:
    """Classe pour prétraiter des datasets d'images médicales (COVID radiography)
    
    - source_path: dossier du dataset raw (ex: data/raw/COVID-19_Radiography_Dataset)
    - output_path: dossier de sortie (ex: data/processed/...). La structure finale sera output_path/class_name1/, output_path/class_name2/, etc.
    - target_size: tuple (width, height) pour la résolution finale (ex: (256, 256))
    - image_mode: mode de couleur (1 seul choix: L pour grayscale, adapté aux radiographies)
    - classes: liste des classes à traiter (ex: ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]). Si None, utilise les 4 classes de base.
    - with_masking: booléen pour appliquer les masques si True. Si False, traite les images sans masquage (redimensionnement uniquement).   
    """
    def __init__(
        self,
        source_path: Path, 
        output_path: Path, 
        target_size: Tuple[int, int], 
        image_mode: Literal["L"] = "L", # Grayscale uniquement pour radiographies
        classes: list[str] | None = None, # None signifie utiliser les 4 classes de base
        with_masking: bool = False 
    ):
        self.source_path = Path(source_path) 
        self.output_path = Path(output_path) 
        self.target_size = target_size
        self.image_mode = image_mode
        self.classes = classes or ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
        self.with_masking = with_masking
        self.console = Console()
       
    def process(self, dry_run: bool = False) -> dict:
        """Lance le prétraitement et retourne les statistiques"""
        self.console.print(f"[bold cyan]🚀 Début du prétraitement[/bold cyan]")
        self.console.print(f"Source: {self.source_path}")
        self.console.print(f"Sortie: {self.output_path}")
        self.console.print(f"Taille: {self.target_size}")
        self.console.print(f"Mode: {self.image_mode}")
        if dry_run:
            self.console.print("[yellow]🔍 Mode dry-run: analyse seulement[/yellow]")
       
        stats = {"total_processed": 0, "errors": [], "per_class": {}}
       
        for class_name in self.classes:
            class_stats = self._process_class(class_name, dry_run=dry_run)
            stats["per_class"][class_name] = class_stats
            stats["total_processed"] += class_stats["processed"]
            stats["errors"].extend(class_stats["errors"])
       
        self._print_summary(stats)
        return stats
   
    def _process_class(self, class_name: str, dry_run: bool = False) -> dict:
        """Traite une classe complète"""
        source_dir = self.source_path / class_name 
        output_dir = self.output_path / class_name

        if not source_dir.exists():
            self.console.print(f"[red]❌ Classe {class_name} introuvable[/red]")
            return {"processed": 0, "errors": [], "found": 0}
       

        img_dir = source_dir / "images"
        mask_dir = source_dir / "masks" if self.with_masking else None
       

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(img_dir.glob("*.png"))
       
        self.console.print(f"\n[bold yellow]📁 {class_name}: {len(image_files)} images[/bold yellow]")
       
        class_errors = []
        processed = 0
       
        for img_path in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
            try:
                img = Image.open(img_path).convert(self.image_mode) # L uniquement pour les radiographies

                if self.with_masking and mask_dir is not None:
                    mask_path = mask_dir / img_path.name
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Mask introuvable pour {img_path.name}")
                    mask = Image.open(mask_path).convert(self.image_mode) # L

                    mask_size = (mask.width, mask.height)  # cv2 utilise (width, height)

                    img = img.resize(mask_size, Image.Resampling.LANCZOS)
                    
                    # Convertir en arrays OpenCV
                    img_cv = np.array(img)
                    mask_cv = np.array(mask)
                    
                    # Préparer le mask pour OpenCV (0/255 requis)
                    if mask_cv.max() <= 1:
                        mask_binary = (mask_cv * 255).astype(np.uint8)
                    else:
                        mask_binary = mask_cv.astype(np.uint8)

                    masked_cv = cv2.bitwise_and(img_cv, img_cv, mask=mask_binary)

                    img = Image.fromarray(masked_cv)

                # Toujours redimensionner à la taille cible finale
                if img.size != self.target_size:
                    img = img.resize(self.target_size, Image.Resampling.LANCZOS)

                if not dry_run:
                    output_file = output_dir / img_path.name
                    img.save(output_file)
                processed += 1
            except Exception as e:
                class_errors.append((img_path.name, str(e)))
       
        self.console.print(f"[green]✓ {class_name}: {processed} images OK[/green]")
        return {"processed": processed, "errors": class_errors, "found": len(image_files)}
   
    def _print_summary(self, stats: dict):
        """Affiche un joli résumé avec tableau"""
        table = Table(title="📊 RÉSUMÉ DU PRÉTRAITEMENT")
        table.add_column("Classe", style="cyan")
        table.add_column("Trouvées", justify="right")
        table.add_column("Traitées", justify="right")
        table.add_column("Erreurs", justify="right")
       
        for class_name, data in stats["per_class"].items():
            table.add_row(
                class_name,
                str(data["found"]),
                str(data["processed"]),
                str(len(data["errors"]))
            )
       
        self.console.print(table)
       
        if stats["errors"]:
            self.console.print(f"\n[red]⚠️  {len(stats['errors'])} erreurs totales[/red]")
        else:
            self.console.print("\n[bold green]🎉 TRAITEMENT 100% RÉUSSI ![/bold green]")
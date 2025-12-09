# 📊 Présentation Beamer - Détection COVID-19

## 📄 Description

Présentation Beamer professionnelle pour le projet de détection COVID-19 par Deep Learning. Cette présentation couvre l'ensemble du projet, de la collecte des données jusqu'au déploiement.

## 📦 Fichiers

- `presentation.tex` : Code source LaTeX de la présentation
- `presentation.pdf` : PDF compilé prêt à utiliser (27 pages)

## 🎯 Contenu de la Présentation

### 1. Introduction (3 slides)
- Contexte et problématique
- Objectifs du projet

### 2. Données et Prétraitement (3 slides)
- Dataset et classes
- Pipeline de prétraitement
- Architecture modulaire

### 3. Modèles de Deep Learning (3 slides)
- Architecture des modèles (Transfer Learning)
- Architecture personnalisée
- Entraînement et optimisation

### 4. Résultats et Performance (2 slides)
- Métriques de performance
- Analyse des erreurs

### 5. Interprétabilité (5 slides)
- Importance de l'interprétabilité
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Comparaison des méthodes

### 6. Application et Déploiement (4 slides)
- Architecture de l'application
- Interface utilisateur Streamlit
- Utilisation du package
- Tests et qualité du code

### 7. Conclusion et Perspectives (4 slides)
- Contributions principales
- Limites et défis
- Perspectives futures
- Remerciements

### 8. Annexes (2 slides)
- Détails techniques
- Références

## 🛠️ Compilation

### Prérequis

Installer LaTeX sur votre système :

**Ubuntu/Debian :**
```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-lang-french
```

**macOS :**
```bash
brew install --cask mactex
```

**Windows :**
Télécharger et installer [MiKTeX](https://miktex.org/download) ou [TeX Live](https://www.tug.org/texlive/)

### Compilation de la présentation

```bash
# Compilation simple
pdflatex presentation.tex

# Compilation avec références (recommandé)
pdflatex presentation.tex
pdflatex presentation.tex
```

## ✏️ Personnalisation

### Modifier les informations

Éditez les lignes suivantes dans `presentation.tex` :

```latex
\title[Détection COVID-19]{Détection COVID-19 par Deep Learning}
\subtitle{Application d'analyse d'images radiographiques}
\author{Votre Nom \and Autre Auteur}
\institute{Votre Institution}
\date{\today}
```

### Ajouter votre logo

Remplacez `example-image` par le chemin de votre image :

```latex
\titlegraphic{\includegraphics[width=2cm]{chemin/vers/votre/logo.png}}
```

### Changer le thème et les couleurs

Modifiez les lignes suivantes :

```latex
\usetheme{Madrid}  % Autres: Boadilla, AnnArbor, Berlin, Copenhagen
\usecolortheme{default}  % Autres: crane, beaver, dolphin, orchid
\setbeamercolor{structure}{fg=blue!70!black}  % Couleur principale
```

### Ajouter des images réelles

Pour remplacer les images d'exemple :

1. Ajoutez vos images dans un dossier `images/` à la racine du projet
2. Remplacez `example-image` par le chemin de vos images :

```latex
\includegraphics[width=\textwidth]{images/votre_image.png}
```

### Modifier les graphiques

Les diagrammes sont créés avec TikZ. Pour les personnaliser :

```latex
% Exemple de diagramme de flux
\begin{tikzpicture}[
    box/.style={rectangle, draw, fill=blue!20, text width=2cm},
    arrow/.style={->,>=stealth,thick}
]
    \node[box] (n1) {Étape 1};
    \node[box, right of=n1] (n2) {Étape 2};
    \draw[arrow] (n1) -- (n2);
\end{tikzpicture}
```

## 📝 Conseils pour la Présentation

### Durée recommandée
- **Version complète** : 30-45 minutes (toutes les slides)
- **Version courte** : 15-20 minutes (sections principales uniquement)
- **Version pitch** : 5-10 minutes (intro + résultats + conclusion)

### Slides à adapter selon le public

**Pour un public technique (data scientists, ingénieurs) :**
- Gardez tous les détails techniques
- Insistez sur les méthodes d'interprétabilité
- Détaillez l'architecture des modèles

**Pour un public médical :**
- Simplifiez les aspects techniques
- Insistez sur l'interprétabilité et la validation clinique
- Mettez en avant l'interface utilisateur

**Pour un public général / business :**
- Concentrez-vous sur le problème et l'impact
- Minimisez les détails techniques
- Insistez sur les résultats et le déploiement

### Notes de présentation

Pour ajouter des notes visibles uniquement en mode présentateur :

```latex
\begin{frame}{Titre}
    Contenu visible
    \note{Notes pour le présentateur}
\end{frame}
```

Pour compiler avec les notes :

```bash
pdflatex "\PassOptionsToClass{notes=only}{beamer}\input{presentation.tex}"
```

## 🎨 Thèmes Beamer Recommandés

- **Madrid** (actuel) : Classique et professionnel
- **Boadilla** : Épuré et moderne
- **Copenhagen** : Navigation latérale
- **Berlin** : Sections visibles en en-tête
- **Frankfurt** : Navigation détaillée

## 📚 Ressources

- [Documentation Beamer](https://ctan.org/pkg/beamer)
- [Galerie de thèmes Beamer](https://hartwork.org/beamer-theme-matrix/)
- [TikZ Documentation](https://tikz.dev/)
- [Overleaf - Beamer](https://www.overleaf.com/learn/latex/Beamer)

## 🐛 Dépannage

### Erreur de compilation

Si la compilation échoue :

1. Vérifiez que tous les packages sont installés
2. Supprimez les fichiers auxiliaires :
   ```bash
   rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.snm presentation.toc
   ```
3. Recompilez

### Problèmes d'encodage

Si les caractères accentués ne s'affichent pas correctement :

```latex
\usepackage[utf8]{inputenc}  % Déjà présent
\usepackage[T1]{fontenc}     % Déjà présent
\usepackage[french]{babel}   % Déjà présent
```

### Images manquantes

Si les images ne s'affichent pas :

1. Vérifiez le chemin des images
2. Utilisez `example-image` pour tester (fourni par LaTeX)
3. Formats supportés : PNG, JPG, PDF

## 📞 Support

Pour toute question ou problème :

- **Repository GitHub** : https://github.com/Data-Team-DST/DS_COVID
- **Issues** : Ouvrez un ticket sur GitHub
- **Contacts** : Voir la dernière slide de la présentation

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

**Dernière mise à jour** : Décembre 2024

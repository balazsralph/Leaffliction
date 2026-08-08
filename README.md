# Leaffliction

Classification d'images par reconnaissance de maladies sur les feuilles (projet 42 — Computer Vision).

## Partie 1 : Analyse du dataset (`Distribution.py`)

Cette partie fournit un programme qui explore un dataset d'images de feuilles et en dresse l'état des lieux, classe par classe, pour une espèce de plante donnée.

**Utilisation :**

```bash
./Distribution.py <répertoire>
```

Exemple :

```bash
./Distribution.py DATA/Apple
```

**Fonctionnement :**

- Le programme prend en argument un répertoire (ex. `DATA/Apple`) et parcourt ses sous-répertoires directs.
- Chaque sous-répertoire est interprété comme une classe nommée `<Plante>_<État>` (ex. `Apple_Black_rot`, `Apple_healthy`, `Apple_rust`, `Apple_scab`), d'où sont extraits le nom de la plante et le nom de la classe.
- Pour chaque plante détectée, il compte le nombre d'images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`) présentes dans chacune de ses classes.
- Il affiche dans le terminal, pour chaque plante, le total d'images ainsi que la répartition par classe.
- Il génère ensuite, pour chaque plante, une figure combinant un **pie chart** (répartition en pourcentage) et un **bar chart** (nombre d'images par classe), avec les noms de classes tirés directement des noms de répertoires.

**Objectif :** vérifier visuellement l'équilibre (ou le déséquilibre) du dataset entre les différentes classes de maladies avant de passer aux étapes suivantes (augmentation de données, transformation d'images, classification).

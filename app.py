"""
Application Streamlit pour les Travaux Pratiques de Traitement d'Images
Auteur: NGUETCHO BIADOU CHLOE GABRIELLE
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================

st.set_page_config(
    page_title="TP Traitement d'Images - NGUETCHO Gabrielle",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# STYLES CSS PERSONNALISÉS (RESPONSIVE & THEME-AWARE)
# ==============================================================================

st.markdown("""
<style>
    /* Style général */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #3B8ED0;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .status-success {
        background-color: rgba(74, 222, 128, 0.1);
        border-left: 4px solid #4ade80;
        color: #4ade80;
    }
    
    .status-warning {
        background-color: rgba(229, 184, 0, 0.1);
        border-left: 4px solid #e5b800;
        color: #e5b800;
    }
    
    .status-error {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        color: #ef4444;
    }
    
    .info-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #3B8ED0;
    }
    
    .theory-content {
        line-height: 1.8;
        font-size: 0.95rem;
    }
    
    /* Boutons personnalisés */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        padding-top: 2rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 1rem;
        }
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #3B8ED0, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BASE DE DONNÉES PÉDAGOGIQUES
# ==============================================================================

class CourseContent:
    """Stocke le contenu théorique issu des cours et les templates de code."""
    
    @staticmethod
    def get_info(tp_id):
        infos = {
            "TP1": (
                "TP1 : Bases du Numérique & Matrice de Pixels",
                """**OBJECTIFS PÉDAGOGIQUES :**
Comprendre qu'une image numérique est une fonction discrète f(x,y) représentée par une matrice. Ce TP manipule la résolution spatiale (échantillonnage) et la résolution colorimétrique (quantification).

**PRINCIPES THÉORIQUES :**
1. **Echantillonnage :** Le passage d'une scène continue à une grille discrète. Réduire la fréquence d'échantillonnage crée une pixellisation (Aliasing).
2. **Quantification :** Discrétisation de l'amplitude du signal. Passer de 256 niveaux (8 bits) à K niveaux réduit la qualité visuelle (phénomène de faux contours).
3. **Profil :** Analyse 1D d'une ligne de l'image (f(x) à y fixé).

**💻 IMPLÉMENTATION PYTHON :**
• Chargement : `cv2.imread(path, 0)` charge l'image en matrice Numpy uint8 (0-255).
• Slicing (Sous-échantillonnage) : `img[::k, ::k]` utilise la syntaxe Numpy pour ne garder qu'un pixel tous les k pixels.
• Quantification : Utilisation de la division entière `(img // diviseur) * diviseur` pour forcer les valeurs des pixels à des paliers fixes.
• Accès Pixel : `val = img[y, x]` montre l'accès direct aux coordonnées matricielles."""
            ),
            
            "TP2": (
                "TP2 : Amélioration par Histogrammes (Luminance & Contraste)",
                """**OBJECTIFS PÉDAGOGIQUES :**
Analyser la distribution statistique des niveaux de gris pour corriger les défauts d'acquisition (sous-exposition, faible contraste).

**PRINCIPES THÉORIQUES :**
1. **Histogramme H(k) :** Compte le nombre d'occurrences de chaque niveau de gris k.
2. **Étirement Linéaire (Stretching) :** Transformation affine g(x) = a*f(x) + b pour étaler la dynamique sur la plage complète [0, 255].
3. **Égalisation :** Transformation non-linéaire qui cherche à rendre l'histogramme 'plat' (uniforme). Elle utilise la Fonction de Répartition Cumulée (CDF) comme fonction de transfert.

**💻 IMPLÉMENTATION PYTHON :**
• Calcul : `plt.hist` ou `np.histogram` génèrent les données statistiques.
• Étirement : `((img - min) / (max - min)) * 255` normalise les pixels (formule min-max).
• Égalisation : `cv2.equalizeHist(img)` applique l'algorithme complet."""
            ),
            
            "TP3": (
                "TP3 : Filtrage Spatial (Lissage & Débruitage)",
                """**OBJECTIFS PÉDAGOGIQUES :**
Modifier l'image par des opérations de voisinage (masques locaux). Illustration des concepts de convolution et de filtrage linéaire vs non-linéaire.

**PRINCIPES THÉORIQUES :**
1. **Convolution (g = f * h) :** Un noyau h(x,y) glisse sur l'image. Chaque pixel est la somme pondérée de ses voisins.
2. **Filtre Moyenneur (Passe-bas) :** Chaque poids = 1/N. Lisse mais floute les contours.
3. **Filtre Gaussien :** Poids en forme de cloche (poids fort au centre). Lisse en préservant mieux la structure.
4. **Filtre Médian (Non-linéaire) :** Remplace le pixel par la valeur médiane du voisinage. Redoutable contre le bruit impulsionnel 'Poivre et Sel'.

**💻 IMPLÉMENTATION PYTHON :**
• Moyenne : `cv2.blur(img, (k,k))` ou convolution manuelle `cv2.filter2D`.
• Gaussien : `cv2.GaussianBlur(img, (k,k), sigma)` génère le noyau gaussien automatiquement.
• Médian : `cv2.medianBlur(img, k)` trie les pixels du voisinage et prend le centre."""
            ),
            
            "TP4": (
                "TP4 : Domaine Fréquentiel (Transformée de Fourier)",
                """**OBJECTIFS PÉDAGOGIQUES :**
Passer du domaine spatial (pixels x,y) au domaine fréquentiel (fréquences u,v). Analyser l'image comme un signal (somme de sinusoïdes).

**PRINCIPES THÉORIQUES :**
1. **FFT (Fast Fourier Transform) :** Décompose l'image. Le centre du spectre contient les basses fréquences (énergie, formes globales). La périphérie contient les hautes fréquences (contours, bruit).
2. **Filtrage Idéal :** Multiplication du spectre par un masque (Cercle blanc ou noir).
3. **Propriété de Rotation :** La rotation spatiale entraine une rotation spectrale.

**💻 IMPLÉMENTATION PYTHON :**
• Analyse : `np.fft.fft2(img)` passe en complexe. `np.fft.fftshift` recentre le zéro (DC) au milieu.
• Spectre Magnitude : `20*np.log(np.abs(fshift))` permet de visualiser le spectre.
• Filtrage : On multiplie directement la matrice complexe par un masque avant d'appliquer la `ifft2`."""
            ),
            
            "TP5": (
                "TP5 : Morphologie Mathématique",
                """**OBJECTIFS PÉDAGOGIQUES :**
Analyse de formes non-linéaire basée sur la théorie des ensembles. Essentiel pour nettoyer des masques binaires après seuillage.

**PRINCIPES THÉORIQUES :**
1. **Élément Structurant (SE) :** La forme 'sonde' (carré, disque) qui analyse l'objet.
2. **Érosion (MIN) :** 'Ronge' les objets. Supprime le bruit blanc isolé.
3. **Dilatation (MAX) :** Épaissit les objets. Comble les trous noirs.
4. **Ouverture (Erosion -> Dilatation) :** Supprime les petits objets sans changer la taille des gros.
5. **Fermeture (Dilatation -> Erosion) :** Bouche les trous à l'intérieur des objets.

**💻 IMPLÉMENTATION PYTHON :**
• Noyau : `cv2.getStructuringElement(cv2.MORPH_RECT, size)`.
• Opérations de base : `cv2.erode` et `cv2.dilate`.
• Opérations avancées : `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)`."""
            ),
            
            "TP6": (
                "TP6 : Segmentation & Clustering",
                """**OBJECTIFS PÉDAGOGIQUES :**
Partitionner l'image en régions homogènes (sens). Approches 'Pixel' vs 'Région'.

**PRINCIPES THÉORIQUES :**
1. **Seuillage (Thresholding) :** Sépare fond/forme. La méthode d'Otsu calcule automatiquement le seuil optimal qui minimise la variance intra-classe.
2. **Croissance de région :** Part d'un 'germe' (seed) et agrège les pixels voisins similaires.
3. **K-Means (Clustering) :** Algorithme non-supervisé. Il regroupe les vecteurs (R,G,B) des pixels en K groupes autour de centres de gravité (centroïdes).

**💻 IMPLÉMENTATION PYTHON :**
• Otsu : `cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)`.
• K-Means : On reformate l'image en liste de pixels via `reshape((-1,3))`. La fonction `cv2.kmeans` retourne les étiquettes de chaque pixel."""
            ),
            
            "TP7": (
                "TP7 : Espaces Colorimétriques (RGB vs HSV)",
                """**OBJECTIFS PÉDAGOGIQUES :**
Dépasser le modèle RGB qui est fortement corrélé pour une segmentation couleur robuste.

**PRINCIPES THÉORIQUES :**
1. **RGB (Red Green Blue) :** Modèle additif technique (capteurs caméras). Difficile de séparer la couleur de son intensité.
2. **HSV (Hue Saturation Value) :** Modèle perceptuel.
   - Hue (Teinte) : La 'couleur' pure (angle sur le cercle chromatique).
   - Saturation : Pureté de la couleur (vivacité).
   - Value : Luminosité (clair ou sombre).

**💻 IMPLÉMENTATION PYTHON :**
• Conversion : `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`.
• Masquage : `cv2.inRange(hsv_img, bornes_inf, bornes_sup)` crée un masque binaire.
• Extraction : `cv2.bitwise_and(src, src, mask=mask)` applique le masque."""
            ),
            
            "TP8": (
                "TP8 : Analyse de Texture (Statistique & Fréquentielle)",
                """**OBJECTIFS PÉDAGOGIQUES :**
Définir une région non par sa couleur moyenne, mais par son arrangement spatial (rugueux, ligné, pointillé).

**PRINCIPES THÉORIQUES :**
1. **GLCM (Matrice de Co-occurrence) :** Analyse statistique du second ordre. Compte combien de fois un niveau de gris `i` est voisin d'un niveau `j`. On en déduit des descripteurs : Contraste, Homogénéité, Énergie.
2. **Filtres de Gabor :** Outil puissant simulant le système visuel humain. C'est une sinusoïde modulée par une gaussienne. Détecte des textures orientées à fréquences spécifiques.

**💻 IMPLÉMENTATION PYTHON :**
• GLCM : Librairie `skimage.feature.graycomatrix` pour calculer la matrice, puis `graycoprops` pour extraire le contraste.
• Gabor : `cv2.getGaborKernel(taille, sigma, theta, lambda, ...)` crée le noyau de convolution complexe."""
            )
        }
        return infos.get(tp_id, ("Information Inconnue", "Pas de détails disponibles pour ce TP."))
    
    @staticmethod
    def get_source_code(tp_ex_id, lang="python"):
        """Génère le code source complet (Python ou Matlab)"""
        
        if lang == "python":
            python_codes = {
                "tp1_ex2": """# TP1 - Exercice 2 : Sous-échantillonnage
import cv2
import matplotlib.pyplot as plt

# 1. Charger l'image en niveaux de gris
img = cv2.imread('image_source.jpg', 0)

if img is None:
    print("Erreur: Image non trouvée.")
    exit()

# 2. Définir les facteurs de sous-échantillonnage k
k_values = [1, 2, 4, 8, 16]

plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values):
    # Manipulation matricielle avec Numpy Slicing
    sous_ech = img[::k, ::k]
    
    plt.subplot(1, 5, i+1)
    plt.imshow(sous_ech, cmap='gray')
    plt.title(f"1 pixel sur {k}")
    plt.axis('off')

plt.tight_layout()
plt.show()""",

                "tp1_ex4": """# TP1 - Exercice 4 : Profil d'Intensité
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg', 0)
h, w = img.shape

# Choix d'une ligne au milieu de l'image
ligne_idx = h // 2

# Extraction de la ligne (vecteur 1D)
profil = img[ligne_idx, :]

# Visualisation
plt.figure(figsize=(10, 8))

# 1. Image avec ligne rouge
plt.subplot(2,1,1)
plt.imshow(img, cmap='gray')
plt.axhline(ligne_idx, color='r', linewidth=2)
plt.title(f"Ligne sélectionnée (y={ligne_idx})")
plt.axis('off')

# 2. Graphique d'intensité f(x)
plt.subplot(2,1,2)
plt.plot(profil, color='black', linewidth=1)
plt.grid(True, alpha=0.3)
plt.title("Amplitude du signal le long de la ligne")
plt.xlabel("Position x (pixels)")
plt.ylabel("Intensité (0-255)")
plt.ylim(0, 260)
plt.show()""",

                "tp2_ex3": """# TP2 - Exercice 3 : Égalisation d'Histogramme
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger en gris
img = cv2.imread('image_source.jpg', 0)

# Égalisation d'histogramme
img_eq = cv2.equalizeHist(img)

# Affichage comparatif
plt.figure(figsize=(12, 6))

# Original
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Originale")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(img.flatten(), 256, [0,256], color='r')
plt.title("Histo Original")
plt.xlim([0,256])

# Égalisé
plt.subplot(2,2,3)
plt.imshow(img_eq, cmap='gray')
plt.title("Égalisée (Contraste Maximisé)")
plt.axis('off')

plt.subplot(2,2,4)
plt.hist(img_eq.flatten(), 256, [0,256], color='b')
plt.title("Histo Égalisé")
plt.xlim([0,256])

plt.tight_layout()
plt.show()""",

                "tp3_ex2": """# TP3 - Exercice 2 : Comparaison Moyenneur vs Médian
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image, ratio=0.05):
    out = image.copy()
    num_salt = np.ceil(ratio * image.size * 0.5).astype(int)
    num_pepper = np.ceil(ratio * image.size * 0.5).astype(int)
    
    # Sel (Blanc)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    out[tuple(coords)] = 255
    # Poivre (Noir)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    out[tuple(coords)] = 0
    return out

img = cv2.imread('image_source.jpg', 0)
img_noisy = add_salt_pepper_noise(img, ratio=0.05)

# 1. Filtre Moyenneur
blur = cv2.blur(img_noisy, (5, 5))

# 2. Filtre Médian
median = cv2.medianBlur(img_noisy, 5)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img_noisy, cmap='gray')
plt.title("Image Bruitée")

plt.subplot(132)
plt.imshow(blur, cmap='gray')
plt.title("Moyenneur 5x5 (Echec)")

plt.subplot(133)
plt.imshow(median, cmap='gray')
plt.title("Médian 5x5 (Réussite)")

plt.show()""",

                "tp4_ex1": """# TP4 - Exercice 1 : Spectre de Fourier 2D (FFT)
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg', 0)

# 1. Transformation de Fourier Discrète 2D
f = np.fft.fft2(img)

# 2. Shift (Centrage)
fshift = np.fft.fftshift(f)

# 3. Magnitude (Spectre d'amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Domaine Spatial')
plt.axis('off')

plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Spectre Fréquentiel (Log Magnitude)')
plt.axis('off')

plt.show()""",

                "tp5_ex3": """# TP5 - Exercice 3 : Gradient Morphologique
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg', 0)

# Binarisation (Otsu)
_, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Élément Structurant
kernel = np.ones((3,3), np.uint8)

# 1. Dilatation
dilated = cv2.dilate(bin_img, kernel, iterations=1)

# 2. Erosion
eroded = cv2.erode(bin_img, kernel, iterations=1)

# 3. Gradient = Dilatation - Erosion
gradient = cv2.subtract(dilated, eroded)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(bin_img, cmap='gray')
plt.title("Binaire")

plt.subplot(132)
plt.imshow(dilated, cmap='gray')
plt.title("Dilatée")

plt.subplot(133)
plt.imshow(gradient, cmap='gray')
plt.title("Gradient (Contours)")

plt.show()""",

                "tp6_ex1": """# TP6 - Exercice 1 : Segmentation Couleur par K-Means
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger en couleur (RGB)
img = cv2.imread('image_source.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Restructuration des données
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# 3. Critères d'arrêt
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 4. Application K-Means avec K=4 Clusters
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 5. Reconstruction de l'image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img)
plt.title("Original RGB")

plt.subplot(122)
plt.imshow(res2)
plt.title(f"K-Means (K={K} couleurs)")

plt.show()""",

                "tp7_ex2": """# TP7 - Exercice 2 : Segmentation dans l'espace HSV
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Conversion vers HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Définition de la plage de couleur (exemple: vert/jaune)
lower_val = np.array([20, 50, 50])
upper_val = np.array([40, 255, 255])

# Création du masque binaire
mask = cv2.inRange(hsv, lower_val, upper_val)

# Application du masque
res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(img_rgb)
plt.title("Image Originale")

plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.title("Masque (Seuillage Hue)")

plt.subplot(133)
plt.imshow(res)
plt.title("Extraction Objet")

plt.show()""",

                "tp8_ex2": """# TP8 - Exercice 2 : Analyse de Texture (GLCM & Gabor)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

img = cv2.imread('image_source.jpg', 0)

# --- 1. Approche Statistique : GLCM ---
glcm = graycomatrix(img, distances=[1], angles=[0], levels=256,
                    symmetric=True, normed=True)

# Extraction des descripteurs
contrast = graycoprops(glcm, 'contrast')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

print(f"Descripteurs de Texture :")
print(f" - Contraste : {contrast:.2f}")
print(f" - Homogénéité: {homogeneity:.2f}")
print(f" - Energie : {energy:.4f}")

# --- 2. Approche Fréquentielle : Filtre de Gabor ---
ksize = 21
sigma = 5.0
theta = np.pi / 4
lambd = 10.0
gamma = 0.5

g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Texture Origine")

plt.subplot(122)
plt.imshow(filtered_img, cmap='gray')
plt.title(f"Réponse Gabor (45 deg)")

plt.show()"""
            }
            return python_codes.get(tp_ex_id, f"# Code Python pour {tp_ex_id} non trouvé")
        
        elif lang == "matlab":
            matlab_codes = {
                "tp1_ex2": """% TP1 - Ex2 : Sous-Echantillonnage Spatial
clc; clear; close all;

% 1. Lire l'image en niveaux de gris
I = imread('image_source.jpg');
if size(I,3)==3
    I = rgb2gray(I);
end

% 2. Facteur k (step)
K_values = [1, 2, 4, 8];

figure('Name', 'Sous-échantillonnage');
for i = 1:length(K_values)
    k = K_values(i);
    I_sub = I(1:k:end, 1:k:end);
    
    subplot(1, 4, i);
    imshow(I_sub);
    title(['1 pixel sur ', num2str(k)]);
end""",

                "tp1_ex4": """% TP1 - Ex4 : Profil d'Intensité (Ligne)
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

[h, w] = size(I);
num_ligne = floor(h/2);

% Extraction vecteur ligne
profil = I(num_ligne, :);

figure;
subplot(2,1,1); imshow(I); hold on;
line([1 w], [num_ligne num_ligne], 'Color', 'r', 'LineWidth', 2);
title('Image');

subplot(2,1,2); plot(profil, 'b'); grid on;
title(['Profil intensité ligne ', num2str(num_ligne)]);
xlim([1 w]); ylim([0 255]);
xlabel('Position x'); ylabel('Niveau gris');""",

                "tp2_ex3": """% TP2 - Ex3 : Égalisation d'histogramme
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

% Égalisation
I_eq = histeq(I);

figure;
subplot(2,2,1); imshow(I); title('Originale');
subplot(2,2,2); imhist(I); title('Histo Orig');

subplot(2,2,3); imshow(I_eq); title('Egalisée');
subplot(2,2,4); imhist(I_eq); title('Histo Egalisé');""",

                "tp3_ex2": r"""import cv2
import matplotlib.pyplot as plt
img = cv2.imread('image_source.jpg', 0)
blur = cv2.blur(img, (5, 5))
median = cv2.medianBlur(img, 5)
plt.subplot(131); plt.imshow(img)
plt.subplot(132); plt.imshow(blur)
plt.subplot(133); plt.imshow(median)
plt.show()""",
                "tp4_ex1": r"""import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image_source.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
ms = 20 * np.log(np.abs(fshift) + 1)
plt.imshow(ms, cmap='gray'); plt.show()""",
                "tp5_ex3": r"""import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image_source.jpg', 0)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3))
grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)
plt.imshow(grad, cmap='gray'); plt.show()""",
                "tp6_ex1": r"""import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image_source.jpg')
Z = img.reshape((-1,3)); Z = np.float32(Z)
ret,label,center=cv2.kmeans(Z,4,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0),10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()].reshape((img.shape))
plt.imshow(res); plt.show()""",
                "tp7_ex2": r"""import cv2
import numpy as np
img = cv2.imread('image_source.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([20,50,50]), np.array([40,255,255]))
res = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(res); plt.show()""",
                "tp8_ex2": r"""import cv2
from skimage.feature import graycomatrix, graycoprops
img = cv2.imread('image_source.jpg', 0)
glcm = graycomatrix(img, [1], [0], levels=256)
print(graycoprops(glcm, 'contrast'))"""
            }
            default = f"# Code complet Python pour {tp_ex_id} dans l'app."
            return python_codes.get(tp_ex_id, default)
        
        # --- MATLAB ---
        elif lang == "matlab":
            # Je mets des placeholders simplifiés pour gagner de la place ici,
            # le moteur principal reprend la logique de l'original
            return f"% Code Matlab pour {tp_ex_id} généré automatiquement.\nI = imread('image.jpg');\nimshow(I);"
        return "Erreur langue"

# ==============================================================================
# MOTEUR DE TRAITEMENT
# ==============================================================================

class ProcessingEngine:
    def load_image(self, uploaded_file, grayscale=False):
        if uploaded_file is None:
            return None
        
        # Convertir l'uploaded file en array numpy
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        if grayscale:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def process(self, tp, ex, uploaded_file):
        """Dispatcheur central"""
        try:
            func = getattr(self, f"algo_{tp}_{ex}")
            return func(uploaded_file)
        except AttributeError:
            return None, "Algorithme non implémenté."
        except Exception as e:
            return None, str(e)
    
    # --- TP1 : Bases ---
    def algo_1_ex2(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a1a')
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original', color='white', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(img[::2, ::2], cmap='gray')
        axes[1].set_title('1/2 (Sous-échantillonné)', color='white', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(img[::8, ::8], cmap='gray')
        axes[2].set_title('1/8 (Pixellisé)', color='white', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig, "Sous-échantillonnage appliqué avec succès"
    
    def algo_1_ex4(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        h, w = img.shape
        mid = h // 2
        line = img[mid, :]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax1.imshow(img, cmap='gray')
        ax1.axhline(mid, color='red', linewidth=2)
        ax1.set_title(f'Image avec ligne sélectionnée (y={mid})', color='white', fontsize=12)
        ax1.axis('off')
        
        ax2.plot(line, color='cyan', linewidth=1.5)
        ax2.set_facecolor('#2b2b2b')
        ax2.set_title('Profil d\'intensité', color='white', fontsize=12)
        ax2.set_xlabel('Position x (pixels)', color='white')
        ax2.set_ylabel('Intensité (0-255)', color='white')
        ax2.tick_params(colors='white')
        ax2.grid(alpha=0.3, color='gray')
        ax2.set_ylim([0, 260])
        
        plt.tight_layout()
        return fig, f"Profil d'intensité calculé pour la ligne {mid}"
    
    # --- TP2 : Histogrammes ---
    def algo_2_ex3(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        eq = cv2.equalizeHist(img)
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0,0].imshow(img, cmap='gray')
        ax[0,0].set_title('Image Originale', color='white', fontsize=12)
        ax[0,0].axis('off')
        
        ax[0,1].hist(img.ravel(), 256, [0,256], color='#ef4444', alpha=0.7)
        ax[0,1].set_facecolor('#2b2b2b')
        ax[0,1].set_title('Histogramme Original', color='white', fontsize=12)
        ax[0,1].tick_params(colors='white')
        ax[0,1].set_xlim([0, 256])
        
        ax[1,0].imshow(eq, cmap='gray')
        ax[1,0].set_title('Image Égalisée', color='white', fontsize=12)
        ax[1,0].axis('off')
        
        ax[1,1].hist(eq.ravel(), 256, [0,256], color='#3B8ED0', alpha=0.7)
        ax[1,1].set_facecolor('#2b2b2b')
        ax[1,1].set_title('Histogramme Égalisé', color='white', fontsize=12)
        ax[1,1].tick_params(colors='white')
        ax[1,1].set_xlim([0, 256])
        
        plt.tight_layout()
        return fig, "Égalisation d'histogramme réalisée avec succès"
    
    # --- TP3 : Filtrage ---
    def algo_3_ex2(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        # Ajout de bruit poivre et sel
        noise = img.copy()
        mask = np.random.randint(0, 100, img.shape)
        noise[mask < 2] = 0
        noise[mask > 98] = 255
        
        med = cv2.medianBlur(noise, 5)
        avg = cv2.blur(noise, (5,5))
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0].imshow(noise, cmap='gray')
        ax[0].set_title('Image Bruitée (Poivre & Sel)', color='white', fontsize=12)
        ax[0].axis('off')
        
        ax[1].imshow(avg, cmap='gray')
        ax[1].set_title('Filtre Moyenne 5x5 (Flou)', color='white', fontsize=12)
        ax[1].axis('off')
        
        ax[2].imshow(med, cmap='gray')
        ax[2].set_title('Filtre Médian 5x5 (Net)', color='white', fontsize=12)
        ax[2].axis('off')
        
        plt.tight_layout()
        return fig, "Comparaison des filtres de débruitage"
    
    # --- TP4 : Fourier ---
    def algo_4_ex1(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Domaine Spatial', color='white', fontsize=12)
        ax[0].axis('off')
        
        ax[1].imshow(magnitude, cmap='gray')
        ax[1].set_title('Spectre Fréquentiel (FFT)', color='white', fontsize=12)
        ax[1].axis('off')
        
        plt.tight_layout()
        return fig, "Transformée de Fourier 2D calculée"
    
    # --- TP5 : Morphologie ---
    def algo_5_ex3(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        
        erosion = cv2.erode(bw, kernel, iterations=1)
        dilatation = cv2.dilate(bw, kernel, iterations=1)
        gradient = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0,0].imshow(bw, cmap='gray')
        ax[0,0].set_title('Image Binaire (Otsu)', color='white', fontsize=12)
        ax[0,0].axis('off')
        
        ax[0,1].imshow(gradient, cmap='gray')
        ax[0,1].set_title('Gradient Morphologique', color='white', fontsize=12)
        ax[0,1].axis('off')
        
        ax[1,0].imshow(erosion, cmap='gray')
        ax[1,0].set_title('Érosion', color='white', fontsize=12)
        ax[1,0].axis('off')
        
        ax[1,1].imshow(dilatation, cmap='gray')
        ax[1,1].set_title('Dilatation', color='white', fontsize=12)
        ax[1,1].axis('off')
        
        plt.tight_layout()
        return fig, "Opérations morphologiques appliquées"
    
    # --- TP6 : Segmentation ---
    def algo_6_ex1(self, uploaded_file):
        img = self.load_image(uploaded_file)
        if img is None:
            return None, "Erreur de chargement"
        
        pixel_vals = img.reshape((-1, 3))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 4
        _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        res2 = res.reshape((img.shape))
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0].imshow(img)
        ax[0].set_title('Image Originale', color='white', fontsize=12)
        ax[0].axis('off')
        
        ax[1].imshow(res2)
        ax[1].set_title(f'Segmentation K-Means (K={k})', color='white', fontsize=12)
        ax[1].axis('off')
        
        plt.tight_layout()
        return fig, "Segmentation par K-Means réalisée"
    
    # --- TP7 : Couleur HSV ---
    def algo_7_ex2(self, uploaded_file):
        img = self.load_image(uploaded_file)
        if img is None:
            return None, "Erreur de chargement"
        
        # Conversion BGR -> RGB déjà faite dans load_image
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Détection rouge/orange
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = mask1 | mask2
        seg = cv2.bitwise_and(img, img, mask=mask)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0].imshow(img)
        ax[0].set_title('Image RGB Originale', color='white', fontsize=12)
        ax[0].axis('off')
        
        ax[1].imshow(hsv[:,:,0], cmap='hsv')
        ax[1].set_title('Canal Teinte (H)', color='white', fontsize=12)
        ax[1].axis('off')
        
        ax[2].imshow(seg)
        ax[2].set_title('Segmentation par Couleur', color='white', fontsize=12)
        ax[2].axis('off')
        
        plt.tight_layout()
        return fig, "Segmentation dans l'espace HSV"
    
    # --- TP8 : Texture ---
    def algo_8_ex2(self, uploaded_file):
        img = self.load_image(uploaded_file, True)
        if img is None:
            return None, "Erreur de chargement"
        
        # GLCM
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, 
                           symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Filtre Gabor
        ksize = 31
        sigma = 4.0
        theta = np.pi / 4
        lambd = 10.0
        gamma = 0.5
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        f_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title(f'GLCM\nContraste: {contrast:.2f}, Homogénéité: {homogeneity:.2f}', 
                       color='white', fontsize=11)
        ax[0].axis('off')
        
        ax[1].imshow(f_img, cmap='gray')
        ax[1].set_title('Réponse Filtre Gabor (45°)', color='white', fontsize=12)
        ax[1].axis('off')
        
        plt.tight_layout()
        return fig, f"Analyse de texture (Contraste: {contrast:.2f})"


# ==============================================================================
# INITIALISATION SESSION STATE
# ==============================================================================

if 'processor' not in st.session_state:
    st.session_state.processor = ProcessingEngine()

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'current_result' not in st.session_state:
    st.session_state.current_result = None

if 'current_message' not in st.session_state:
    st.session_state.current_message = None

# ==============================================================================
# INTERFACE PRINCIPALE
# ==============================================================================

# --- HEADER CENTRAL ---
st.markdown("""
<div style="text-align: center; padding: 0rem 1rem 1rem 1rem;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        NGUETCHO BIADOU CHLOE GABRIELLE
    </h1>
    <p style="font-size: 1.1rem; color: #64748b; font-weight: 500;">
        Travaux Pratiques de Traitement d'Images
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# --- SIDEBAR COMPACTE ---
with st.sidebar:
    # Upload moderne et épuré
    st.markdown("""
    <div style="padding: 1rem 0;">
        <p style="font-size: 0.9rem; font-weight: 600; color: #64748b; margin-bottom: 0.5rem;">
            📂 IMAGE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Charger",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.markdown(
            f'<div style="padding: 0.5rem; border-radius: 6px; background-color: rgba(74, 222, 128, 0.1); border-left: 3px solid #4ade80; font-size: 0.85rem; margin: 0.5rem 0;">✓ {uploaded_file.name[:20]}...</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="padding: 0.5rem; border-radius: 6px; background-color: rgba(251, 191, 36, 0.1); border-left: 3px solid #fbbf24; font-size: 0.85rem; margin: 0.5rem 0;">Aucune image</div>',
            unsafe_allow_html=True
        )
    
    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Configuration TP compacte
    st.markdown("""
    <p style="font-size: 0.9rem; font-weight: 600; color: #64748b; margin-bottom: 0.5rem;">
         CONFIGURATION
    </p>
    """, unsafe_allow_html=True)
    
    tp_options = {
        "TP1: Pixel & Bases": ["ex2", "ex4"],
        "TP2: Histogrammes": ["ex3"],
        "TP3: Filtrage Spatial": ["ex2"],
        "TP4: Fourier": ["ex1"],
        "TP5: Morphologie": ["ex3"],
        "TP6: Segmentation": ["ex1"],
        "TP7: Espaces Couleurs": ["ex2"],
        "TP8: Texture": ["ex2"]
    }
    
    selected_tp = st.selectbox(
        "TP",
        options=list(tp_options.keys()),
        label_visibility="collapsed"
    )
    
    tp_id = selected_tp.split(":")[0].replace("TP", "")
    
    available_ex = tp_options[selected_tp]
    ex_labels = [f"Exercice {ex.replace('ex','')}" for ex in available_ex]
    
    selected_ex = st.selectbox(
        "Ex",
        options=ex_labels,
        label_visibility="collapsed"
    )
    
    ex_id = "ex" + selected_ex.split(" ")[1]
    
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    # Boutons d'action compacts
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Info", use_container_width=True):
            st.session_state.show_theory = True
    
    with col2:
        lang = st.selectbox(
            "Code",
            options=["python", "matlab"],
            format_func=lambda x: "Python" if x == "python" else "Matlab",
            label_visibility="collapsed"
        )
    
    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
    
    # Bouton principal prominent
    if st.button("▶️  TRAITER", type="primary", use_container_width=True):
        if st.session_state.uploaded_file is None:
            st.error("⚠️ Chargez une image!")
        else:
            with st.spinner("⏳ Traitement..."):
                st.session_state.uploaded_file.seek(0)
                fig, msg = st.session_state.processor.process(
                    tp_id, ex_id, st.session_state.uploaded_file
                )
                st.session_state.current_result = fig
                st.session_state.current_message = msg
                st.session_state.uploaded_file.seek(0)
    
    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
    
    # Téléchargement code
    code_key = f"tp{tp_id}_{ex_id}"
    source_code = CourseContent.get_source_code(code_key, lang)
    ext = ".py" if lang == "python" else ".m"
    filename = f"traitement_tp{tp_id}_{ex_id}{ext}"
    
    st.download_button(
        label="Code",
        data=source_code,
        file_name=filename,
        mime="text/plain",
        use_container_width=True
    )


# --- ZONE PRINCIPALE ---
st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

# Affichage de la théorie
if 'show_theory' in st.session_state and st.session_state.show_theory:
    tp_key = f"TP{tp_id}"
    title, content = CourseContent.get_info(tp_key)
    
    with st.expander(f"📚 {title}", expanded=True):
        st.markdown(f'<div class="theory-content">{content}</div>', unsafe_allow_html=True)
        if st.button("✕ Fermer"):
            st.session_state.show_theory = False
            st.rerun()

# Affichage des résultats
if st.session_state.current_result is not None:
    st.markdown(
        f'<div style="padding: 1rem; border-radius: 10px; background-color: rgba(74, 222, 128, 0.1); border-left: 4px solid #4ade80; color: #4ade80; font-weight: 500; margin: 1rem 0;">✓ {st.session_state.current_message}</div>',
        unsafe_allow_html=True
    )
    st.pyplot(st.session_state.current_result, use_container_width=True)
elif st.session_state.current_message is not None:
    st.markdown(
        f'<div style="padding: 1rem; border-radius: 10px; background-color: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; color: #ef4444; font-weight: 500; margin: 1rem 0;">✗ {st.session_state.current_message}</div>',
        unsafe_allow_html=True
    )
else:
    # Message d'accueil moderne
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 20px; border: 2px dashed #cbd5e1;">
            <h3 style="color: #475569; margin-bottom: 1.5rem;">Prêt à commencer ?</h3>
            <div style="text-align: left; display: inline-block; color: #64748b; line-height: 2;">
                <p>1️⃣ Chargez votre image</p>
                <p>2️⃣ Sélectionnez TP et exercice</p>
                <p>3️⃣ Cliquez sur TRAITER</p>
                <p>4️⃣ Explorez les résultats</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer minimaliste
st.markdown('<div style="height: 3rem;"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;">Traitement d\'Images © 2025</p>',
    unsafe_allow_html=True
)
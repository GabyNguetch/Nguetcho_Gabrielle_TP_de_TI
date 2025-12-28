import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import threading
import time
from skimage.feature import graycomatrix, graycoprops # Pour la texture (TP8)

# --- Configuration du Thème ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ==============================================================================
# BASE DE DONNÉES PÉDAGOGIQUES (Cours + Codes Sources)
# ==============================================================================
class CourseContent:
    """Stocke le contenu théorique issu des cours et les templates de code."""
    
    @staticmethod
    def get_info(tp_id):
        infos = {
            "TP1": (
                "TP1 : Bases du Numérique & Matrice de Pixels",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Comprendre qu'une image numérique est une fonction discrète f(x,y) représentée "
                "par une matrice. Ce TP manipule la résolution spatiale (échantillonnage) et "
                "la résolution colorimétrique (quantification).\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. Echantillonnage : Le passage d'une scène continue à une grille discrète. "
                "Réduire la fréquence d'échantillonnage crée une pixellisation (Aliasing).\n"
                "2. Quantification : Discrétisation de l'amplitude du signal. Passer de "
                "256 niveaux (8 bits) à K niveaux réduit la qualité visuelle (phénomène de faux contours).\n"
                "3. Profil : Analyse 1D d'une ligne de l'image (f(x) à y fixé).\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Chargement : `cv2.imread(path, 0)` charge l'image en matrice Numpy uint8 (0-255).\n"
                "• Slicing (Sous-échantillonnage) : `img[::k, ::k]` utilise la syntaxe Numpy "
                "pour ne garder qu'un pixel tous les k pixels (pas d'interpolation ici, c'est brut).\n"
                "• Quantification : Utilisation de la division entière `(img // diviseur) * diviseur` "
                "pour forcer les valeurs des pixels à des paliers fixes.\n"
                "• Accès Pixel : `val = img[y, x]` montre l'accès direct aux coordonnées matricielles."
            ),
            
            "TP2": (
                "TP2 : Amélioration par Histogrammes (Luminance & Contraste)",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Analyser la distribution statistique des niveaux de gris pour corriger les défauts "
                "d'acquisition (sous-exposition, faible contraste). Se base sur le cours 'Histogramme d'une image'.\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. Histogramme H(k) : Compte le nombre d'occurrences de chaque niveau de gris k.\n"
                "2. Étirement Linéaire (Stretching) : Transformation affine g(x) = a*f(x) + b pour étaler "
                "la dynamique sur la plage complète [0, 255].\n"
                "3. Égalisation : Transformation non-linéaire qui cherche à rendre l'histogramme 'plat' (uniforme). "
                "Elle utilise la Fonction de Répartition Cumulée (CDF) comme fonction de transfert.\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Calcul : `plt.hist` ou `np.histogram` génèrent les données statistiques.\n"
                "• Étirement : `((img - min) / (max - min)) * 255` normalise les pixels (formule min-max).\n"
                "• Égalisation : `cv2.equalizeHist(img)` applique l'algorithme complet : calcul du CDF, "
                "normalisation et mapping (Look-Up Table) en une seule fonction optimisée en C++."
            ),
            
            "TP3": (
                "TP3 : Filtrage Spatial (Lissage & Débruitage)",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Modifier l'image par des opérations de voisinage (masques locaux). Illustration des "
                "concepts de convolution et de filtrage linéaire vs non-linéaire.\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. Convolution (g = f * h) : Un noyau h(x,y) glisse sur l'image. Chaque pixel est "
                "la somme pondérée de ses voisins.\n"
                "2. Filtre Moyenneur (Passe-bas) : Chaque poids = 1/N. Lisse mais floute les contours.\n"
                "3. Filtre Gaussien : Poids en forme de cloche (poids fort au centre). Lisse en préservant mieux la structure.\n"
                "4. Filtre Médian (Non-linéaire) : Remplace le pixel par la valeur médiane du voisinage. "
                "Redoutable contre le bruit impulsionnel 'Poivre et Sel' (valeurs extrêmes).\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Moyenne : `cv2.blur(img, (k,k))` ou convolution manuelle `cv2.filter2D`.\n"
                "• Gaussien : `cv2.GaussianBlur(img, (k,k), sigma)` génère le noyau gaussien automatiquement.\n"
                "• Médian : `cv2.medianBlur(img, k)` trie les pixels du voisinage et prend le centre. "
                "Ce n'est PAS une convolution matricielle."
            ),
            
            "TP4": (
                "TP4 : Domaine Fréquentiel (Transformée de Fourier)",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Passer du domaine spatial (pixels x,y) au domaine fréquentiel (fréquences u,v). "
                "Analyser l'image comme un signal (somme de sinusoïdes).\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. FFT (Fast Fourier Transform) : Décompose l'image. Le centre du spectre contient "
                "les basses fréquences (énergie, formes globales). La périphérie contient les hautes "
                "fréquences (contours, bruit).\n"
                "2. Filtrage Idéal : Multiplication du spectre par un masque (Cercle blanc ou noir).\n"
                "3. Propriété de Rotation : La rotation spatiale entraine une rotation spectrale.\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Analyse : `np.fft.fft2(img)` passe en complexe. `np.fft.fftshift` recentre le zéro (DC) au milieu.\n"
                "• Spectre Magnitude : `20*np.log(np.abs(fshift))` permet de visualiser le spectre "
                "dont la dynamique est trop grande (échelle logarithmique).\n"
                "• Filtrage : On multiplie directement la matrice complexe par un masque (0 ou 1) avant "
                "d'appliquer la `ifft2` (Inverse FFT) pour reconstruire l'image."
            ),
            
            "TP5": (
                "TP5 : Morphologie Mathématique",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Analyse de formes non-linéaire basée sur la théorie des ensembles. Essentiel pour "
                "nettoyer des masques binaires après seuillage.\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. Élément Structurant (SE) : La forme 'sonde' (carré, disque) qui analyse l'objet.\n"
                "2. Érosion (MIN) : 'Ronge' les objets. Supprime le bruit blanc isolé.\n"
                "3. Dilatation (MAX) : Épaissit les objets. Comble les trous noirs.\n"
                "4. Ouverture (Erosion -> Dilatation) : Supprime les petits objets sans changer la taille des gros.\n"
                "5. Fermeture (Dilatation -> Erosion) : Bouche les trous à l'intérieur des objets.\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Noyau : `cv2.getStructuringElement(cv2.MORPH_RECT, size)`.\n"
                "• Opérations de base : `cv2.erode` et `cv2.dilate`.\n"
                "• Opérations avancées : `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)` combine automatiquement "
                "les opérations de base pour l'ouverture, fermeture, ou gradient morphologique."
            ),
            
            "TP6": (
                "TP6 : Segmentation & Clustering",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Partitionner l'image en régions homogènes (sens). Approches 'Pixel' vs 'Région'.\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. Seuillage (Thresholding) : Sépare fond/forme. La méthode d'Otsu calcule "
                "automatiquement le seuil optimal qui minimise la variance intra-classe.\n"
                "2. Croissance de région : Part d'un 'germe' (seed) et agrège les pixels voisins similaires.\n"
                "3. K-Means (Clustering) : Algorithme non-supervisé. Il regroupe les vecteurs (R,G,B) des pixels "
                "en K groupes autour de centres de gravité (centroïdes) calculés itérativement.\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Otsu : `cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)`.\n"
                "• K-Means : On reformate l'image en liste de pixels via `reshape((-1,3))`. "
                "La fonction `cv2.kmeans` prend ces vecteurs et retourne les étiquettes (labels) de chaque pixel. "
                "On reconstruit ensuite l'image en coloriant chaque label avec la couleur de son centre."
            ),
            
            "TP7": (
                "TP7 : Espaces Colorimétriques (RGB vs HSV)",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Dépasser le modèle RGB qui est fortement corrélé (la lumière modifie R, G et B simultanément) "
                "pour une segmentation couleur robuste.\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. RGB (Red Green Blue) : Modèle additif technique (capteurs caméras). Difficile de séparer la couleur de son intensité.\n"
                "2. HSV (Hue Saturation Value) : Modèle perceptuel.\n"
                "   - Hue (Teinte) : La 'couleur' pure (angle sur le cercle chromatique).\n"
                "   - Saturation : Pureté de la couleur (vivacité).\n"
                "   - Value : Luminosité (clair ou sombre).\n"
                "La segmentation couleur en HSV consiste à isoler une plage de Teinte, quelle que soit la Valeur.\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• Conversion : `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`.\n"
                "• Masquage : `cv2.inRange(hsv_img, bornes_inf, bornes_sup)` crée un masque binaire (0 ou 255) "
                "où les pixels sont dans l'intervalle cible.\n"
                "• Extraction : `cv2.bitwise_and(src, src, mask=mask)` applique le masque."
            ),
            
            "TP8": (
                "TP8 : Analyse de Texture (Statistique & Fréquentielle)",
                "OBJECTIFS PÉDAGOGIQUES :\n"
                "Définir une région non par sa couleur moyenne, mais par son arrangement spatial (rugueux, ligné, pointillé).\n\n"
                
                "PRINCIPES THÉORIQUES :\n"
                "1. GLCM (Matrice de Co-occurrence) : Analyse statistique du second ordre. "
                "Compte combien de fois un niveau de gris `i` est voisin d'un niveau `j` à une distance `d`. "
                "On en déduit des descripteurs : Contraste (variations locales), Homogénéité, Énergie.\n"
                "2. Filtres de Gabor : Outil puissant simulant le système visuel humain. C'est une sinusoïde (fréquence) "
                "modulée par une gaussienne (localisation). Détecte des textures orientées à fréquences spécifiques.\n\n"
                
                "💻 IMPLÉMENTATION PYTHON :\n"
                "• GLCM : Librairie `skimage.feature.graycomatrix` pour calculer la matrice, puis `graycoprops` pour extraire "
                "le contraste ou la corrélation.\n"
                "• Gabor : `cv2.getGaborKernel(taille, sigma, theta, lambda, ...)` crée le noyau de convolution complexe. "
                "On l'applique via `filter2D` pour voir la réponse de la texture."
            )
        }
        return infos.get(tp_id, ("Information Inconnue", "Pas de détails disponibles pour ce TP."))
    
@staticmethod
def get_source_code(tp_ex_id, lang="python"):
        """
        Génère le code source complet (Python ou Matlab) pour l'exercice demandé.
        Le code est commenté pédagogiquement.
        """
        
        # ======================================================================
        #  CODES PYTHON
        # ======================================================================
        if lang == "python":
            python_codes = {
                # --- TP1 : BASES ---
                "tp1_ex2": r"""
# TP1 - Exercice 2 : Sous-échantillonnage
import cv2
import matplotlib.pyplot as plt

# 1. Charger l'image en niveaux de gris (flag 0)
# 'cameraman.tif' est une image classique pour ce test.
img = cv2.imread('image_source.jpg', 0)

if img is None:
    print("Erreur: Image non trouvée.")
    exit()

# 2. Définir les facteurs de sous-échantillonnage k
# k=2 signifie qu'on garde 1 pixel sur 2
k_values = [1, 2, 4, 8, 16]

plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values):
    # Manipulation matricielle avec Numpy Slicing [debut:fin:pas]
    # Pas d'interpolation ici (Zoom destructif pour observer l'aliasing)
    sous_ech = img[::k, ::k]
    
    plt.subplot(1, 5, i+1)
    plt.imshow(sous_ech, cmap='gray')
    plt.title(f"1 pixel sur {k}")
    plt.axis('off')

plt.tight_layout()
plt.show()
""",
                "tp1_ex4": r"""
# TP1 - Exercice 4 : Profil d'Intensité
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg', 0)
h, w = img.shape

# Choix d'une ligne au milieu de l'image
ligne_idx = h // 2

# Extraction de la ligne (vecteur 1D)
# Slicing numpy: ligne 'ligne_idx', toutes les colonnes ':'
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
plt.show()
""",

                # --- TP2 : HISTOGRAMMES ---
                "tp2_ex3": r"""
# TP2 - Exercice 3 : Égalisation d'Histogramme
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger en gris
img = cv2.imread('image_source.jpg', 0)

# L'égalisation applique une transformation non-linéaire qui aplatit l'histogramme
# en utilisant la fonction de distribution cumulative (CDF).
# cv2.equalizeHist est optimisé en C++.
img_eq = cv2.equalizeHist(img)

# Affichage comparatif
plt.figure(figsize=(12, 6))

# Original
plt.subplot(2,2,1); plt.imshow(img, cmap='gray'); plt.title("Originale"); plt.axis('off')
plt.subplot(2,2,2); plt.hist(img.flatten(), 256, [0,256], color='r')
plt.title("Histo Original"); plt.xlim([0,256])

# Égalisé
plt.subplot(2,2,3); plt.imshow(img_eq, cmap='gray'); plt.title("Égalisée (Contraste Maximisé)"); plt.axis('off')
plt.subplot(2,2,4); plt.hist(img_eq.flatten(), 256, [0,256], color='b')
plt.title("Histo Égalisé"); plt.xlim([0,256])

plt.tight_layout()
plt.show()
""",

                # --- TP3 : FILTRAGE ---
                "tp3_ex2": r"""
# TP3 - Exercice 2 : Comparaison Moyenneur vs Médian (Débruitage)
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

# 1. Filtre Moyenneur (Blur) - Linéaire
# Chaque pixel est la moyenne de son voisinage 5x5
# Inconvénient : Floute l'image et étale le bruit.
blur = cv2.blur(img_noisy, (5, 5))

# 2. Filtre Médian - Non Linéaire
# Trie les pixels du voisinage et prend la médiane.
# Avantage : Elimine totalement le bruit sel & poivre et préserve les bords.
median = cv2.medianBlur(img_noisy, 5)

plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(img_noisy, cmap='gray'); plt.title("Image Bruitée")
plt.subplot(132); plt.imshow(blur, cmap='gray'); plt.title("Moyenneur 5x5 (Echec)")
plt.subplot(133); plt.imshow(median, cmap='gray'); plt.title("Médian 5x5 (Réussite)")
plt.show()
""",

                # --- TP4 : FOURIER ---
                "tp4_ex1": r"""
# TP4 - Exercice 1 : Spectre de Fourier 2D (FFT)
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg', 0)

# 1. Transformation de Fourier Discrète 2D
# On utilise numpy pour une précision float64 complexe.
f = np.fft.fft2(img)

# 2. Shift (Centrage)
# Par défaut, la fréquence zéro (DC) est en haut à gauche (0,0).
# fftshift déplace le DC au centre de l'image (N/2, M/2).
fshift = np.fft.fftshift(f)

# 3. Magnitude (Spectre d'amplitude)
# Le spectre a une très grande dynamique. On utilise une échelle log pour visualiser.
# Formule : 20 * log(1 + module(Nombre Complexe))
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure(figsize=(10, 5))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Domaine Spatial'), plt.axis('off')
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Spectre Fréquentiel (Log Magnitude)'), plt.axis('off')
plt.show()
""",

                # --- TP5 : MORPHOLOGIE ---
                "tp5_ex3": r"""
# TP5 - Exercice 3 : Gradient Morphologique
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image_source.jpg', 0)

# Pré-traitement : Binarisation (Otsu) pour travailler sur des formes nettes
_, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Définition de l'Elément Structurant (SE)
# Carré 3x3
kernel = np.ones((3,3), np.uint8)

# 1. Dilatation (Agrandissement)
dilated = cv2.dilate(bin_img, kernel, iterations=1)

# 2. Erosion (Rétrécissement)
eroded = cv2.erode(bin_img, kernel, iterations=1)

# 3. Gradient = Dilatation - Erosion
# Ceci extrait les frontières des objets (épaisseur dépendant du kernel)
# OpenCV a une fonction dédiée : cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
gradient = cv2.subtract(dilated, eroded)

plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(bin_img, cmap='gray'); plt.title("Binaire")
plt.subplot(132); plt.imshow(dilated, cmap='gray'); plt.title("Dilatée")
plt.subplot(133); plt.imshow(gradient, cmap='gray'); plt.title("Gradient (Contours)")
plt.show()
""",

                # --- TP6 : SEGMENTATION K-MEANS ---
                "tp6_ex1": r"""
# TP6 - Exercice 1 : Segmentation Couleur par K-Means
import cv2
import numpy as np
import matplotlib.pyplot as plt

# K-Means clustering couleur
# 1. Charger en couleur (RGB)
img = cv2.imread('image_source.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Restructuration des données
# K-Means attend un tableau 2D de type float32 : (N_pixels, 3_canaux)
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# 3. Définition des critères d'arrêt (Epsilon ou Iterations Max)
# Arrêter si 10 itérations OU précision epsilon 1.0 atteinte
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 4. Application K-Means avec K=4 Clusters
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 5. Reconstruction de l'image
# Convertir les centres (couleurs dominantes) en uint8
center = np.uint8(center)
# Remplacer chaque pixel par la couleur de son centre
res = center[label.flatten()]
# Redimensionner comme l'image originale
res2 = res.reshape((img.shape))

plt.figure(figsize=(10, 5))
plt.subplot(121); plt.imshow(img); plt.title("Original RGB")
plt.subplot(122); plt.imshow(res2); plt.title(f"K-Means (K={K} couleurs)")
plt.show()
""",

                # --- TP7 : COULEUR HSV ---
                "tp7_ex2": r"""
# TP7 - Exercice 2 : Segmentation dans l'espace HSV
import cv2
import numpy as np
import matplotlib.pyplot as plt

# L'espace HSV (Hue Saturation Value) sépare l'information chromatique (Hue)
# de l'intensité (Value). C'est plus robuste aux ombres que le RGB.

img = cv2.imread('image_source.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Conversion vers HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Définition de la plage de couleur à garder
# Exemple : Filtrer le vert/jaune
# OpenCV H: [0,180], S: [0,255], V: [0,255]
lower_val = np.array([20, 50, 50])
upper_val = np.array([40, 255, 255])

# Création du masque binaire
# pixel = 255 si dans la plage, 0 sinon
mask = cv2.inRange(hsv, lower_val, upper_val)

# Application du masque sur l'image originale
res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(img_rgb); plt.title("Image Originale")
plt.subplot(132); plt.imshow(mask, cmap='gray'); plt.title("Masque (Seuillage Hue)")
plt.subplot(133); plt.imshow(res); plt.title("Extraction Objet")
plt.show()
""",

                # --- TP8 : TEXTURE GLCM ---
                "tp8_ex2": r"""
# TP8 - Exercice 2 : Analyse de Texture (GLCM & Gabor)
# Nécessite : pip install scikit-image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

img = cv2.imread('image_source.jpg', 0)

# --- 1. Approche Statistique : GLCM ---
# Grey Level Co-occurrence Matrix
# Analyse les paires de pixels voisins à distance 1 et angle 0 (droite)
# Niveaux limités à 256
glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, 
                    symmetric=True, normed=True)

# Extraction des descripteurs de Haralick
contrast = graycoprops(glcm, 'contrast')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

print(f"Descripteurs de Texture :")
print(f" - Contraste : {contrast:.2f} (fort = variations locales)")
print(f" - Homogénéité: {homogeneity:.2f} (fort = image lisse)")
print(f" - Energie : {energy:.4f}")

# --- 2. Approche Fréquentielle : Filtre de Gabor ---
# Simule le cortex visuel. Sensible à une fréquence et une orientation.
ksize = 21 # Taille du noyau
sigma = 5.0 # Ecart-type enveloppe gaussienne
theta = np.pi / 4 # Orientation (45 degrés)
lambd = 10.0 # Longueur d'onde du cosinus
gamma = 0.5 # Ratio d'aspect

# Création noyau
g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

# Application (Convolution)
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(121); plt.imshow(img, cmap='gray'); plt.title("Texture Origine")
plt.subplot(122); plt.imshow(filtered_img, cmap='gray'); 
plt.title(f"Réponse Gabor (45 deg)\nContraste GLCM: {contrast:.1f}")
plt.show()
"""
            }
            default = f"# Code Python pour {tp_ex_id} non trouvé dans la base de démonstration."
            return python_codes.get(tp_ex_id, default)

        # ======================================================================
        #  CODES MATLAB
        # ======================================================================
        elif lang == "matlab":
            matlab_codes = {
                # --- TP1 ---
                "tp1_ex2": r"""
% TP1 - Ex2 : Sous-Echantillonnage Spatial
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
    
    % Extraction matricielle directe: de 1 à la fin avec un pas de k
    % (Row Start : Step : End)
    I_sub = I(1:k:end, 1:k:end);
    
    subplot(1, 4, i);
    imshow(I_sub);
    title(['1 pixel sur ', num2str(k)]);
end
""",
                "tp1_ex4": r"""
% TP1 - Ex4 : Profil d'Intensité (Ligne)
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

[h, w] = size(I);
num_ligne = floor(h/2); % Milieu

% Extraction vecteur ligne
profil = I(num_ligne, :);

figure;
subplot(2,1,1); imshow(I); hold on;
% Tracer ligne rouge sur l'image
line([1 w], [num_ligne num_ligne], 'Color', 'r', 'LineWidth', 2);
title('Image');

subplot(2,1,2); plot(profil, 'b'); grid on;
title(['Profil intensité ligne ', num2str(num_ligne)]);
xlim([1 w]); ylim([0 255]);
xlabel('Position x'); ylabel('Niveau gris');
""",

                # --- TP2 ---
                "tp2_ex3": r"""
% TP2 - Ex3 : Égalisation d'histogramme
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

% Calcul Histogramme cumulé normalisé (manuel ou via histeq)
% Ici méthode 'toolbox' efficace :
I_eq = histeq(I); 

figure;
subplot(2,2,1); imshow(I); title('Originale');
subplot(2,2,2); imhist(I); title('Histo Orig');

subplot(2,2,3); imshow(I_eq); title('Egalisée');
subplot(2,2,4); imhist(I_eq); title('Histo Egalisé');
""",

                # --- TP3 ---
                "tp3_ex2": r"""
% TP3 - Ex2 : Filtres spatiaux (Comparaison)
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

% Ajout Bruit Poivre et Sel (Salt & Pepper) pour le test
J = imnoise(I, 'salt & pepper', 0.05);

% 1. Filtre Moyenneur (Linear)
h_moy = fspecial('average', [5 5]); % Noyau 5x5 de 1/25
I_moy = imfilter(J, h_moy, 'replicate'); 

% 2. Filtre Médian (Non-linear - Rank filter)
% Plus efficace pour le bruit impulsionnel
I_med = medfilt2(J, [5 5]);

figure;
subplot(1,3,1); imshow(J); title('Bruitée (Poivre et Sel)');
subplot(1,3,2); imshow(I_moy); title('Moyenne 5x5 (Flou)');
subplot(1,3,3); imshow(I_med); title('Médiane 5x5 (Net)');
""",

                # --- TP4 ---
                "tp4_ex1": r"""
% TP4 - Ex1 : Spectre de Fourier
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end
I_double = double(I);

% 1. FFT 2D
F = fft2(I_double);

% 2. Centrage (Fréq 0 au centre)
F_sh = fftshift(F);

% 3. Module (Magnitude) Logarithmique
% log(1 + abs(F)) pour compression dynamique
S = log(1 + abs(F_sh));

figure;
subplot(1,2,1); imshow(uint8(I)); title('Spatial');
% Affichage en fausses couleurs avec imagesc
subplot(1,2,2); imagesc(S); axis image; colormap jet; 
title('Spectre Fréquentiel (Log)');
colorbar;
""",

                # --- TP5 ---
                "tp5_ex3": r"""
% TP5 - Ex3 : Gradient Morphologique
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

% Seuillage Otsu
level = graythresh(I);
BW = imbinarize(I, level);

% Élément structurant (Square 3x3)
se = strel('square', 3);

% Dilatation & Érosion
bw_dil = imdilate(BW, se);
bw_ero = imerode(BW, se);

% Gradient = Dilatation - Erosion
grad = bw_dil - bw_ero;

figure;
subplot(1,3,1); imshow(BW); title('Binaire');
subplot(1,3,2); imshow(bw_dil); title('Dilatation');
subplot(1,3,3); imshow(grad); title('Gradient (Contours)');
""",

                # --- TP6 ---
                "tp6_ex1": r"""
% TP6 - Ex1 : Segmentation K-Means
clc; clear; close all;

I = imread('image_source.jpg');
% On garde en couleur

% Conversion Matrice (Ligne x Colonne, 3) -> Vecteur (N_pixels, 3)
rows = size(I, 1);
cols = size(I, 2);
data = double(reshape(I, rows * cols, 3));

% K-Means avec K=4 clusters
K = 4;
[cluster_idx, cluster_center] = kmeans(data, K, ...
                                      'Distance', 'sqEuclidean', ...
                                      'Replicates', 3);

% Reconstruction de l'image segmentée
% Remplacement de chaque pixel par le centre de son cluster
pixel_labels = reshape(cluster_idx, rows, cols);
segmented_images = cell(1, 3);
rgb_label = repmat(pixel_labels, [1 1 3]);

% Coloration pour visualisation (Mapping centre -> uint8)
res = reshape(cluster_center(cluster_idx, :), rows, cols, 3);
res = uint8(res);

figure;
subplot(1,2,1); imshow(I); title('Original');
subplot(1,2,2); imshow(res); title(['Segmentation K-Means (K=' num2str(K) ')']);
""",

                # --- TP7 ---
                "tp7_ex2": r"""
% TP7 - Ex2 : Segmentation dans l'espace HSV
clc; clear; close all;

I = imread('image_source.jpg');

% 1. Conversion RGB -> HSV
I_hsv = rgb2hsv(I);

% Canaux H, S, V sont normalisés entre [0, 1] dans Matlab
H = I_hsv(:,:,1);
S = I_hsv(:,:,2);
V = I_hsv(:,:,3);

% 2. Création de masque
% Exemple : Segmentation des teintes rouges
% Le rouge est autour de 0 (ou 1 car cyclique)
% Seuil H < 0.05 ou H > 0.95
% Seuil Saturation > 0.4 (pour ne pas prendre les blancs/gris)
mask = ((H < 0.05) | (H > 0.95)) & (S > 0.4);

% 3. Application masque
% Mise à noir des pixels hors masque
I_masked = I;
r = I(:,:,1); r(~mask) = 0; I_masked(:,:,1) = r;
g = I(:,:,2); g(~mask) = 0; I_masked(:,:,2) = g;
b = I(:,:,3); b(~mask) = 0; I_masked(:,:,3) = b;

figure;
subplot(1,3,1); imshow(I); title('RGB Original');
subplot(1,3,2); imshow(mask); title('Masque (Teinte)');
subplot(1,3,3); imshow(I_masked); title('Extraction Objet');
""",

                # --- TP8 ---
                "tp8_ex2": r"""
% TP8 - Ex2 : Analyse Texture (GLCM)
clc; clear; close all;

I = imread('image_source.jpg');
if size(I,3)==3, I=rgb2gray(I); end

% 1. GLCM (Grey-Level Co-occurrence Matrix)
% Calcul pour un offset [0 1] (pixel voisin droite immédiat)
glcm = graycomatrix(I, 'Offset', [0 1], 'NumLevels', 256, 'Symmetric', true);

% 2. Propriétés (Haralick)
stats = graycoprops(glcm, {'Contrast', 'Homogeneity', 'Energy'});

fprintf('Texture Properties:\n');
fprintf('- Contraste: %.4f\n', stats.Contrast);
fprintf('- Homogénéité: %.4f\n', stats.Homogeneity);
fprintf('- Energie: %.4f\n', stats.Energy);

% 3. Filtre de Gabor
% Analyse locale fréquentielle
wavelength = 8;
orientation = 90; % Vertical
[mag, phase] = imgaborfilt(I, wavelength, orientation);

figure;
subplot(1,2,1); imshow(I); title('Image Source');
subplot(1,2,2); imshow(mag, []); 
title(['Gabor Magnitude (Orientation=' num2str(orientation) ')']);
"""
            }
            default = f"% Code Matlab pour {tp_ex_id} non disponible."
            return matlab_codes.get(tp_ex_id, default)
        
        return "Erreur de langue"
# ==============================================================================
# MOTEUR DE TRAITEMENT (LOGIQUE MÉTIER)
# ==============================================================================
class ProcessingEngine:
    def load_image(self, path, grayscale=False):
        if path is None: return None
        if grayscale:
            return cv2.imread(path, 0)
        else:
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def process(self, tp, ex, path):
        """Dispatcheur central"""
        try:
            func = getattr(self, f"algo_{tp}_{ex}")
            return func(path)
        except AttributeError:
            return None, "Algorithme non implémenté."
        except Exception as e:
            return None, str(e)

    # --- TP1 : Bases ---
    def algo_1_ex2(self, path):
        img = self.load_image(path, True)
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        fig.patch.set_facecolor('#242424')
        
        # Original
        axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original', color='white'); axes[0].axis('off')
        # K=2
        axes[1].imshow(img[::2, ::2], cmap='gray'); axes[1].set_title('1/2 (Zoomé)', color='white'); axes[1].axis('off')
        # K=4
        axes[2].imshow(img[::8, ::8], cmap='gray'); axes[2].set_title('1/8 (Pixellisé)', color='white'); axes[2].axis('off')
        return fig, "Sous-échantillonnage K=1, 2, 8"

    def algo_1_ex4(self, path):
        img = self.load_image(path, True)
        h, w = img.shape
        mid = h // 2
        line = img[mid, :]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        fig.patch.set_facecolor('#242424')
        
        ax1.imshow(img, cmap='gray'); ax1.axhline(mid, color='red'); ax1.axis('off')
        ax2.plot(line, color='cyan'); ax2.set_facecolor('#333333'); 
        ax2.tick_params(colors='white'); ax2.grid(alpha=0.3)
        return fig, f"Profil d'intensité Ligne {mid}"

    # --- TP2 : Histogrammes ---
    def algo_2_ex3(self, path):
        img = self.load_image(path, True)
        eq = cv2.equalizeHist(img)
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        fig.patch.set_facecolor('#242424')
        
        ax[0,0].imshow(img, cmap='gray'); ax[0,0].set_title('Orig', color='white'); ax[0,0].axis('off')
        ax[0,1].hist(img.ravel(), 256, [0,256], color='gray'); ax[0,1].set_facecolor('#333333'); ax[0,1].tick_params(colors='white')
        
        ax[1,0].imshow(eq, cmap='gray'); ax[1,0].set_title('Egalisée', color='white'); ax[1,0].axis('off')
        ax[1,1].hist(eq.ravel(), 256, [0,256], color='cyan'); ax[1,1].set_facecolor('#333333'); ax[1,1].tick_params(colors='white')
        plt.tight_layout()
        return fig, "Égalisation d'histogramme (Contraste)"

    # --- TP3 : Filtrage ---
    def algo_3_ex2(self, path):
        img = self.load_image(path, True)
        
        # Bruit poivre et sel pour tester le médian
        noise = img.copy()
        mask = np.random.randint(0, 100, img.shape)
        noise[mask < 2] = 0
        noise[mask > 98] = 255
        
        med = cv2.medianBlur(noise, 5)
        avg = cv2.blur(noise, (5,5))
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#242424')
        ax[0].imshow(noise, cmap='gray'); ax[0].set_title('Bruit P&S', color='white'); ax[0].axis('off')
        ax[1].imshow(avg, cmap='gray'); ax[1].set_title('Moyenne 5x5 (Flou)', color='white'); ax[1].axis('off')
        ax[2].imshow(med, cmap='gray'); ax[2].set_title('Médian 5x5 (Net)', color='white'); ax[2].axis('off')
        return fig, "Comparaison Moyenne vs Médiane"

    # --- TP4 : Fourier ---
    def algo_4_ex1(self, path):
        img = self.load_image(path, True)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#242424')
        ax[0].imshow(img, cmap='gray'); ax[0].axis('off'); ax[0].set_title("Spatial", color='white')
        ax[1].imshow(magnitude, cmap='gray'); ax[1].axis('off'); ax[1].set_title("Spectre Fréquentiel", color='white')
        return fig, "Transformée de Fourier 2D"

    # --- TP5 : Morpho ---
    def algo_5_ex3(self, path):
        img = self.load_image(path, True)
        ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        
        erosion = cv2.erode(bw, kernel, iterations=1)
        dilatation = cv2.dilate(bw, kernel, iterations=1)
        gradient = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)
        
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        fig.patch.set_facecolor('#242424')
        
        ax[0,0].imshow(bw, cmap='gray'); ax[0,0].set_title('Binaire', color='white')
        ax[0,1].imshow(gradient, cmap='gray'); ax[0,1].set_title('Gradient Morpho', color='white')
        ax[1,0].imshow(erosion, cmap='gray'); ax[1,0].set_title('Erosion', color='white')
        ax[1,1].imshow(dilatation, cmap='gray'); ax[1,1].set_title('Dilatation', color='white')
        for a in ax.flatten(): a.axis('off')
        return fig, "Opérations Morphologiques"

    # --- TP6 : Segmentation ---
    def algo_6_ex1(self, path):
        img = self.load_image(path) # Color
        pixel_vals = img.reshape((-1, 3))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 4
        _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        res2 = res.reshape((img.shape))
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#242424')
        ax[0].imshow(img); ax[0].axis('off'); ax[0].set_title("Original", color='white')
        ax[1].imshow(res2); ax[1].axis('off'); ax[1].set_title(f"Segmentation K-Means (K={k})", color='white')
        return fig, "Clustering de couleurs K-Means"

    # --- TP7 : Couleur HSV ---
    def algo_7_ex2(self, path):
        img = self.load_image(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Ex: Detecter le rouge/orange
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = mask1 | mask2
        seg = cv2.bitwise_and(img, img, mask=mask)
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#242424')
        ax[0].imshow(img); ax[0].set_title("RGB", color='white')
        ax[1].imshow(hsv[:,:,0], cmap='hsv'); ax[1].set_title("Canal Teinte (H)", color='white')
        ax[2].imshow(seg); ax[2].set_title("Segmentation par Couleur", color='white')
        for a in ax: a.axis('off')
        return fig, "Segmentation dans l'espace HSV"

    # --- TP8 : Texture ---
    def algo_8_ex2(self, path):
        img = self.load_image(path, True)
        
        # GLCM Matrix simple (angle 0, dist 1)
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Filtre Gabor simple
        ksize = 31
        sigma = 4.0
        theta = np.pi / 4
        lambd = 10.0
        gamma = 0.5
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        f_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#242424')
        
        ax[0].imshow(img, cmap='gray'); ax[0].set_title(f"GLCM\nContraste: {contrast:.2f}, Homo: {homogeneity:.2f}", color='white')
        ax[1].imshow(f_img, cmap='gray'); ax[1].set_title("Réponse filtre Gabor (45 deg)", color='white')
        for a in ax: a.axis('off')
        return fig, "Analyse de texture (GLCM & Gabor)"


# ==============================================================================
# INTERFACE GRAPHIQUE (GUI)
# ==============================================================================
class ImageProcessingGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuration Fenêtre ---
        self.title("Travaux pratiques de Traitement d'Images : NGUETCHO BIADOU CHLOE GABRIELLE")
        self.geometry("1400x900")
        
        # --- État de l'application ---
        self.img_path = None
        self.processor = ProcessingEngine()
        self.selected_tp_id = "1"
        self.selected_ex_id = "ex2"
        self.download_language = ctk.StringVar(value="python")

        self.layout_setup()

    def layout_setup(self):
        # Grille principale: Sidebar (col 0), Main (col 1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === 1. SIDEBAR (GAUCHE) ===
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Header Logo
        self.logo = ctk.CTkLabel(self.sidebar, text="NGUETCHO Gabrielle", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo.pack(pady=(40, 5))
        self.version = ctk.CTkLabel(self.sidebar, text="TP de traitement d'images", text_color="gray70", font=("Arial", 12))
        self.version.pack(pady=(0, 20))

        # Zone Image
        self.box_img = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.box_img.pack(fill="x", padx=15, pady=10)
        self.lbl_import = ctk.CTkLabel(self.box_img, text="FICHIER IMAGE", font=("Arial", 12, "bold"), text_color="#3B8ED0", anchor="w")
        self.lbl_import.pack(fill="x")
        
        self.btn_load = ctk.CTkButton(self.box_img, text="📂 Charger Image...", height=40, 
                                      fg_color="#1f538d", hover_color="#14375e", 
                                      command=self.load_file)
        self.btn_load.pack(fill="x", pady=5)
        self.lbl_fileinfo = ctk.CTkLabel(self.box_img, text="Aucune image sélectionnée", text_color="gray50", font=("Arial", 11))
        self.lbl_fileinfo.pack(fill="x")

        # Séparateur
        ctk.CTkFrame(self.sidebar, height=1, fg_color="gray30").pack(fill="x", padx=15, pady=15)

        # Zone TP
        self.lbl_conf = ctk.CTkLabel(self.sidebar, text="CONFIGURATION DU LAB", font=("Arial", 12, "bold"), text_color="#3B8ED0", anchor="w")
        self.lbl_conf.pack(fill="x", padx=15)

        # Menus déroulants
        self.tp_options = {
            "TP1: Pixel & Bases": ["ex2", "ex4"],
            "TP2: Histogrammes": ["ex3"],
            "TP3: Filtrage Spatial": ["ex2"],
            "TP4: Fourier": ["ex1"],
            "TP5: Morphologie": ["ex3"],
            "TP6: Segmentation": ["ex1"],
            "TP7: Espaces Couleurs": ["ex2"],
            "TP8: Texture": ["ex2"]
        }
        
        self.combo_tp = ctk.CTkOptionMenu(self.sidebar, values=list(self.tp_options.keys()), command=self.on_tp_change)
        self.combo_tp.pack(fill="x", padx=15, pady=(10, 5))
        
        self.combo_ex = ctk.CTkOptionMenu(self.sidebar, values=["Exercice 2"])
        self.combo_ex.pack(fill="x", padx=15, pady=5)

        # --- Bouton INFO TP (Nouveau) ---
        self.btn_info = ctk.CTkButton(self.sidebar, text="ℹ️  Infos Théoriques du TP", 
                                      fg_color="transparent", border_width=1, border_color="#3B8ED0", text_color="#3B8ED0",
                                      command=self.show_tp_theory)
        self.btn_info.pack(fill="x", padx=15, pady=(20, 10))

        # Footer Actions (Sidebar bas)
        self.frame_actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_actions.pack(side="bottom", fill="x", padx=15, pady=30)

        self.btn_run = ctk.CTkButton(self.frame_actions, text="▶  LANCER LE TRAITEMENT", height=50, 
                                     font=("Arial", 14, "bold"), fg_color="#106A43", hover_color="#0b4a2f",
                                     command=self.run_process)
        self.btn_run.pack(fill="x", pady=10)

        # === 2. MAIN AREA (DROITE) ===
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Zone Header Droite
        self.header = ctk.CTkFrame(self.main_area, height=60, fg_color="#1f1f1f", corner_radius=10)
        self.header.pack(fill="x", pady=(0, 10))
        self.lbl_title = ctk.CTkLabel(self.header, text="Visualisation des Résultats", font=("Arial", 18, "bold"))
        self.lbl_title.place(relx=0.02, rely=0.5, anchor="w")
        self.lbl_status = ctk.CTkLabel(self.header, text="En attente...", text_color="#e5b800")
        self.lbl_status.place(relx=0.98, rely=0.5, anchor="e")

        # Zone Contenu (Graphique)
        self.plot_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b", corner_radius=10)
        self.plot_frame.pack(fill="both", expand=True)
        self.plot_container = None # Pour stocker le canvas

        # Zone Footer Droite (Téléchargement)
        self.footer = ctk.CTkFrame(self.main_area, height=60, fg_color="#1f1f1f", corner_radius=10)
        self.footer.pack(fill="x", pady=(10, 0))
        
        self.lbl_dl = ctk.CTkLabel(self.footer, text="Code Source :", font=("Arial", 12, "bold"))
        self.lbl_dl.pack(side="left", padx=15, pady=10)
        
        self.radio_py = ctk.CTkRadioButton(self.footer, text="Python", variable=self.download_language, value="python")
        self.radio_py.pack(side="left", padx=10)
        self.radio_mat = ctk.CTkRadioButton(self.footer, text="Matlab", variable=self.download_language, value="matlab")
        self.radio_mat.pack(side="left", padx=10)
        
        self.btn_download = ctk.CTkButton(self.footer, text="📥 Télécharger le fichier source", 
                                          command=self.download_code, fg_color="#333")
        self.btn_download.pack(side="right", padx=15, pady=10)

        # Init state
        self.on_tp_change(list(self.tp_options.keys())[0])

    # --- LOGIQUE ---

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.tif;*.jpeg;*.bmp")])
        if filename:
            self.img_path = filename
            short_name = filename.split('/')[-1]
            if len(short_name) > 25: short_name = short_name[:22] + "..."
            self.lbl_fileinfo.configure(text=f"Chargé: {short_name}", text_color="#4ade80") # Vert clair

    def on_tp_change(self, choice):
        # Update TP selection variable
        self.selected_tp_id = choice.split(":")[0].replace("TP", "") # "TP1" -> "1"
        
        # Update exercices dropdown
        exs = self.tp_options[choice]
        nice_names = [f"Exercice {x.replace('ex','')}" for x in exs]
        self.combo_ex.configure(values=nice_names)
        self.combo_ex.set(nice_names[0])
        self.selected_ex_id = exs[0]

    def show_tp_theory(self):
        """Affiche une fenêtre pop-up avec les infos théoriques du TP"""
        tp_key = f"TP{self.selected_tp_id}"
        title, content = CourseContent.get_info(tp_key)
        
        # Création d'une fenêtre top-level
        info_window = ctk.CTkToplevel(self)
        info_window.title(f"Théorie : {title}")
        info_window.geometry("600x500")
        info_window.attributes("-topmost", True)
        
        # Titre
        lbl_h = ctk.CTkLabel(info_window, text=title, font=("Arial", 20, "bold"), text_color="#3B8ED0")
        lbl_h.pack(pady=15, padx=20, anchor="w")
        
        # Contenu défilant
        textbox = ctk.CTkTextbox(info_window, font=("Arial", 14), width=560, height=400)
        textbox.pack(pady=10, padx=20, fill="both", expand=True)
        textbox.insert("0.0", content)
        textbox.configure(state="disabled") # Lecture seule

    def run_process(self):
        if not self.img_path:
            messagebox.showwarning("Erreur", "Veuillez charger une image d'abord.")
            return

        # Mise à jour IDs d'exercice depuis le combo (au cas où ça a changé)
        current_ex_choice = self.combo_ex.get() # "Exercice 2"
        self.selected_ex_id = "ex" + current_ex_choice.split(" ")[1] # "ex2"

        self.lbl_status.configure(text="Traitement en cours...", text_color="#e5b800")
        self.update_idletasks()

        # Threading pour ne pas figer l'UI
        thread = threading.Thread(target=self._run_process_thread)
        thread.start()

    def _run_process_thread(self):
        fig, msg = self.processor.process(self.selected_tp_id, self.selected_ex_id, self.img_path)
        self.after(50, lambda: self._display_result(fig, msg))

    def _display_result(self, fig, msg):
        # Nettoyer l'ancien graphe
        if self.plot_container:
            self.plot_container.get_tk_widget().destroy()
            for widget in self.plot_frame.winfo_children(): widget.destroy()

        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            self.plot_container = canvas
            self.plot_container.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            self.lbl_status.configure(text=f"Terminé : {msg}", text_color="#4ade80")
        else:
            lbl_err = ctk.CTkLabel(self.plot_frame, text=f"Erreur : {msg}", text_color="#ef4444")
            lbl_err.pack(pady=100)
            self.lbl_status.configure(text="Echec", text_color="#ef4444")

    def download_code(self):
        lang = self.download_language.get() # 'python' ou 'matlab'
        
        # ID pour le dictionnaire de code (tp1_ex2)
        current_ex_choice = self.combo_ex.get() 
        ex_id = "ex" + current_ex_choice.split(" ")[1]
        key_id = f"tp{self.selected_tp_id}_{ex_id}"
        
        # Récupération du code
        code_str = CourseContent.get_source_code(key_id, lang)
        
        # Extension
        ext = ".py" if lang == "python" else ".m"
        file_types = [("Python File", "*.py")] if lang == "python" else [("Matlab File", "*.m")]
        
        # Boite de dialogue de sauvegarde
        f = filedialog.asksaveasfile(mode='w', defaultextension=ext, 
                                     filetypes=file_types,
                                     initialfile=f"traitement_tp{self.selected_tp_id}_{ex_id}{ext}")
        if f:
            f.write(code_str)
            f.close()
            messagebox.showinfo("Succès", f"Fichier {lang.capitalize()} enregistré avec succès.")

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    try:
        app = ImageProcessingGUI()
        app.mainloop()
    except ImportError as e:
        print(f"Erreur critique: Il manque une librairie : {e}")
        print("Installez : pip install customtkinter opencv-python numpy matplotlib scikit-image")
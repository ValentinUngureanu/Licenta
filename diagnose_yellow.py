"""
Script de diagnoza pentru pragurile pixelilor galbeni din GetYellowPix.

Cum sa-l rulezi:
    python diagnose_yellow.py

Pune-l langa codintial.py si schimba IMG_PATH daca e cazul.
"""

import numpy as np
from skimage import io
import cv2

IMG_PATH = r'C:\Facultate\AN4\Licenta\Licenta-Cod\ORIGINAL_IMAGES\0.jpg'

img = io.imread(IMG_PATH)

# Normalizare la 3 canale RGB (acelasi pattern ca in main())
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
elif img.ndim == 3 and img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

print(f'Imagine: {img.shape}, dtype = {img.dtype}')
print(f'Range R: [{img[:,:,0].min()}, {img[:,:,0].max()}]')
print(f'Range G: [{img[:,:,1].min()}, {img[:,:,1].max()}]')
print(f'Range B: [{img[:,:,2].min()}, {img[:,:,2].max()}]')

# === Test 1: Pragurile actuale din GetYellowPix ===
r_q, g_q, b_q = 169, 169, 105
mask_current = (img[:,:,0] >= r_q) & (img[:,:,1] >= g_q) & (img[:,:,2] <= b_q)
n_current = mask_current.sum()
print(f'\n[Praguri actuale] R>={r_q}, G>={g_q}, B<={b_q}: {n_current} pixeli')

# === Test 2: Praguri progresiv mai relaxate pe B ===
print('\n[Test praguri B coborate progresiv]')
for b_test in [105, 100, 98, 95, 90, 85, 80, 70, 60, 50]:
    mask = (img[:,:,0] >= 169) & (img[:,:,1] >= 169) & (img[:,:,2] <= b_test)
    print(f'  R>=169, G>=169, B<={b_test}: {mask.sum()} pixeli')

# === Test 3: Cauta pixeli "galbui" mai larg ===
# Galben = R mare, G mare, B mic. Hai sa vedem ce gasim la praguri mai relaxate.
print('\n[Test cu praguri mai relaxate pe R/G]')
for r_test in [200, 180, 160, 150, 140]:
    mask = (img[:,:,0] >= r_test) & (img[:,:,1] >= r_test) & (img[:,:,2] <= 100)
    print(f'  R>={r_test}, G>={r_test}, B<=100: {mask.sum()} pixeli')

# === Test 4: Identificare automata - pixeli unde R~G si B mic ===
# Galbenul pur: R aproape egal cu G, B mult mai mic
diff_rg = np.abs(img[:,:,0].astype(int) - img[:,:,1].astype(int))
yellow_like = (img[:,:,0] >= 150) & (img[:,:,1] >= 150) & (diff_rg < 30) & (img[:,:,2] < 150)
print(f'\n[Auto-detect] Pixeli "galbui" (R,G mari, ~egali, B mic): {yellow_like.sum()}')

if yellow_like.sum() > 0:
    ys, xs = np.where(yellow_like)
    rs = img[ys, xs, 0]
    gs = img[ys, xs, 1]
    bs = img[ys, xs, 2]
    print(f'  Pozitie: rows [{ys.min()}, {ys.max()}], cols [{xs.min()}, {xs.max()}]')
    print(f'  R range: [{rs.min()}, {rs.max()}], median = {int(np.median(rs))}')
    print(f'  G range: [{gs.min()}, {gs.max()}], median = {int(np.median(gs))}')
    print(f'  B range: [{bs.min()}, {bs.max()}], median = {int(np.median(bs))}')
    print(f'\n  >>> Sugestie: foloseste R>={rs.min()}, G>={gs.min()}, B<={bs.max()}')
else:
    print('  Nu am gasit pixeli galbui nici cu praguri relaxate. Verifica vizual culoarea markerului.')

# === Test 5: Salvez o masca vizuala ===
out = img.copy()
out[mask_current] = [255, 0, 0]  # rosu = ce gaseste pragul actual
out[yellow_like & ~mask_current] = [0, 255, 255]  # cyan = ce ar gasi auto-detect (in plus)
cv2.imwrite('diagnose_yellow_mask.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print('\n[Salvat] diagnose_yellow_mask.png')
print('  ROSU = pixeli detectati cu pragul ACTUAL (169/169/105)')
print('  CYAN = pixeli detectati de auto-detect dar PIERDUTI de pragul actual')
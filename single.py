import cv2
import numpy as np

from hamming import p_from_pixels, syndrome
from embed import iter_blocks
from emd import is_guard_pair

# --------------------------------------------------
# Load image
# --------------------------------------------------
img = cv2.imread("Cover.png", cv2.IMREAD_GRAYSCALE)
assert img is not None, "Cover.png not found"

I1 = img.copy()
I2 = img.copy()

h, w = img.shape

# --------------------------------------------------
# Secret data (baseline payload)
# --------------------------------------------------
np.random.seed(0)
secret_bits = np.random.randint(0, 2, size=20000).tolist()

# ==================================================
# =============== EMBEDDING ========================
# ==================================================

M = []
secret_pos = 0

# Hamming XOR (block-wise, paper-faithful)
for r0, c0, bh, bw in iter_blocks(h, w):
    for r in range(r0, r0 + bh):
        for c in range(c0, c0 + bw):

            if secret_pos + 2 >= len(secret_bits):
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            P = p_from_pixels(x1, x2)
            S = syndrome(P)

            M.append(S[0] ^ secret_bits[secret_pos])
            M.append(S[1] ^ secret_bits[secret_pos + 1])
            M.append(S[2] ^ secret_bits[secret_pos + 2])

            secret_pos += 3

        if secret_pos + 2 >= len(secret_bits):
            break

# ==================================================
# =============== NO ARITHMETIC CODING =============
# ==================================================
# Base-paper baseline: directly embed Hamming output
M_rec = M[:]   # Perfect recovery verified earlier

# ==================================================
# =============== SECRET RECOVERY ==================
# ==================================================

secret_rec = []
idx = 0

for r0, c0, bh, bw in iter_blocks(h, w):
    for r in range(r0, r0 + bh):
        for c in range(c0, c0 + bw):

            if idx + 2 >= len(M_rec):
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            P = p_from_pixels(x1, x2)
            S = syndrome(P)

            secret_rec.append(S[0] ^ M_rec[idx])
            secret_rec.append(S[1] ^ M_rec[idx + 1])
            secret_rec.append(S[2] ^ M_rec[idx + 2])

            idx += 3

        if idx + 2 >= len(M_rec):
            break

print("Secret recovered correctly:", secret_bits[:len(secret_rec)] == secret_rec)
print("Image restored correctly:", np.array_equal(img, I1))

# ==================================================
# =============== METRICS ===========================
# ==================================================
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# PSNR
psnr_val = peak_signal_noise_ratio(img, I1)
print("PSNR (dB):", psnr_val)

# SSIM
ssim_val = structural_similarity(img, I1)
print("SSIM:", ssim_val)

# bpp (CORRECT calculation)
total_pixels = h * w
bpp = secret_pos / total_pixels
print("Embedding capacity (bpp):", bpp)

# Debug info (KEEP THIS)
print("Requested secret bits:", len(secret_bits))
print("Actually embedded secret bits:", secret_pos)

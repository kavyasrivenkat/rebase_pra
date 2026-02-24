import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from hamming import p_from_pixels, syndrome
from embed import iter_blocks, embed_bits_into_block
from extract import extract_bits_from_block, restore_pairs_after_payload
from emd import is_guard_pair

# ==================================================
# CONFIG
# ==================================================
DATASET_DIR = "dataset"

results = []

print("Processing dataset (MAXIMUM EMD CAPACITY)...\n")

# ==================================================
# PROCESS EACH IMAGE
# ==================================================
for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.lower().endswith((".png", ".bmp", ".tif", ".jpg")):
        continue

    path = os.path.join(DATASET_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    total_pixels = h * w

    # --------------------------------------------------
    # COUNT VALID EMD PAIRS
    # --------------------------------------------------
    valid_pairs = 0
    for r0, c0, bh, bw in iter_blocks(h, w):
        for r in range(r0, r0 + bh):
            for c in range(c0, c0 + bw):
                x = int(img[r, c])
                if not is_guard_pair(x, x):
                    valid_pairs += 1

    # --------------------------------------------------
    # MAXIMUM PAYLOAD (2 bits per valid pair)
    # --------------------------------------------------
    max_payload_bits = int(1.5 * valid_pairs)

    # --------------------------------------------------
    # GENERATE LONG RANDOM PAYLOAD
    # --------------------------------------------------
    np.random.seed(0)
    secret_bits = np.random.randint(0, 2, size=3 * max_payload_bits).tolist()

    I1 = img.copy()
    I2 = img.copy()

    # --------------------------------------------------
    # HAMMING XOR (PRODUCE LARGE M)
    # --------------------------------------------------
    M = []
    secret_pos = 0

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

                M.extend([
                    S[0] ^ secret_bits[secret_pos],
                    S[1] ^ secret_bits[secret_pos + 1],
                    S[2] ^ secret_bits[secret_pos + 2]
                ])
                secret_pos += 3

            if secret_pos + 2 >= len(secret_bits):
                break

    # --------------------------------------------------
    # HARD LIMIT TO TRUE MAX EMD CAPACITY
    # --------------------------------------------------
    M = M[:max_payload_bits]

    # --------------------------------------------------
    # EMD EMBEDDING (FULL CAPACITY)
    # --------------------------------------------------
    bit_pos = 0

    for r0, c0, bh, bw in iter_blocks(h, w):
        if bit_pos >= max_payload_bits:
            break

        bit_pos, _ = embed_bits_into_block(
            I1, I2,
            r0, c0,
            M, bit_pos,
            [], 0
        )

    stego1 = I1.copy()
    stego2 = I2.copy()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    psnr1 = peak_signal_noise_ratio(img, stego1)
    psnr2 = peak_signal_noise_ratio(img, stego2)
    psnr_avg = (psnr1 + psnr2) / 2.0

    ssim1 = structural_similarity(img, stego1)
    ssim2 = structural_similarity(img, stego2)
    ssim_avg = (ssim1 + ssim2) / 2.0

    # --------------------------------------------------
    # EXTRACTION + RESTORATION (VERIFY REVERSIBILITY)
    # --------------------------------------------------
    payload_out = []
    for r0, c0, bh, bw in iter_blocks(h, w):
        got_p, _ = extract_bits_from_block(
            I1, I2,
            r0, c0,
            10**9, payload_out,
            0, []
        )
        restore_pairs_after_payload(I1, I2, r0, c0, bh, bw, got_p)

    reversible = np.array_equal(img, I1)

    # --------------------------------------------------
    # ACTUAL EMBEDDED bpp
    # --------------------------------------------------
    actual_bpp = bit_pos / total_pixels

    results.append((
        fname,
        psnr1, psnr2, psnr_avg,
        ssim1, ssim2, ssim_avg,
        actual_bpp,
        reversible
    ))

    print(
        f"{fname:15s} | "
        f"PSNRavg: {psnr_avg:.2f} | "
        f"SSIMavg: {ssim_avg:.4f} | "
        f"bpp: {actual_bpp:.3f} | "
        f"Reversible: {reversible}"
    )

# ==================================================
# DATASET SUMMARY
# ==================================================
print("\n=== DATASET AVERAGE (MAX EMD CAPACITY) ===")

print(f"Avg PSNR1 : {np.mean([r[1] for r in results]):.2f}")
print(f"Avg PSNR2 : {np.mean([r[2] for r in results]):.2f}")
print(f"Avg PSNR  : {np.mean([r[3] for r in results]):.2f}")

print(f"Avg SSIM1 : {np.mean([r[4] for r in results]):.4f}")
print(f"Avg SSIM2 : {np.mean([r[5] for r in results]):.4f}")
print(f"Avg SSIM  : {np.mean([r[6] for r in results]):.4f}")

print(f"Avg bpp   : {np.mean([r[7] for r in results]):.3f}")

# ==================================================
# SAVE RESULTS
# ==================================================
with open("results_max_capacity.csv", "w") as f:
    f.write("Image,PSNR1,PSNR2,PSNRavg,SSIM1,SSIM2,SSIMavg,bpp,Reversible\n")
    for r in results:
        f.write(
            f"{r[0]},"
            f"{r[1]:.3f},{r[2]:.3f},{r[3]:.3f},"
            f"{r[4]:.5f},{r[5]:.5f},{r[6]:.5f},"
            f"{r[7]:.4f},{r[8]}\n"
        )

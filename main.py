# main.py
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from hamming import p_from_pixels, syndrome
from embed import iter_blocks, embed_bits_into_block
from extract import extract_bits_from_block, restore_pairs_after_payload
from emd import is_guard_pair
from arithmetic import encode_bits

# ================= CONFIG =================
DATASET_DIR = "dataset"
PAYLOAD_RATIO = 0.114   # ≈ 30,000 bits for 512×512

# ================= BIT HELPERS =================
def bytes_to_bits(data: bytes):
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return bits

# ================= PROCESS DATASET =================
results = []
print("Processing dataset...\n")

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.lower().endswith((".png", ".bmp", ".tif", ".jpg")):
        continue

    img = cv2.imread(os.path.join(DATASET_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    total_pixels = h * w
    payload_bits = int(PAYLOAD_RATIO * total_pixels)

    np.random.seed(0)
    secret_bits = np.random.randint(0, 2, payload_bits).tolist()

    I1 = img.copy()
    I2 = img.copy()

    # =====================================================
    # 1. BUILD GLOBAL M (PAYLOAD-MATCHED)
    # =====================================================
    M = []
    secret_pos = 0

    for r0, c0, bh, bw in iter_blocks(h, w):
        for r in range(r0, r0 + bh):
            for c in range(c0, c0 + bw):

                if secret_pos >= payload_bits:
                    break

                x1 = int(I1[r, c])
                x2 = int(I2[r, c])
                if is_guard_pair(x1, x2):
                    continue

                P = p_from_pixels(x1, x2)
                S = syndrome(P)  # 3 bits

                # how many payload bits are still needed?
                remaining = payload_bits - secret_pos
                take = min(3, remaining)

                for k in range(take):
                    M.append(S[k] ^ secret_bits[secret_pos])
                    secret_pos += 1

            if secret_pos >= payload_bits:
                break
        if secret_pos >= payload_bits:
            break

    # =====================================================
    # 2. GLOBAL ARITHMETIC CODING
    # =====================================================
    compressed_bytes = encode_bits(M)
    Dcomp = bytes_to_bits(compressed_bytes)

    # =====================================================
    # 3. SEQUENTIAL BLOCK-WISE EMD EMBEDDING
    # =====================================================
    bit_pos = 0
    block_index = 0

    for r0, c0, bh, bw in iter_blocks(h, w):
        if bit_pos >= len(Dcomp):
            break

        swap = (block_index % 2 == 1)

        bit_pos, _ = embed_bits_into_block(
            I1, I2,
            r0, c0,
            Dcomp, bit_pos,
            [], 0,
            swap
        )

        block_index += 1

    stego1 = I1.copy()
    stego2 = I2.copy()

    # =====================================================
    # 4. METRICS
    # =====================================================
    psnr1 = peak_signal_noise_ratio(img, stego1)
    psnr2 = peak_signal_noise_ratio(img, stego2)
    psnr_avg = (psnr1 + psnr2) / 2.0

    ssim1 = structural_similarity(img, stego1)
    ssim2 = structural_similarity(img, stego2)
    ssim_avg = (ssim1 + ssim2) / 2.0

    # =====================================================
    # 5. EXTRACTION + RESTORATION
    # =====================================================
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
    bpp = payload_bits / total_pixels

    results.append((psnr1, psnr2, psnr_avg, ssim_avg, bpp))

    print(
        f"{fname:15s} | "
        f"PSNR1: {psnr1:.2f} | "
        f"PSNR2: {psnr2:.2f} | "
        f"PSNRavg: {psnr_avg:.2f} | "
        f"bpp: {bpp:.3f} | "
        f"Reversible: {reversible}"
    )

# ================= DATASET AVERAGE =================
print("\n=== DATASET AVERAGE ===")
print(f"Avg PSNR1  : {np.mean([r[0] for r in results]):.2f}")
print(f"Avg PSNR2  : {np.mean([r[1] for r in results]):.2f}")
print(f"Avg PSNR   : {np.mean([r[2] for r in results]):.2f}")
print(f"Avg SSIM   : {np.mean([r[3] for r in results]):.4f}")
print(f"Avg bpp    : {np.mean([r[4] for r in results]):.3f}")

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from bilateral import (
    bilateral_mean,
    bilateral_median,
    energy_normalize,
    compute_psnr,
    compute_l1,
    tonemap,
    exposure_scale,
)
from render_aov import load_aov


def filter_channel(color, aov, mode, radius, sigma_s, sigma_n, sigma_z):
    if mode == "mean":
        return bilateral_mean(color, aov["obj_id"], aov["depth"], aov["normal"],
                              radius=radius, sigma_s=sigma_s, sigma_n=sigma_n, sigma_z=sigma_z)
    elif mode == "median":
        return bilateral_median(color, aov["obj_id"], aov["depth"], aov["normal"],
                               radius=radius, sigma_n=sigma_n, sigma_z=sigma_z)
    raise ValueError(f"Unknown mode: {mode}")


def save_png(path: Path, ldr_image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(ldr_image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def colorize_obj_id(obj_id: np.ndarray) -> np.ndarray:
    """карта индексов объектов -> в различимые цвета"""
    palette = np.array([
        [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66],
        [0, 114, 178], [213, 94, 0], [204, 121, 167], [150, 150, 150],
        [120, 200, 120], [200, 120, 200],
    ], dtype=np.float64) / 255.0
    out = np.zeros(obj_id.shape + (3,), dtype=np.float64)
    for obj in np.unique(obj_id):
        if obj < 0:
            continue
        out[obj_id == obj] = palette[int(obj) % len(palette)]
    return out


def main():
    parser = argparse.ArgumentParser(description="Bilateral filtering of synthesized image")
    parser.add_argument("--aov", type=Path, required=True, help="noisy AOV NPZ")
    parser.add_argument("--reference", type=Path, default=None, help="reference AOV NPZ (high spp)")
    parser.add_argument("--mode", choices=["mean", "median"], default="mean")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--sigma-s", type=float, default=3.0)
    parser.add_argument("--sigma-n", type=float, default=0.3)
    parser.add_argument("--sigma-z", type=float, default=0.1)
    parser.add_argument("--split-direct-indirect", action="store_true", default=True,
                        help="filter direct and indirect separately (default)")
    parser.add_argument("--no-split", dest="split_direct_indirect", action="store_false")
    parser.add_argument("--energy-normalize", choices=["object", "none"], default="object")
    parser.add_argument("--output", type=Path, default=Path("outputs/filtered.png"))
    parser.add_argument("--gamma", type=float, default=2.2)
    args = parser.parse_args()

    aov = load_aov(args.aov)
    direct = aov["direct"]
    indirect = aov["indirect"]
    noisy_total = direct + indirect

    # ── Шаг 1-3: фильтрация ──
    if args.split_direct_indirect:
        filtered_direct = filter_channel(direct, aov, args.mode, args.radius,
                                         args.sigma_s, args.sigma_n, args.sigma_z)
        filtered_indirect = filter_channel(indirect, aov, args.mode, args.radius,
                                           args.sigma_s, args.sigma_n, args.sigma_z)
        filtered_total = filtered_direct + filtered_indirect
    else:
        filtered_total = filter_channel(noisy_total, aov, args.mode, args.radius,
                                        args.sigma_s, args.sigma_n, args.sigma_z)

    # ── Шаг 4: энергетическая нормировка по объектам ──
    if args.energy_normalize == "object":
        filtered_total, ratios = energy_normalize(filtered_total, noisy_total, aov["obj_id"])
        max_dev = max((abs(r - 1.0) for r in ratios.values()), default=0.0)
        print(f"Energy normalization: {len(ratios)} objects, max |ratio-1| = {max_dev:.2e}")

    # ── Шаг 5: метрики относительно эталона ──
    if args.reference is not None:
        ref = load_aov(args.reference)
        reference_total = ref["direct"] + ref["indirect"]
        scale = exposure_scale(reference_total)

        noisy_ldr = tonemap(noisy_total, scale, args.gamma)
        filtered_ldr = tonemap(filtered_total, scale, args.gamma)
        reference_ldr = tonemap(reference_total, scale, args.gamma)

        psnr_noisy = compute_psnr(noisy_ldr, reference_ldr)
        psnr_filtered = compute_psnr(filtered_ldr, reference_ldr)
        l1_noisy = compute_l1(noisy_ldr, reference_ldr)
        l1_filtered = compute_l1(filtered_ldr, reference_ldr)

        print(f"PSNR noisy vs reference: {psnr_noisy:.2f} dB,  L1 = {l1_noisy:.5f}")
        print(f"PSNR filtered vs reference: {psnr_filtered:.2f} dB,  L1 = {l1_filtered:.5f}")
        if l1_filtered > 1e-12:
            print(f"L1 improvement: x{l1_noisy / l1_filtered:.2f}")

        save_png(args.output.with_name(args.output.stem + "_reference.png"), reference_ldr)
        save_png(args.output.with_name(args.output.stem + "_noisy.png"), noisy_ldr)
    else:
        scale = exposure_scale(noisy_total)
        filtered_ldr = tonemap(filtered_total, scale, args.gamma)

    save_png(args.output, filtered_ldr)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bilateral import (
    make_spatial_kernel,
    bilateral_mean,
    bilateral_median,
    energy_normalize,
    compute_mse,
    compute_psnr,
    compute_l1,
    tonemap,
    exposure_scale,
    _range_weight,
)
import render_aov 


LAB_DIR = Path(__file__).resolve().parent


def flat_geometry(height, width, obj_value=0):
    obj_id = np.full((height, width), obj_value, dtype=np.int32)
    depth = np.ones((height, width), dtype=np.float64)
    normal = np.zeros((height, width, 3), dtype=np.float64)
    normal[:, :, 1] = 1.0
    return obj_id, depth, normal


def test_spatial_kernel_sums_to_one():
    kernel = make_spatial_kernel(radius=3, sigma_s=2.0)
    assert kernel.shape == (7, 7)
    assert abs(kernel.sum() - 1.0) < 1e-12


def test_spatial_kernel_is_symmetric():
    kernel = make_spatial_kernel(radius=2, sigma_s=1.5)
    assert np.allclose(kernel, kernel[::-1, :])
    assert np.allclose(kernel, kernel[:, ::-1])


def test_spatial_kernel_peak_at_center():
    kernel = make_spatial_kernel(radius=3, sigma_s=2.0)
    center = kernel[3, 3]
    assert center == kernel.max()


def test_mean_preserves_constant_image():
    h, w = 16, 16
    obj_id, depth, normal = flat_geometry(h, w)
    color = np.full((h, w, 3), 0.5, dtype=np.float64)
    out = bilateral_mean(color, obj_id, depth, normal, radius=3)
    assert np.allclose(out, color, atol=1e-9)


def test_mean_reduces_noise_variance():
    rng = np.random.default_rng(0)
    h, w = 64, 64
    obj_id, depth, normal = flat_geometry(h, w)
    clean = np.full((h, w, 3), 0.5)
    noisy = clean + rng.normal(0, 0.1, size=clean.shape)
    out = bilateral_mean(noisy, obj_id, depth, normal, radius=3, sigma_s=3.0)
    assert out.var() < noisy.var()


def test_mean_does_not_blur_across_object_boundary():
    h, w = 16, 16
    obj_id = np.zeros((h, w), dtype=np.int32)
    obj_id[:, w // 2:] = 1
    depth = np.ones((h, w))
    normal = np.zeros((h, w, 3)); normal[:, :, 1] = 1.0
    color = np.zeros((h, w, 3))
    color[:, :w // 2] = 0.2
    color[:, w // 2:] = 0.8
    out = bilateral_mean(color, obj_id, depth, normal, radius=3)
    assert np.allclose(out[:, 0], 0.2, atol=1e-9)
    assert np.allclose(out[:, -1], 0.8, atol=1e-9)
    assert np.allclose(out[:, w // 2 - 1], 0.2, atol=1e-9)
    assert np.allclose(out[:, w // 2], 0.8, atol=1e-9)


def test_mean_respects_depth_discontinuity():
    h, w = 16, 16
    obj_id = np.zeros((h, w), dtype=np.int32)  # один объект
    depth = np.ones((h, w))
    depth[:, w // 2:] = 100.0  # резкий скачок глубины
    normal = np.zeros((h, w, 3)); normal[:, :, 1] = 1.0
    color = np.zeros((h, w, 3))
    color[:, :w // 2] = 0.2
    color[:, w // 2:] = 0.8
    out = bilateral_mean(color, obj_id, depth, normal, radius=3, sigma_z=0.1)
    assert np.allclose(out[:, w // 2 - 1], 0.2, atol=1e-6)
    assert np.allclose(out[:, w // 2], 0.8, atol=1e-6)


def test_mean_respects_normal_discontinuity():
    h, w = 16, 16
    obj_id = np.zeros((h, w), dtype=np.int32)
    depth = np.ones((h, w))
    normal = np.zeros((h, w, 3))
    normal[:, :w // 2, 1] = 1.0   # нормаль вверх
    normal[:, w // 2:, 0] = 1.0   # нормаль вбок (90°)
    color = np.zeros((h, w, 3))
    color[:, :w // 2] = 0.2
    color[:, w // 2:] = 0.8
    out = bilateral_mean(color, obj_id, depth, normal, radius=3, sigma_n=0.3)
    assert np.allclose(out[:, w // 2 - 1], 0.2, atol=1e-4)
    assert np.allclose(out[:, w // 2], 0.8, atol=1e-4)


def test_mean_leaves_background_untouched():
    h, w = 8, 8
    obj_id = np.full((h, w), -1, dtype=np.int32)  # весь фон
    depth = np.zeros((h, w))
    normal = np.zeros((h, w, 3))
    color = np.random.default_rng(1).random((h, w, 3))
    out = bilateral_mean(color, obj_id, depth, normal, radius=2)
    assert np.allclose(out, color)

def test_median_preserves_constant_image():
    h, w = 16, 16
    obj_id, depth, normal = flat_geometry(h, w)
    color = np.full((h, w, 3), 0.3)
    out = bilateral_median(color, obj_id, depth, normal, radius=2)
    assert np.allclose(out, color, atol=1e-6)


def test_median_removes_impulse_noise():
    h, w = 16, 16
    obj_id, depth, normal = flat_geometry(h, w)
    color = np.full((h, w, 3), 0.3)
    color[8, 8] = 1.0  # импульс «соль»
    out = bilateral_median(color, obj_id, depth, normal, radius=2)
    assert out[8, 8, 0] < 0.5  # импульс подавлен


def test_median_does_not_introduce_new_values():
    h, w = 12, 12
    obj_id, depth, normal = flat_geometry(h, w)
    rng = np.random.default_rng(2)
    values = np.array([0.1, 0.4, 0.7])
    color = values[rng.integers(0, 3, size=(h, w))][:, :, None] * np.ones(3)
    out = bilateral_median(color, obj_id, depth, normal, radius=1)
    assert np.all(np.isin(np.round(out, 6), np.round(values, 6)))


def test_median_does_not_cross_object_boundary():
    h, w = 16, 16
    obj_id = np.zeros((h, w), dtype=np.int32)
    obj_id[:, w // 2:] = 1
    depth = np.ones((h, w))
    normal = np.zeros((h, w, 3)); normal[:, :, 1] = 1.0
    color = np.zeros((h, w, 3))
    color[:, :w // 2] = 0.2
    color[:, w // 2:] = 0.8
    out = bilateral_median(color, obj_id, depth, normal, radius=3)
    assert np.allclose(out[:, w // 2 - 1], 0.2, atol=1e-6)
    assert np.allclose(out[:, w // 2], 0.8, atol=1e-6)


def test_energy_normalize_preserves_per_object_sum():
    h, w = 16, 16
    obj_id = np.zeros((h, w), dtype=np.int32)
    obj_id[:, w // 2:] = 1
    original = np.random.default_rng(3).random((h, w, 3))
    filtered = original * 0.5 + 0.1
    out, ratios = energy_normalize(filtered, original, obj_id)
    for obj in (0, 1):
        mask = obj_id == obj
        assert np.allclose(out[mask].sum(axis=0), original[mask].sum(axis=0), atol=1e-8)
    # Отношения суммарной яркости после нормировки близки к 1
    for r in ratios.values():
        assert abs(r - 1.0) < 1e-6


def test_energy_normalize_ignores_background():
    h, w = 8, 8
    obj_id = np.full((h, w), -1, dtype=np.int32)
    obj_id[0, 0] = 0
    original = np.ones((h, w, 3))
    filtered = np.full((h, w, 3), 2.0)
    out, ratios = energy_normalize(filtered, original, obj_id)
    assert list(ratios.keys()) == [0]
    assert np.allclose(out[0, 0], original[0, 0])


def test_psnr_of_identical_is_infinite():
    img = np.random.default_rng(4).random((8, 8, 3))
    assert compute_psnr(img, img) == float("inf")
    assert compute_l1(img, img) == 0.0
    assert compute_mse(img, img) == 0.0


def test_psnr_decreases_with_error():
    ref = np.full((8, 8, 3), 0.5)
    small = ref + 0.01
    large = ref + 0.1
    assert compute_psnr(small, ref) > compute_psnr(large, ref)


def test_l1_matches_manual():
    a = np.zeros((4, 4, 3))
    b = np.full((4, 4, 3), 0.25)
    assert abs(compute_l1(a, b) - 0.25) < 1e-12


def test_tonemap_in_range():
    radiance = np.array([[[0.0, 5.0, 100.0]]])
    out = tonemap(radiance, scale=0.1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_exposure_scale_normalizes_max():
    radiance = np.array([[[0.0, 2.0, 4.0]]])
    scale = exposure_scale(radiance)
    assert abs(scale * 4.0 - 1.0) < 1e-12


def test_range_weight_zero_across_objects():
    obj = np.array([[0]])
    nb = np.array([[1]])
    depth = np.array([[1.0]])
    normal = np.array([[[0.0, 1.0, 0.0]]])
    gr = _range_weight(obj, depth, normal, nb, depth, normal, 0.3, 0.1)
    assert gr[0, 0] == 0.0


def test_range_weight_one_for_identical():
    obj = np.array([[0]])
    depth = np.array([[1.0]])
    normal = np.array([[[0.0, 1.0, 0.0]]])
    gr = _range_weight(obj, depth, normal, obj, depth, normal, 0.3, 0.1)
    assert abs(gr[0, 0] - 1.0) < 1e-12


def test_aov_save_load_roundtrip(tmp_path):
    aov = {
        "direct": np.random.default_rng(5).random((4, 4, 3)),
        "indirect": np.random.default_rng(6).random((4, 4, 3)),
        "depth": np.random.default_rng(7).random((4, 4)),
        "obj_id": np.random.default_rng(8).integers(-1, 5, size=(4, 4)).astype(np.int32),
        "normal": np.random.default_rng(9).random((4, 4, 3)),
    }
    path = tmp_path / "aov.npz"
    render_aov.save_aov(path, aov)
    loaded = render_aov.load_aov(path)
    for key in aov:
        assert np.allclose(loaded[key], aov[key])


def test_build_scene_aov_has_lights_and_objects():
    scene, object_id = render_aov.build_scene_aov()
    assert len(scene.light_ids) >= 1
    assert len(object_id) == len(scene.triangles)
    # Все 8 объектов присутствуют
    assert set(object_id) >= set(range(8))


def test_render_aov_small_produces_channels():
    scene, object_id = render_aov.build_scene_aov()
    aov = render_aov.render_aov(scene, object_id, width=8, height=8,
                                samples=1, max_depth=2, num_workers=1)
    assert aov["direct"].shape == (8, 8, 3)
    assert aov["obj_id"].shape == (8, 8)
    assert aov["normal"].shape == (8, 8, 3)
    # Хотя бы часть пикселей видит сцену (obj_id >= 0)
    assert np.any(aov["obj_id"] >= 0)
    assert np.all(aov["direct"] >= 0.0)


def test_filtering_reduces_l1_vs_reference():
    scene, object_id = render_aov.build_scene_aov()
    noisy = render_aov.render_aov(scene, object_id, 24, 24, samples=2, max_depth=3, num_workers=1)
    reference = render_aov.render_aov(scene, object_id, 24, 24, samples=16, max_depth=4, num_workers=1)

    noisy_total = noisy["direct"] + noisy["indirect"]
    fd = bilateral_mean(noisy["direct"], noisy["obj_id"], noisy["depth"], noisy["normal"], radius=3)
    fi = bilateral_mean(noisy["indirect"], noisy["obj_id"], noisy["depth"], noisy["normal"], radius=3)
    filtered_total, _ = energy_normalize(fd + fi, noisy_total, noisy["obj_id"])

    reference_total = reference["direct"] + reference["indirect"]
    scale = exposure_scale(reference_total)
    l1_noisy = compute_l1(tonemap(noisy_total, scale), tonemap(reference_total, scale))
    l1_filtered = compute_l1(tonemap(filtered_total, scale), tonemap(reference_total, scale))
    assert l1_filtered < l1_noisy


def test_cli_smoke(tmp_path):
    # Маленький AOV для скорости
    scene, object_id = render_aov.build_scene_aov()
    aov = render_aov.render_aov(scene, object_id, 16, 16, samples=1, max_depth=2, num_workers=1)
    aov_path = tmp_path / "aov.npz"
    render_aov.save_aov(aov_path, aov)
    out_path = tmp_path / "filtered.png"

    result = subprocess.run(
        [sys.executable, str(LAB_DIR / "main.py"),
         "--aov", str(aov_path), "--mode", "mean", "--radius", "2",
         "--output", str(out_path)],
        cwd=str(LAB_DIR), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

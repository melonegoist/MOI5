import math
from typing import Dict, Tuple

import numpy as np


def make_spatial_kernel(radius: int, sigma_s: float) -> np.ndarray:
    """нормированное гауссово пространственное ядро (2r+1)×(2r+1).

    Ядро нормировано так, что сумма всех его элементов равна 1
    (условие Σ Gs(p-q) = 1)
    """
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_s ** 2))
    kernel /= kernel.sum()
    return kernel


def _pad_edge(arr: np.ndarray, radius: int) -> np.ndarray:
    """Дополнение массива на radius пикселей с каждой стороны краевыми значениями"""
    if arr.ndim == 2:
        pad = ((radius, radius), (radius, radius))
    else:
        pad = ((radius, radius), (radius, radius)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, pad, mode="edge")


def _range_weight(
    obj_id: np.ndarray, depth: np.ndarray, normal: np.ndarray,
    nb_obj: np.ndarray, nb_depth: np.ndarray, nb_normal: np.ndarray,
    sigma_n: float, sigma_z: float,
) -> np.ndarray:
    """Вычисление диапазонного веса Gr(p,q) = w_obj · w_n · w_z для всего изображения"""
    # Жёсткий вес по объекту: 1 если объекты совпадают (и оба валидны)
    w_obj = ((obj_id == nb_obj) & (obj_id >= 0) & (nb_obj >= 0)).astype(np.float64)
    # Мягкий вес по нормали
    ndot = np.clip(np.sum(normal * nb_normal, axis=2), 0.0, 1.0)
    w_n = np.exp(-(1.0 - ndot) / (sigma_n ** 2))
    # Мягкий вес по глубине
    w_z = np.exp(-((depth - nb_depth) ** 2) / (2.0 * sigma_z ** 2))
    return w_obj * w_n * w_z


def bilateral_mean(
    color: np.ndarray, obj_id: np.ndarray, depth: np.ndarray, normal: np.ndarray,
    radius: int = 3, sigma_s: float = 3.0, sigma_n: float = 0.3, sigma_z: float = 0.1,
) -> np.ndarray:
    """Билатеральная фильтрация взвешенным средним.

    g(p) = (1/Wp) Σ_q f(q) · Gs(p-q) · Gr(p,q),   Wp = Σ_q Gs(p-q) · Gr(p,q).

    Сохраняет границы объектов за счёт диапазонного веса Gr.
    Фоновые пиксели (obj_id < 0) остаются без изменений.
    """
    height, width = obj_id.shape
    spatial = make_spatial_kernel(radius, sigma_s)

    padded_obj = _pad_edge(obj_id, radius)
    padded_depth = _pad_edge(depth, radius)
    padded_normal = _pad_edge(normal, radius)
    padded_color = _pad_edge(color, radius)

    accum = np.zeros_like(color, dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gs = spatial[dy + radius, dx + radius]
            ys, xs = radius + dy, radius + dx
            nb_obj = padded_obj[ys:ys + height, xs:xs + width]
            nb_depth = padded_depth[ys:ys + height, xs:xs + width]
            nb_normal = padded_normal[ys:ys + height, xs:xs + width]
            nb_color = padded_color[ys:ys + height, xs:xs + width]

            gr = _range_weight(obj_id, depth, normal, nb_obj, nb_depth, nb_normal,
                               sigma_n, sigma_z)
            w = gs * gr
            accum += w[:, :, None] * nb_color
            weight_sum += w

    safe = weight_sum > 1e-12
    out = color.copy()
    out[safe] = accum[safe] / weight_sum[safe, None]
    
    background = obj_id < 0
    out[background] = color[background]
    return out

def bilateral_median(
    color: np.ndarray, obj_id: np.ndarray, depth: np.ndarray, normal: np.ndarray,
    radius: int = 3, sigma_n: float = 0.3, sigma_z: float = 0.1,
) -> np.ndarray:
    """Медианная билатеральная фильтрация

    Медиана не вносит новых значений яркости и устойчива к импульсному шуму
    После медианной фильтрации необходима энергетическая нормировка
    """
    height, width = obj_id.shape
    padded_obj = _pad_edge(obj_id, radius)
    padded_color = _pad_edge(color, radius)

    offsets = [(dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    num = len(offsets)

    stack = np.full((num, height, width, color.shape[2]), np.nan, dtype=np.float32)
    for k, (dy, dx) in enumerate(offsets):
        ys, xs = radius + dy, radius + dx
        nb_obj = padded_obj[ys:ys + height, xs:xs + width]
        nb_color = padded_color[ys:ys + height, xs:xs + width]
        same_object = (nb_obj == obj_id) & (obj_id >= 0) & (nb_obj >= 0)
        masked = np.where(same_object[:, :, None], nb_color.astype(np.float32), np.nan)
        stack[k] = masked

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0)

    out = color.copy()
    valid = ~np.isnan(median).any(axis=2)
    out[valid] = median[valid].astype(color.dtype)
    background = obj_id < 0
    out[background] = color[background]
    return out


def energy_normalize(
    filtered: np.ndarray, original: np.ndarray, obj_id: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Нормирует яркость так, чтобы для каждого объекта O суммарная яркость
    после фильтрации равнялась суммарной яркости до фильтрации:

        Σ_{p∈O} g_norm(p) = Σ_{p∈O} f(p).

    Это сохраняет локальную и глобальную яркость (физическая корректность).
    Нормировка выполняется покомпонентно (по каждому цветовому каналу)
    """
    out = filtered.copy().astype(np.float64)
    ratios: Dict[int, float] = {}

    for obj in np.unique(obj_id):
        if obj < 0:
            continue
        mask = obj_id == obj
        original_sum = original[mask].sum(axis=0)   
        filtered_sum = filtered[mask].sum(axis=0)    
        scale = np.ones(3, dtype=np.float64)
        for channel in range(3):
            if filtered_sum[channel] > 1e-12:
                scale[channel] = original_sum[channel] / filtered_sum[channel]
        out[mask] = filtered[mask] * scale

        lum_orig = float(np.sum(original[mask]))
        lum_norm = float(np.sum(out[mask]))
        ratios[int(obj)] = lum_norm / lum_orig if lum_orig > 1e-12 else 1.0

    return out, ratios


def compute_mse(image: np.ndarray, reference: np.ndarray) -> float:
    diff = image.astype(np.float64) - reference.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(image: np.ndarray, reference: np.ndarray, max_value: float = 1.0) -> float:
    """Пиковое отношение сигнал/шум (дБ). Изображения ожидаются в [0, max_value]"""
    mse = compute_mse(image, reference)
    if mse < 1e-20:
        return float("inf")
    return 10.0 * math.log10((max_value ** 2) / mse)


def compute_l1(image: np.ndarray, reference: np.ndarray) -> float:
    """Средняя абсолютная ошибка (L1) между двумя изображениями."""
    return float(np.mean(np.abs(image.astype(np.float64) - reference.astype(np.float64))))


def tonemap(radiance: np.ndarray, scale: float, gamma: float = 2.2) -> np.ndarray:
    """Приводит HDR-яркость к [0,1]: нормировка на общий масштаб + гамма-коррекция."""
    normalized = np.clip(radiance * scale, 0.0, 1.0)
    return np.power(normalized, 1.0 / gamma)


def exposure_scale(reference_radiance: np.ndarray) -> float:
    """Общий масштаб экспозиции — обратный максимум яркости эталона.
    """
    max_value = float(np.max(reference_radiance))
    return 1.0 / max_value if max_value > 1e-12 else 1.0

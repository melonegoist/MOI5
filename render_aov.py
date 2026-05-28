import argparse
import math
import multiprocessing
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from path_tracer import (
    Camera,
    Material,
    Scene,
    Triangle,
    Vec3,
    EPS,
    estimate_direct_light,
    intersect_scene,
    reflect,
    sample_cosine_hemisphere,
)


def build_scene_aov() -> Tuple[Scene, List[int]]:
    materials = [
        Material(kd=Vec3(0.75, 0.75, 0.75), ks=Vec3(0.0, 0.0, 0.0)),  # 0 floor
        Material(kd=Vec3(0.75, 0.75, 0.75), ks=Vec3(0.0, 0.0, 0.0)),  # 1 ceiling
        Material(kd=Vec3(0.75, 0.75, 0.75), ks=Vec3(0.0, 0.0, 0.0)),  # 2 back
        Material(kd=Vec3(0.75, 0.15, 0.15), ks=Vec3(0.0, 0.0, 0.0)),  # 3 left (red)
        Material(kd=Vec3(0.15, 0.60, 0.20), ks=Vec3(0.0, 0.0, 0.0)),  # 4 right (green)
        Material(kd=Vec3(0.0, 0.0, 0.0), ks=Vec3(0.0, 0.0, 0.0)),     # 5 light
        Material(kd=Vec3(0.45, 0.45, 0.45), ks=Vec3(0.0, 0.0, 0.0)),  # 6 cube
        Material(kd=Vec3(0.45, 0.45, 0.45), ks=Vec3(0.0, 0.0, 0.0)),  # 7 pyramid
    ]

    triangles: List[Triangle] = []
    triangle_object_id: List[int] = []
    light_ids: List[int] = []

    def add_quad(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3,
                 material_id: int, object_id: int,
                 emission: Vec3 = Vec3(0.0, 0.0, 0.0)):
        triangles.append(Triangle(v0, v1, v2, material_id, emission))
        triangle_object_id.append(object_id)
        triangles.append(Triangle(v0, v2, v3, material_id, emission))
        triangle_object_id.append(object_id)

    add_quad(Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(1, 0, -3), Vec3(-1, 0, -3), 0, 0)   # floor
    add_quad(Vec3(-1, 2, -1), Vec3(-1, 2, -3), Vec3(1, 2, -3), Vec3(1, 2, -1), 1, 1)   # ceiling
    add_quad(Vec3(-1, 0, -3), Vec3(1, 0, -3), Vec3(1, 2, -3), Vec3(-1, 2, -3), 2, 2)   # back
    add_quad(Vec3(-1, 0, -1), Vec3(-1, 0, -3), Vec3(-1, 2, -3), Vec3(-1, 2, -1), 3, 3) # left (red)
    add_quad(Vec3(1, 0, -3), Vec3(1, 0, -1), Vec3(1, 2, -1), Vec3(1, 2, -3), 4, 4)     # right (green)

    # Площадной источник света на потолке
    light_emission = Vec3(18.0, 16.0, 13.0)
    lc0 = Vec3(-0.35, 1.99, -1.6)
    lc1 = Vec3(0.35, 1.99, -1.6)
    lc2 = Vec3(0.35, 1.99, -2.3)
    lc3 = Vec3(-0.35, 1.99, -2.3)
    triangles.append(Triangle(lc0, lc2, lc1, 5, light_emission))
    triangle_object_id.append(5)
    light_ids.append(len(triangles) - 1)
    triangles.append(Triangle(lc0, lc3, lc2, 5, light_emission))
    triangle_object_id.append(5)
    light_ids.append(len(triangles) - 1)

    # Куб справа
    cx, cz, s = 0.45, -2.1, 0.32
    a = Vec3(cx - s, 0, cz - s); b = Vec3(cx + s, 0, cz - s)
    c = Vec3(cx + s, 0, cz + s); d = Vec3(cx - s, 0, cz + s)
    e = Vec3(cx - s, 2 * s, cz - s); f = Vec3(cx + s, 2 * s, cz - s)
    g = Vec3(cx + s, 2 * s, cz + s); h = Vec3(cx - s, 2 * s, cz + s)
    add_quad(a, b, f, e, 6, 6)
    add_quad(b, c, g, f, 6, 6)
    add_quad(c, d, h, g, 6, 6)
    add_quad(d, a, e, h, 6, 6)
    add_quad(e, f, g, h, 6, 6)

    # Пирамида слева
    p0 = Vec3(-0.55, 0.0, -2.4); p1 = Vec3(-0.15, 0.0, -2.4)
    p2 = Vec3(-0.15, 0.0, -2.0); p3 = Vec3(-0.55, 0.0, -2.0)
    apex = Vec3(-0.35, 0.7, -2.2)
    no_em = Vec3(0.0, 0.0, 0.0)
    triangles.append(Triangle(p0, p1, p2, 7, no_em)); triangle_object_id.append(7)
    triangles.append(Triangle(p0, p2, p3, 7, no_em)); triangle_object_id.append(7)
    for base_a, base_b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
        triangles.append(Triangle(base_a, base_b, apex, 7, no_em))
        triangle_object_id.append(7)

    camera = Camera.look_at(
        origin=Vec3(0.0, 1.0, 0.4),
        target=Vec3(0.0, 1.0, -2.0),
        up_hint=Vec3(0.0, 1.0, 0.0),
        fov_deg=50.0,
        aspect=1.0,
    )
    scene = Scene(materials=materials, triangles=triangles, light_ids=light_ids, camera=camera)
    return scene, triangle_object_id


def trace_path_split(
    primary_ray_origin: Vec3,
    primary_ray_direction: Vec3,
    materials: Sequence[Material],
    triangles: Sequence[Triangle],
    light_ids: Sequence[int],
    max_bounces: int,
) -> Tuple[Vec3, Vec3]:
    """Трассирует один путь

    Прямая яркость L_d — вклад, пойманный на первой поверхности

    Вторичная яркость L_i — всё, что приходит после первого отскока
    (многократные переотражения, NEE на последующих поверхностях,
    эмиссия источников, в которые попал луч после отражений).
    """
    direct = Vec3(0.0, 0.0, 0.0)
    indirect = Vec3(0.0, 0.0, 0.0)
    throughput = Vec3(1.0, 1.0, 1.0)
    current_origin = primary_ray_origin
    current_direction = primary_ray_direction

    for bounce in range(max_bounces):
        hit = intersect_scene(current_origin, current_direction, triangles)
        if hit is None:
            break
        hit_triangle = triangles[hit.tri_id]
        hit_material = materials[hit_triangle.material_id]

        is_first = bounce == 0

        # Попали в источник
        if hit_triangle.emission.max_component() > 0.0:
            contribution = throughput * hit_triangle.emission
            if is_first:
                direct += contribution
            else:
                indirect += contribution
            break

        # Прямое освещение (NEE)
        nee = estimate_direct_light(hit, triangles, light_ids, throughput, hit_material.kd)
        if is_first:
            direct += nee
        else:
            indirect += nee

        # Русская рулетка (после первого отскока)
        if bounce >= 1:
            survival_probability = min(0.95, max(0.05, throughput.max_component()))
            if random.random() > survival_probability:
                break
            throughput = throughput / survival_probability

        # Выбор типа отражения (выборка по значимости BSDF)
        weight_diffuse = hit_material.kd.luminance()
        weight_specular = hit_material.ks.luminance()
        total_weight = weight_diffuse + weight_specular
        if total_weight <= EPS:
            break
        probability_diffuse = weight_diffuse / total_weight

        if random.random() < probability_diffuse:
            new_direction, pdf = sample_cosine_hemisphere(hit.normal)
            cos_outgoing = max(0.0, hit.normal.dot(new_direction))
            lambertian_brdf = hit_material.kd * (1.0 / math.pi)
            throughput = throughput * lambertian_brdf * (cos_outgoing / max(EPS, pdf)) / max(EPS, probability_diffuse)
        else:
            new_direction = reflect(current_direction, hit.normal)
            probability_specular = 1.0 - probability_diffuse
            throughput = throughput * (hit_material.ks / max(EPS, probability_specular))

        current_origin = hit.position + hit.normal * 1e-4
        current_direction = new_direction

    return direct, indirect

_worker_scene: Optional[Scene] = None
_worker_object_id: Optional[List[int]] = None
_worker_width: int = 0
_worker_height: int = 0
_worker_samples: int = 0
_worker_max_depth: int = 0


def _worker_init(scene, object_id, width, height, samples, max_depth):
    global _worker_scene, _worker_object_id
    global _worker_width, _worker_height, _worker_samples, _worker_max_depth
    _worker_scene = scene
    _worker_object_id = object_id
    _worker_width = width
    _worker_height = height
    _worker_samples = samples
    _worker_max_depth = max_depth


def _render_row(task):
    row_y, seed = task
    random.seed(seed)
    scene = _worker_scene
    object_id = _worker_object_id
    width = _worker_width
    height = _worker_height
    samples = _worker_samples
    max_depth = _worker_max_depth

    direct_row = np.zeros((width, 3), dtype=np.float64)
    indirect_row = np.zeros((width, 3), dtype=np.float64)
    depth_row = np.zeros(width, dtype=np.float64)
    objid_row = np.full(width, -1, dtype=np.int32)
    normal_row = np.zeros((width, 3), dtype=np.float64)

    for column_x in range(width):
        # G-buffer: первичный луч через центр пикселя
        center_u = (column_x + 0.5) / width
        center_v = (row_y + 0.5) / height
        center_dir = scene.camera.generate_ray(center_u, center_v)
        primary_hit = intersect_scene(scene.camera.origin, center_dir, scene.triangles)
        if primary_hit is not None:
            depth_row[column_x] = primary_hit.t
            objid_row[column_x] = object_id[primary_hit.tri_id]
            normal_row[column_x, 0] = primary_hit.normal.x
            normal_row[column_x, 1] = primary_hit.normal.y
            normal_row[column_x, 2] = primary_hit.normal.z

        # jitter-выборка антиалиасинг
        d_sum = Vec3(0.0, 0.0, 0.0)
        i_sum = Vec3(0.0, 0.0, 0.0)
        for _ in range(samples):
            u = (column_x + random.random()) / width
            v = (row_y + random.random()) / height
            ray_d = scene.camera.generate_ray(u, v)
            d, i = trace_path_split(
                scene.camera.origin, ray_d,
                scene.materials, scene.triangles, scene.light_ids, max_depth,
            )
            d_sum += d
            i_sum += i
        d_avg = d_sum / float(samples)
        i_avg = i_sum / float(samples)
        direct_row[column_x] = (d_avg.x, d_avg.y, d_avg.z)
        indirect_row[column_x] = (i_avg.x, i_avg.y, i_avg.z)

    return row_y, direct_row, indirect_row, depth_row, objid_row, normal_row


def render_aov(scene: Scene, object_id: List[int], width: int, height: int,
               samples: int, max_depth: int, num_workers: int = 0):
    scene.camera.half_width = scene.camera.half_height * (width / height)

    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, height)

    direct = np.zeros((height, width, 3), dtype=np.float64)
    indirect = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.zeros((height, width), dtype=np.float64)
    obj_id = np.full((height, width), -1, dtype=np.int32)
    normal = np.zeros((height, width, 3), dtype=np.float64)

    base_seed = random.randint(0, 2 ** 31)
    tasks = [(y, base_seed + y) for y in range(height)]

    progress_step = max(1, height // 20)
    completed = 0
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(scene, object_id, width, height, samples, max_depth),
    ) as pool:
        for y, d_row, i_row, z_row, o_row, n_row in pool.imap_unordered(_render_row, tasks, chunksize=4):
            direct[y] = d_row
            indirect[y] = i_row
            depth[y] = z_row
            obj_id[y] = o_row
            normal[y] = n_row
            completed += 1
            if completed % progress_step == 0 or completed == height:
                print(f"Progress: {completed}/{height} lines", flush=True)

    return {
        "direct": direct,
        "indirect": indirect,
        "depth": depth,
        "obj_id": obj_id,
        "normal": normal,
    }


def save_aov(path: Path, aov: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        direct=aov["direct"],
        indirect=aov["indirect"],
        depth=aov["depth"],
        obj_id=aov["obj_id"],
        normal=aov["normal"],
    )


def load_aov(path: Path) -> dict:
    data = np.load(path)
    return {
        "direct": data["direct"],
        "indirect": data["indirect"],
        "depth": data["depth"],
        "obj_id": data["obj_id"],
        "normal": data["normal"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="AOV path tracer for Lab 5")
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--samples", type=int, default=4, help="samples per pixel")
    parser.add_argument("--max-depth", type=int, default=5, help="maximum path depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/aov.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    scene, object_id = build_scene_aov()
    aov = render_aov(
        scene, object_id, args.width, args.height,
        args.samples, args.max_depth, args.workers,
    )
    save_aov(args.output, aov)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

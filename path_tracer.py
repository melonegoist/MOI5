import argparse
import math
import multiprocessing
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


EPS = 1e-6


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value):
        if isinstance(value, Vec3):
            return Vec3(self.x * value.x, self.y * value.y, self.z * value.z)
        return Vec3(self.x * value, self.y * value, self.z * value)

    __rmul__ = __mul__

    def __truediv__(self, value: float) -> "Vec3":
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n < EPS:
            return Vec3(0.0, 0.0, 0.0)
        return self / n

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def luminance(self) -> float:
        return 0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z

    def clamp01(self) -> "Vec3":
        return Vec3(
            max(0.0, min(1.0, self.x)),
            max(0.0, min(1.0, self.y)),
            max(0.0, min(1.0, self.z)),
        )


@dataclass
class Material:
    kd: Vec3
    ks: Vec3

    def __post_init__(self):
        self.kd = self.kd.clamp01()
        self.ks = self.ks.clamp01()
        total = Vec3(self.kd.x + self.ks.x, self.kd.y + self.ks.y, self.kd.z + self.ks.z)
        scale = max(total.x, total.y, total.z)
        if scale > 1.0:
            self.kd = self.kd / scale
            self.ks = self.ks / scale


@dataclass
class Triangle:
    a: Vec3
    b: Vec3
    c: Vec3
    material_id: int
    emission: Vec3

    def normal(self) -> Vec3:
        return (self.b - self.a).cross(self.c - self.a).normalized()

    def area(self) -> float:
        return 0.5 * (self.b - self.a).cross(self.c - self.a).norm()


@dataclass
class Hit:
    t: float
    position: Vec3
    normal: Vec3
    tri_id: int


@dataclass
class Camera:
    origin: Vec3
    forward: Vec3
    right: Vec3
    up: Vec3
    half_width: float
    half_height: float

    @staticmethod
    def look_at(origin: Vec3, target: Vec3, up_hint: Vec3, fov_deg: float, aspect: float) -> "Camera":
        forward = (target - origin).normalized()
        right = forward.cross(up_hint).normalized()
        up = right.cross(forward).normalized()
        half_height = math.tan(math.radians(fov_deg) * 0.5)
        half_width = half_height * aspect
        return Camera(origin, forward, right, up, half_width, half_height)

    def generate_ray(self, u: float, v: float) -> Vec3:
        px = (2.0 * u - 1.0) * self.half_width
        py = (1.0 - 2.0 * v) * self.half_height
        direction = (self.forward + self.right * px + self.up * py).normalized()
        return direction


@dataclass
class Scene:
    materials: List[Material]
    triangles: List[Triangle]
    light_ids: List[int]
    camera: Camera


def moller_trumbore(ray_origin: Vec3, ray_direction: Vec3, triangle: Triangle) -> Optional[float]:
    """Пересечение луча с треугольником через барицентрические координаты.
    Возвращает параметр луча t или None если пересечения нет.
    """
    edge1 = triangle.b - triangle.a
    edge2 = triangle.c - triangle.a
    p_vector = ray_direction.cross(edge2)
    determinant = edge1.dot(p_vector)
    if abs(determinant) < EPS:
        return None
    inverse_determinant = 1.0 / determinant
    origin_to_vertex = ray_origin - triangle.a
    barycentric_u = origin_to_vertex.dot(p_vector) * inverse_determinant
    if barycentric_u < 0.0 or barycentric_u > 1.0:
        return None
    q_vector = origin_to_vertex.cross(edge1)
    barycentric_v = ray_direction.dot(q_vector) * inverse_determinant
    if barycentric_v < 0.0 or (barycentric_u + barycentric_v) > 1.0:
        return None
    ray_parameter = edge2.dot(q_vector) * inverse_determinant
    if ray_parameter <= EPS:
        return None
    return ray_parameter


def intersect_scene(ray_origin: Vec3, ray_direction: Vec3, triangles: Sequence[Triangle]) -> Optional[Hit]:
    """Перебирает все треугольники, возвращает ближайшее пересечение."""
    nearest_distance = float("inf")
    nearest_hit: Optional[Hit] = None
    for triangle_index, triangle in enumerate(triangles):
        distance = moller_trumbore(ray_origin, ray_direction, triangle)
        if distance is None or distance >= nearest_distance:
            continue
        hit_position = ray_origin + ray_direction * distance
        surface_normal = triangle.normal()

        if surface_normal.dot(ray_direction) > 0.0:
            surface_normal = surface_normal * -1.0
        nearest_distance = distance
        nearest_hit = Hit(t=distance, position=hit_position, normal=surface_normal, tri_id=triangle_index)
    return nearest_hit


def orthonormal_basis(normal: Vec3) -> Tuple[Vec3, Vec3]:
    if abs(normal.x) > 0.1:
        tangent = Vec3(0.0, 1.0, 0.0).cross(normal).normalized()
    else:
        tangent = Vec3(1.0, 0.0, 0.0).cross(normal).normalized()
    bitangent = normal.cross(tangent).normalized()
    return tangent, bitangent


def sample_cosine_hemisphere(normal: Vec3) -> Tuple[Vec3, float]:
    """Косинусная выборка по полусфере
    Направления выбираются с PDF = cos(theta)/pi — выборка по значимости
    """
    random_radial = random.random()
    random_angular = random.random()
    disk_radius = math.sqrt(random_radial)
    angle = 2.0 * math.pi * random_angular
    local_x = disk_radius * math.cos(angle)
    local_y = disk_radius * math.sin(angle)
    local_z = math.sqrt(max(0.0, 1.0 - random_radial))
    tangent, bitangent = orthonormal_basis(normal)
    direction = (tangent * local_x + bitangent * local_y + normal * local_z).normalized()
    pdf = max(EPS, normal.dot(direction) / math.pi)
    return direction, pdf


def reflect(incident_direction: Vec3, surface_normal: Vec3) -> Vec3:
    """Зеркальное отражение по закону r = i - 2*(i.n)*n."""
    return (incident_direction - surface_normal * (2.0 * incident_direction.dot(surface_normal))).normalized()


def sample_triangle_uniform(triangle: Triangle) -> Tuple[Vec3, float]:
    """Равномерная выборка точки на треугольнике через барицентрические координаты.
    Корень из random_1 обеспечивает равномерное распределение по площади.
    """
    random_1 = random.random()
    random_2 = random.random()
    sqrt_random_1 = math.sqrt(random_1)
    barycentric_u = 1.0 - sqrt_random_1
    barycentric_v = random_2 * sqrt_random_1
    barycentric_w = 1.0 - barycentric_u - barycentric_v
    sampled_point = triangle.a * barycentric_u + triangle.b * barycentric_v + triangle.c * barycentric_w
    pdf_area = 1.0 / max(EPS, triangle.area())
    return sampled_point, pdf_area


def build_scene() -> Scene:
    """Встроенная тестовая сцена Cornell Box с дополнительными объектами"""
    materials = [
        Material(kd=Vec3(0.75, 0.75, 0.75), ks=Vec3(0.0, 0.0, 0.0)),  # white
        Material(kd=Vec3(0.75, 0.15, 0.15), ks=Vec3(0.0, 0.0, 0.0)),  # red
        Material(kd=Vec3(0.15, 0.75, 0.15), ks=Vec3(0.0, 0.0, 0.0)),  # green
        Material(kd=Vec3(0.05, 0.05, 0.05), ks=Vec3(0.9, 0.9, 0.9)),  # mirror-like
        Material(kd=Vec3(0.8, 0.8, 0.8), ks=Vec3(0.0, 0.0, 0.0)),  # matte object
    ]

    triangles: List[Triangle] = []
    light_ids: List[int] = []

    def add_quad(
        vertex_0: Vec3,
        vertex_1: Vec3,
        vertex_2: Vec3,
        vertex_3: Vec3,
        material_id: int,
        emission: Vec3 = Vec3(0.0, 0.0, 0.0),
    ):
        triangles.append(Triangle(vertex_0, vertex_1, vertex_2, material_id, emission))
        triangles.append(Triangle(vertex_0, vertex_2, vertex_3, material_id, emission))

    add_quad(Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(1, 0, -3), Vec3(-1, 0, -3), 0)  # floor
    add_quad(Vec3(-1, 2, -1), Vec3(-1, 2, -3), Vec3(1, 2, -3), Vec3(1, 2, -1), 0)  # ceiling
    add_quad(Vec3(-1, 0, -3), Vec3(1, 0, -3), Vec3(1, 2, -3), Vec3(-1, 2, -3), 0)  # back wall
    add_quad(Vec3(-1, 0, -1), Vec3(-1, 0, -3), Vec3(-1, 2, -3), Vec3(-1, 2, -1), 1)  # left wall
    add_quad(Vec3(1, 0, -3), Vec3(1, 0, -1), Vec3(1, 2, -1), Vec3(1, 2, -3), 2)  # right wall

    light_emission = Vec3(20.0, 18.0, 15.0)
    light_corner_0 = Vec3(-0.35, 1.99, -1.6)
    light_corner_1 = Vec3(0.35, 1.99, -1.6)
    light_corner_2 = Vec3(0.35, 1.99, -2.3)
    light_corner_3 = Vec3(-0.35, 1.99, -2.3)
    triangles.append(Triangle(light_corner_0, light_corner_2, light_corner_1, 0, light_emission))
    light_ids.append(len(triangles) - 1)
    triangles.append(Triangle(light_corner_0, light_corner_3, light_corner_2, 0, light_emission))
    light_ids.append(len(triangles) - 1)

    pyramid_base_0 = Vec3(-0.55, 0.0, -2.5)
    pyramid_base_1 = Vec3(-0.15, 0.0, -2.5)
    pyramid_base_2 = Vec3(-0.35, 0.0, -2.1)
    pyramid_apex = Vec3(-0.35, 0.65, -2.3)
    no_emission = Vec3(0.0, 0.0, 0.0)
    triangles.extend(
        [
            Triangle(pyramid_base_0, pyramid_base_1, pyramid_base_2, 4, no_emission),
            Triangle(pyramid_base_0, pyramid_base_1, pyramid_apex, 4, no_emission),
            Triangle(pyramid_base_1, pyramid_base_2, pyramid_apex, 4, no_emission),
            Triangle(pyramid_base_2, pyramid_base_0, pyramid_apex, 4, no_emission),
        ]
    )

    mirror_corner_0 = Vec3(0.2, 0.0, -2.7)
    mirror_corner_1 = Vec3(0.75, 0.0, -2.35)
    mirror_corner_2 = Vec3(0.55, 0.9, -2.35)
    mirror_corner_3 = Vec3(0.0, 0.9, -2.7)
    add_quad(mirror_corner_0, mirror_corner_1, mirror_corner_2, mirror_corner_3, 3)

    camera = Camera.look_at(
        origin=Vec3(0.0, 1.0, -0.2),
        target=Vec3(0.0, 0.9, -2.1),
        up_hint=Vec3(0.0, 1.0, 0.0),
        fov_deg=45.0,
        aspect=1.0,
    )
    return Scene(materials=materials, triangles=triangles, light_ids=light_ids, camera=camera)


def parse_vec3_csv(value: str, name: str) -> Vec3:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must have format r,g,b with 3 values.")
    try:
        x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise ValueError(f"{name} contains non-numeric values: {value}") from exc
    return Vec3(x, y, z)


def parse_face_vertex(face_token: str, vertex_count: int) -> int:
    raw = face_token.split("/")[0]
    idx = int(raw)
    if idx == 0:
        raise ValueError("OBJ index 0 is invalid.")
    if idx > 0:
        return idx - 1
    return vertex_count + idx


def compute_bbox(vertices: Sequence[Vec3]) -> Tuple[Vec3, Vec3]:
    if not vertices:
        raise ValueError("Scene has no vertices.")
    min_v = Vec3(vertices[0].x, vertices[0].y, vertices[0].z)
    max_v = Vec3(vertices[0].x, vertices[0].y, vertices[0].z)
    for v in vertices[1:]:
        min_v = Vec3(min(min_v.x, v.x), min(min_v.y, v.y), min(min_v.z, v.z))
        max_v = Vec3(max(max_v.x, v.x), max(max_v.y, v.y), max(max_v.z, v.z))
    return min_v, max_v


def _parse_mtl(mtl_path: Path, fallback_kd: Vec3, fallback_ks: Vec3) -> Dict[str, Tuple[Vec3, Vec3]]:
    result: Dict[str, Tuple[Vec3, Vec3]] = {}
    current: Optional[str] = None
    kd = Vec3(fallback_kd.x, fallback_kd.y, fallback_kd.z)
    ks = Vec3(0.0, 0.0, 0.0)
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0].lower()
            if tag == "newmtl":
                if current is not None:
                    result[current] = (kd, ks)
                current = parts[1] if len(parts) > 1 else "default"
                kd = Vec3(fallback_kd.x, fallback_kd.y, fallback_kd.z)
                ks = Vec3(0.0, 0.0, 0.0)
            elif tag == "kd" and len(parts) >= 4:
                kd = Vec3(float(parts[1]), float(parts[2]), float(parts[3]))
            elif tag == "ks" and len(parts) >= 4:
                ks = Vec3(float(parts[1]), float(parts[2]), float(parts[3]))
    if current is not None:
        result[current] = (kd, ks)
    return result


def build_scene_from_obj(obj_path: Path, obj_kd: Vec3, obj_ks: Vec3, light_emission: Vec3) -> Scene:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    material_map: Dict[str, int] = {}  # name -> index into materials list
    material_props: List[Tuple[Vec3, Vec3]] = []  # (kd, ks) per index
    mtl_library: Dict[str, Tuple[Vec3, Vec3]] = {}
    current_material = "default"

    def ensure_material(name: str, kd: Optional[Vec3] = None, ks: Optional[Vec3] = None) -> int:
        if name not in material_map:
            material_map[name] = len(material_props)
            material_props.append((
                kd if kd is not None else obj_kd,
                ks if ks is not None else obj_ks,
            ))
        return material_map[name]

    ensure_material(current_material)
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            tag = parts[0]
            if tag == "mtllib" and len(parts) >= 2:
                mtl_path = obj_path.parent / parts[1]
                if mtl_path.exists():
                    mtl_library = _parse_mtl(mtl_path, obj_kd, obj_ks)
            elif tag == "v" and len(parts) >= 4:
                vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "usemtl" and len(parts) >= 2:
                current_material = parts[1]
                if current_material in mtl_library:
                    mtl_kd, mtl_ks = mtl_library[current_material]
                    ensure_material(current_material, mtl_kd, mtl_ks)
                else:
                    ensure_material(current_material)
            elif tag == "f" and len(parts) >= 4:
                face_vertex_indices: List[int] = []
                for token in parts[1:]:
                    vertex_index = parse_face_vertex(token, len(vertices))
                    if vertex_index < 0 or vertex_index >= len(vertices):
                        raise ValueError(f"Face index out of range in OBJ: {token}")
                    face_vertex_indices.append(vertex_index)
                material_id = ensure_material(current_material)

                for i in range(1, len(face_vertex_indices) - 1):
                    vertex_a = vertices[face_vertex_indices[0]]
                    vertex_b = vertices[face_vertex_indices[i]]
                    vertex_c = vertices[face_vertex_indices[i + 1]]
                    triangles.append(Triangle(vertex_a, vertex_b, vertex_c, material_id, Vec3(0.0, 0.0, 0.0)))

    if not triangles:
        raise ValueError(f"OBJ scene is empty or has no faces: {obj_path}")

    materials = [Material(kd=kd, ks=ks) for kd, ks in material_props]
    if not materials:
        materials = [Material(kd=obj_kd, ks=obj_ks)]

    min_v, max_v = compute_bbox(vertices)
    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    extent = max(size.x, size.y, size.z)
    extent = max(extent, 1e-3)

    light_y = max_v.y + 0.35 * extent
    half = 0.25 * extent
    l0 = Vec3(center.x - half, light_y, center.z - half)
    l1 = Vec3(center.x + half, light_y, center.z - half)
    l2 = Vec3(center.x + half, light_y, center.z + half)
    l3 = Vec3(center.x - half, light_y, center.z + half)
    light_material_id = 0
    triangles.append(Triangle(l0, l2, l1, light_material_id, light_emission))
    light_ids = [len(triangles) - 1]
    triangles.append(Triangle(l0, l3, l2, light_material_id, light_emission))
    light_ids.append(len(triangles) - 1)

    camera = Camera.look_at(
        origin=Vec3(center.x, center.y + 0.25 * extent, max_v.z + 2.1 * extent),
        target=center,
        up_hint=Vec3(0.0, 1.0, 0.0),
        fov_deg=45.0,
        aspect=1.0,
    )
    return Scene(materials=materials, triangles=triangles, light_ids=light_ids, camera=camera)


def export_triangles_to_obj(obj_path: Path, triangles: Sequence[Triangle]):
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8") as output_file:
        output_file.write("# Exported by path_tracer.py\n")
        output_file.write("o exported_scene\n")
        vertex_index = 1
        for triangle in triangles:
            output_file.write(f"v {triangle.a.x:.8f} {triangle.a.y:.8f} {triangle.a.z:.8f}\n")
            output_file.write(f"v {triangle.b.x:.8f} {triangle.b.y:.8f} {triangle.b.z:.8f}\n")
            output_file.write(f"v {triangle.c.x:.8f} {triangle.c.y:.8f} {triangle.c.z:.8f}\n")
            output_file.write(f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}\n")
            vertex_index += 3


def _illuminance_at_point(hit: Hit, light_triangle: Triangle) -> float:
    """Оценка освещённости (irradiance) от источника в точке поверхности.
    Использует центроид треугольника источника как репрезентативную точку.
    Формула: E ~ Le * area * cos_surface * cos_light / distance^2
    """
    light_centroid = (light_triangle.a + light_triangle.b + light_triangle.c) / 3.0
    direction_to_light = light_centroid - hit.position
    distance_squared = direction_to_light.dot(direction_to_light)
    if distance_squared <= EPS:
        return 0.0
    direction_to_light_normalized = direction_to_light / math.sqrt(distance_squared)
    cos_surface = max(0.0, hit.normal.dot(direction_to_light_normalized))
    if cos_surface <= 0.0:
        return 0.0
    cos_light = max(0.0, light_triangle.normal().dot(direction_to_light_normalized * -1.0))
    if cos_light <= 0.0:
        return 0.0
    emission_luminance = light_triangle.emission.luminance()
    light_area = light_triangle.area()
    return emission_luminance * light_area * cos_surface * cos_light / distance_squared


def estimate_direct_light(
    hit: Hit,
    triangles: Sequence[Triangle],
    light_ids: Sequence[int],
    throughput: Vec3,
    kd: Vec3,
) -> Vec3:
    if not light_ids or kd.max_component() <= 0.0:
        return Vec3(0.0, 0.0, 0.0)

    # Детерминированный выбор самого значимого источника
    best_light_id = -1
    best_illuminance = 0.0
    for light_id in light_ids:
        illuminance = _illuminance_at_point(hit, triangles[light_id])
        if illuminance > best_illuminance:
            best_illuminance = illuminance
            best_light_id = light_id
    if best_light_id < 0:
        return Vec3(0.0, 0.0, 0.0)

    light_triangle = triangles[best_light_id]

    sampled_point_on_light, pdf_area = sample_triangle_uniform(light_triangle)
    direction_to_light = sampled_point_on_light - hit.position
    distance_squared = direction_to_light.dot(direction_to_light)
    if distance_squared <= EPS:
        return Vec3(0.0, 0.0, 0.0)
    light_direction = direction_to_light / math.sqrt(distance_squared)

    cos_surface = max(0.0, hit.normal.dot(light_direction))
    if cos_surface <= 0.0:
        return Vec3(0.0, 0.0, 0.0)
    light_normal = light_triangle.normal()
    cos_light = max(0.0, light_normal.dot(light_direction * -1.0))
    if cos_light <= 0.0:
        return Vec3(0.0, 0.0, 0.0)

    shadow_ray_origin = hit.position + hit.normal * 1e-4
    shadow_hit = intersect_scene(shadow_ray_origin, light_direction, triangles)
    if shadow_hit is None or shadow_hit.tri_id != best_light_id:
        return Vec3(0.0, 0.0, 0.0)

    lambertian_brdf = kd * (1.0 / math.pi)
    geometric_coupling = cos_surface * cos_light / max(EPS, distance_squared)
    return throughput * lambertian_brdf * light_triangle.emission * (geometric_coupling / max(EPS, pdf_area))


def trace_path(
    primary_ray_origin: Vec3,
    primary_ray_direction: Vec3,
    materials: Sequence[Material],
    triangles: Sequence[Triangle],
    light_ids: Sequence[int],
    max_bounces: int,
) -> Vec3:
    accumulated_radiance = Vec3(0.0, 0.0, 0.0)
    throughput = Vec3(1.0, 1.0, 1.0)
    current_origin = primary_ray_origin
    current_direction = primary_ray_direction

    for bounce in range(max_bounces):
        hit = intersect_scene(current_origin, current_direction, triangles)
        if hit is None:
            break
        hit_triangle = triangles[hit.tri_id]
        hit_material = materials[hit_triangle.material_id]

        # Попали в источник — добавляем его яркость с учётом фильтра пути
        if hit_triangle.emission.max_component() > 0.0:
            accumulated_radiance += throughput * hit_triangle.emission
            break

        # Прямое освещение (NEE) в точке столкновения
        accumulated_radiance += estimate_direct_light(
            hit, triangles, light_ids, throughput, hit_material.kd
        )

        # Русская рулетка на этапе поверхности.
        # Решает: продолжать путь или оборвать. Применяется ДО выбора отскока,
        # начиная с bounce >= 1 (первый отскок всегда выполняется)
        if bounce >= 1:
            survival_probability = min(0.95, max(0.05, throughput.max_component()))
            if random.random() > survival_probability:
                break
            throughput = throughput / survival_probability

        # Выбор типа отражения (диффузия / зеркало) — выборка по значимости BSDF
        # Вероятность пропорциональна яркости соответствующего коэффициента
        weight_diffuse = hit_material.kd.luminance()
        weight_specular = hit_material.ks.luminance()
        total_weight = weight_diffuse + weight_specular
        if total_weight <= EPS:
            break
        probability_diffuse = weight_diffuse / total_weight

        event_random = random.random()
        if event_random < probability_diffuse:
            # Диффузный отскок (Ламберт): косинусная выборка по полусфере
            # PDF = cos(theta)/pi — выборка по значимости
            new_direction, pdf = sample_cosine_hemisphere(hit.normal)
            cos_outgoing = max(0.0, hit.normal.dot(new_direction))
            lambertian_brdf = hit_material.kd * (1.0 / math.pi)
            throughput = throughput * lambertian_brdf * (cos_outgoing / max(EPS, pdf)) / max(EPS, probability_diffuse)
            current_origin = hit.position + hit.normal * 1e-4
            current_direction = new_direction
        else:
            # Зеркальный отскок: строгое отражение r = i - 2*(i.n)*n
            new_direction = reflect(current_direction, hit.normal)
            probability_specular = 1.0 - probability_diffuse
            throughput = throughput * (hit_material.ks / max(EPS, probability_specular))
            current_origin = hit.position + hit.normal * 1e-4
            current_direction = new_direction

    return accumulated_radiance


def tonemap_and_gamma(image: List[List[Vec3]], gamma: float) -> List[List[Tuple[int, int, int]]]:
    """нормировка по максимуму + гамма-коррекция"""
    max_brightness = 0.0
    for row in image:
        for pixel in row:
            max_brightness = max(max_brightness, pixel.max_component())
    normalization_scale = 1.0 / max(max_brightness, EPS)

    output_pixels: List[List[Tuple[int, int, int]]] = []
    inverse_gamma = 1.0 / gamma
    for row in image:
        output_row: List[Tuple[int, int, int]] = []
        for pixel in row:
            normalized = (pixel * normalization_scale).clamp01()
            gamma_corrected = Vec3(
                normalized.x ** inverse_gamma,
                normalized.y ** inverse_gamma,
                normalized.z ** inverse_gamma,
            )
            output_row.append(
                (
                    max(0, min(255, int(gamma_corrected.x * 255.0 + 0.5))),
                    max(0, min(255, int(gamma_corrected.y * 255.0 + 0.5))),
                    max(0, min(255, int(gamma_corrected.z * 255.0 + 0.5))),
                )
            )
        output_pixels.append(output_row)
    return output_pixels


def write_ppm(path: Path, pixels: List[List[Tuple[int, int, int]]], width: int, height: int):
    with path.open("w", encoding="ascii") as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n255\n")
        for row in pixels:
            for r, g, b in row:
                f.write(f"{r} {g} {b}\n")


_worker_scene: Optional[Scene] = None
_worker_width: int = 0
_worker_height: int = 0
_worker_spp: int = 0
_worker_max_bounces: int = 0


def _worker_init(scene: Scene, width: int, height: int, spp: int, max_bounces: int):
    global _worker_scene, _worker_width, _worker_height, _worker_spp, _worker_max_bounces
    _worker_scene = scene
    _worker_width = width
    _worker_height = height
    _worker_spp = spp
    _worker_max_bounces = max_bounces


def _render_row(task: Tuple[int, int]) -> Tuple[int, List[Vec3]]:
    """Задача воркера: рендерит одну строку пикселей."""
    row_y, random_seed = task
    random.seed(random_seed)
    scene = _worker_scene
    width = _worker_width
    height = _worker_height
    spp = _worker_spp
    max_bounces = _worker_max_bounces
    row_pixels: List[Vec3] = []
    for column_x in range(width):
        pixel_color = Vec3(0.0, 0.0, 0.0)
        for _ in range(spp):
            # Случайный сдвиг внутри пикселя — антиалиасинг
            screen_u = (column_x + random.random()) / width
            screen_v = (row_y + random.random()) / height
            ray_direction = scene.camera.generate_ray(screen_u, screen_v)
            pixel_color += trace_path(
                scene.camera.origin,
                ray_direction,
                scene.materials,
                scene.triangles,
                scene.light_ids,
                max_bounces,
            )
        # Усреднение по spp — оценка интеграла методом Монте-Карло
        row_pixels.append(pixel_color / float(spp))
    return row_y, row_pixels


def render(
    scene: Scene,
    width: int,
    height: int,
    spp: int,
    max_bounces: int,
    gamma: float,
    output_file: Path,
    num_workers: int = 0,
):
    """Главная функция рендеринга. Распределяет работу по строкам между воркерами."""
    scene.camera.half_width = scene.camera.half_height * (width / height)

    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, height)

    base_seed = random.randint(0, 2**31)
    row_tasks = [(row_y, base_seed + row_y) for row_y in range(height)]

    image: List[List[Vec3]] = [
        [Vec3(0.0, 0.0, 0.0) for _ in range(width)] for _ in range(height)
    ]

    progress_log_step = max(1, height // 20)
    rows_completed = 0
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(scene, width, height, spp, max_bounces),
    ) as pool:
        for row_y, row_pixels in pool.imap_unordered(_render_row, row_tasks, chunksize=4):
            image[row_y] = row_pixels
            rows_completed += 1
            if rows_completed % progress_log_step == 0 or rows_completed == height:
                print(f"Progress: {rows_completed}/{height} lines", flush=True)

    final_pixels = tonemap_and_gamma(image, gamma=gamma)
    write_ppm(output_file, final_pixels, width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple path tracing lab work renderer")
    parser.add_argument("--width", type=int, default=600, help="Image width (>=500 recommended)")
    parser.add_argument("--height", type=int, default=600, help="Image height (>=500 recommended)")
    parser.add_argument("--spp", type=int, default=128, help="Samples per pixel")
    parser.add_argument("--max-bounces", type=int, default=8, help="Maximum path depth")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma correction value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("render.ppm"), help="Output PPM file")
    parser.add_argument("--obj", type=Path, default=None, help="Import geometry from OBJ file")
    parser.add_argument(
        "--obj-kd",
        type=str,
        default="0.8,0.8,0.8",
        help="Diffuse reflectance for imported OBJ materials, format r,g,b",
    )
    parser.add_argument(
        "--obj-ks",
        type=str,
        default="0.0,0.0,0.0",
        help="Mirror reflectance for imported OBJ materials, format r,g,b",
    )
    parser.add_argument(
        "--light-emission",
        type=str,
        default="20,18,15",
        help="Area light emission for imported OBJ scene, format r,g,b",
    )
    parser.add_argument("--export-obj", type=Path, default=None, help="Export current scene triangles to OBJ file")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export OBJ scene without rendering PPM",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0 = auto, uses all CPU cores)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("Image size must be positive.")
    if args.spp < 1:
        raise ValueError("spp must be >= 1.")
    obj_kd = parse_vec3_csv(args.obj_kd, "obj-kd")
    obj_ks = parse_vec3_csv(args.obj_ks, "obj-ks")
    light_emission = parse_vec3_csv(args.light_emission, "light-emission")
    random.seed(args.seed)

    if args.obj is not None:
        scene = build_scene_from_obj(args.obj, obj_kd=obj_kd, obj_ks=obj_ks, light_emission=light_emission)
    else:
        scene = build_scene()

    if args.export_obj is not None:
        export_triangles_to_obj(args.export_obj, scene.triangles)
        if args.export_only:
            return

    render(
        scene=scene,
        width=args.width,
        height=args.height,
        spp=args.spp,
        max_bounces=args.max_bounces,
        gamma=args.gamma,
        output_file=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

#!/usr/bin/env python3
"""Two-dimensional lattice DEM for geometry-dependent beam fracture."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

LOAD_CAPACITY_CUTOFF_FRACTION = 0.01


@dataclass(frozen=True)
class Parameters:
    length: float = 1.0
    height: float = 0.1
    width: float = 0.1
    youngs_modulus: float = 9.0e9
    tensile_strength: float = 0.7e6
    compressive_strength: float = 3.0e6
    max_displacement: float = 0.005
    steps: int = 100
    layers: int = 11


@dataclass(frozen=True)
class Bond:
    first: int
    second: int
    normal_x: float
    normal_z: float
    length: float
    stiffness: float


def inner(first: list[float], second: list[float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def conjugate_gradient(
    multiply,
    rhs: list[float],
    diagonal: list[float],
    initial: list[float],
) -> list[float]:
    result = initial[:]
    residual = [b - a for b, a in zip(rhs, multiply(result))]
    rhs_norm = math.sqrt(inner(rhs, rhs))
    tolerance = max(1e-10, rhs_norm * 1e-9)
    if math.sqrt(inner(residual, residual)) <= tolerance:
        return result
    if any(value <= 0.0 for value in diagonal):
        raise RuntimeError("The lattice has an unconstrained degree of freedom")

    preconditioned = [r / d for r, d in zip(residual, diagonal)]
    direction = preconditioned[:]
    residual_product = inner(residual, preconditioned)
    for _ in range(max(500, 4 * len(rhs))):
        product = multiply(direction)
        denominator = inner(direction, product)
        if denominator <= 0.0:
            raise RuntimeError("The lattice stiffness matrix is not positive definite")
        alpha = residual_product / denominator
        result = [value + alpha * step for value, step in zip(result, direction)]
        residual = [value - alpha * change for value, change in zip(residual, product)]
        if math.sqrt(inner(residual, residual)) <= tolerance:
            return result
        preconditioned = [r / d for r, d in zip(residual, diagonal)]
        next_product = inner(residual, preconditioned)
        beta = next_product / residual_product
        direction = [value + beta * old for value, old in zip(preconditioned, direction)]
        residual_product = next_product
    raise RuntimeError("Conjugate-gradient solver did not converge")


def build_lattice(parameters: Parameters) -> tuple[list[tuple[float, float]], list[Bond], int, float, float]:
    if parameters.layers < 5 or parameters.layers % 2 == 0:
        raise ValueError("layers must be an odd integer of at least 5")

    nz = parameters.layers
    dz = parameters.height / (nz - 1)
    nx = max(5, round(parameters.length / dz) + 1)
    if nx % 2 == 0:
        nx += 1
    dx = parameters.length / (nx - 1)
    nodes = [
        (i * dx, -parameters.height / 2.0 + j * dz)
        for j in range(nz)
        for i in range(nx)
    ]
    axial_stiffness = 0.75 * parameters.youngs_modulus * parameters.width
    diagonal_stiffness = 0.375 * parameters.youngs_modulus * parameters.width
    bonds: list[Bond] = []
    for j in range(nz):
        for i in range(nx):
            first = j * nx + i
            for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < nx and 0 <= nj < nz):
                    continue
                second = nj * nx + ni
                x1, z1 = nodes[first]
                x2, z2 = nodes[second]
                delta_x, delta_z = x2 - x1, z2 - z1
                length = math.hypot(delta_x, delta_z)
                bonds.append(
                    Bond(
                        first,
                        second,
                        delta_x / length,
                        delta_z / length,
                        length,
                        diagonal_stiffness if di and dj else axial_stiffness,
                    )
                )
    center_column = nx // 2
    return nodes, bonds, nx, dx, center_column


def load_schedule(parameters: Parameters) -> list[float]:
    crack_estimate = (
        parameters.tensile_strength
        * parameters.length**2
        / (12.0 * parameters.youngs_modulus * parameters.height)
    )
    crack_region = min(
        parameters.max_displacement,
        max(1.5 * crack_estimate, 0.1 * parameters.max_displacement),
    )
    crack_steps = max(1, 9 * parameters.steps // 10)
    continuation_steps = parameters.steps - crack_steps
    schedule = [
        crack_region * step / crack_steps
        for step in range(1, crack_steps + 1)
    ]
    if continuation_steps:
        schedule.extend(
            crack_region
            + (parameters.max_displacement - crack_region) * step / continuation_steps
            for step in range(1, continuation_steps + 1)
        )
    return schedule


def run_simulation(parameters: Parameters, output_dir: Path) -> dict[str, object]:
    if min(parameters.length, parameters.height, parameters.width) <= 0.0:
        raise ValueError("Beam dimensions must be positive")
    if parameters.youngs_modulus <= 0.0:
        raise ValueError("Young's modulus must be positive")
    if parameters.tensile_strength <= 0.0 or parameters.compressive_strength <= 0.0:
        raise ValueError("Tensile and compressive strengths must be positive")
    if parameters.max_displacement <= 0.0 or parameters.steps < 1:
        raise ValueError("Maximum displacement and number of steps must be positive")

    nodes, bonds, nx, dx, center_column = build_lattice(parameters)
    nz = parameters.layers
    node_count = len(nodes)
    dof_count = 2 * node_count
    left_support_nodes = {j * nx for j in range(nz)}
    right_support_nodes = {j * nx + nx - 1 for j in range(nz)}
    support_nodes = left_support_nodes | right_support_nodes
    punch_nodes = {
        (nz - 1) * nx + i
        for i in range(nx)
        if abs(i - center_column) <= 1
    }
    fixed_values: dict[int, float] = {}
    for node in support_nodes:
        fixed_values[2 * node] = 0.0
        fixed_values[2 * node + 1] = 0.0
    punch_dofs = {2 * node + 1 for node in punch_nodes}
    for dof in punch_dofs:
        fixed_values[dof] = 0.0
    free_dofs = [dof for dof in range(dof_count) if dof not in fixed_values]
    free_map = {dof: i for i, dof in enumerate(free_dofs)}
    displacement = [0.0] * dof_count
    alive = [True] * len(bonds)
    failure_modes = [""] * len(bonds)
    regularizer = parameters.youngs_modulus * parameters.width * dx * 1e-10
    tensile_strain_limit = parameters.tensile_strength / parameters.youngs_modulus
    compressive_strain_limit = parameters.compressive_strength / parameters.youngs_modulus
    history: list[dict[str, float | int | str]] = []
    frames: list[dict[str, object]] = []
    first_failure: dict[str, float | int | str] | None = None
    termination_reason = "maximum_displacement_reached"
    schedule = load_schedule(parameters)
    output_dir.mkdir(parents=True, exist_ok=True)

    def coefficients(bond: Bond) -> tuple[tuple[int, float], ...]:
        first_dof = 2 * bond.first
        second_dof = 2 * bond.second
        return (
            (first_dof, -bond.normal_x),
            (first_dof + 1, -bond.normal_z),
            (second_dof, bond.normal_x),
            (second_dof + 1, bond.normal_z),
        )

    bond_coefficients = [coefficients(bond) for bond in bonds]

    def extension(index: int, values: list[float]) -> float:
        return sum(
            coefficient * values[dof]
            for dof, coefficient in bond_coefficients[index]
        )

    def solve_equilibrium(reference: list[float]) -> None:
        diagonal = [regularizer] * len(free_dofs)
        known_product = [0.0] * len(free_dofs)
        for index, bond in enumerate(bonds):
            if not alive[index]:
                continue
            coefficients_for_bond = bond_coefficients[index]
            known_extension = sum(
                coefficient * fixed_values.get(dof, 0.0)
                for dof, coefficient in coefficients_for_bond
            )
            known_force = bond.stiffness * known_extension
            for dof, coefficient in coefficients_for_bond:
                free_index = free_map.get(dof)
                if free_index is not None:
                    diagonal[free_index] += bond.stiffness * coefficient**2
                    known_product[free_index] += known_force * coefficient

        rhs = [
            -known_product[i] + regularizer * reference[dof]
            for i, dof in enumerate(free_dofs)
        ]

        def multiply(free_values: list[float]) -> list[float]:
            result = [regularizer * value for value in free_values]
            for index, bond in enumerate(bonds):
                if not alive[index]:
                    continue
                coefficients_for_bond = bond_coefficients[index]
                bond_extension = sum(
                    coefficient * free_values[free_map[dof]]
                    for dof, coefficient in coefficients_for_bond
                    if dof in free_map
                )
                force = bond.stiffness * bond_extension
                for dof, coefficient in coefficients_for_bond:
                    free_index = free_map.get(dof)
                    if free_index is not None:
                        result[free_index] += force * coefficient
            return result

        solution = conjugate_gradient(
            multiply,
            rhs,
            diagonal,
            [displacement[dof] for dof in free_dofs],
        )
        for dof, value in zip(free_dofs, solution):
            displacement[dof] = value
        for dof, value in fixed_values.items():
            displacement[dof] = value

    def load_path_exists() -> bool:
        parent = list(range(node_count))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for index, bond in enumerate(bonds):
            if alive[index]:
                root_first, root_second = find(bond.first), find(bond.second)
                if root_first != root_second:
                    parent[root_second] = root_first
        support_roots = {find(node) for node in support_nodes}
        return any(find(node) in support_roots for node in punch_nodes)

    def punch_reaction() -> float:
        resultant = 0.0
        for index, bond in enumerate(bonds):
            if not alive[index]:
                continue
            force = bond.stiffness * extension(index, displacement)
            for dof, coefficient in bond_coefficients[index]:
                if dof in punch_dofs:
                    resultant += force * coefficient
        return abs(resultant)

    for step, travel in enumerate(schedule, start=1):
        failed_before_step = sum(mode != "" for mode in failure_modes)
        previous_peak_force = max(
            (float(row["force_N"]) for row in history),
            default=0.0,
        )
        for dof in punch_dofs:
            fixed_values[dof] = -travel
        reference = displacement[:]
        damage_iteration = 0
        while True:
            solve_equilibrium(reference)
            reaction_before_damage = punch_reaction()
            newly_failed: list[int] = []
            failure_utilizations: dict[int, float] = {}
            for index, bond in enumerate(bonds):
                if not alive[index]:
                    continue
                strain = extension(index, displacement) / bond.length
                if strain > tensile_strain_limit:
                    newly_failed.append(index)
                    failure_modes[index] = "tension"
                    failure_utilizations[index] = strain / tensile_strain_limit
                elif strain < -compressive_strain_limit:
                    newly_failed.append(index)
                    failure_modes[index] = "compression"
                    failure_utilizations[index] = abs(strain) / compressive_strain_limit
            if not newly_failed:
                break

            if first_failure is None:
                first_index = max(
                    newly_failed,
                    key=lambda index: failure_utilizations[index],
                )
                first_bond = bonds[first_index]
                first_failure = {
                    "step": step,
                    "displacement_m": travel,
                    "force_N": reaction_before_damage,
                    "x_m": (
                        nodes[first_bond.first][0] + nodes[first_bond.second][0]
                    ) / 2.0,
                    "z_m": (
                        nodes[first_bond.first][1] + nodes[first_bond.second][1]
                    ) / 2.0,
                    "bond_angle_deg": math.degrees(
                        math.atan2(first_bond.normal_z, first_bond.normal_x)
                    ),
                    "failure_mode": failure_modes[first_index],
                }
            for index in newly_failed:
                alive[index] = False
            damage_iteration += 1
            if not load_path_exists():
                termination_reason = "loss_of_load_path_after_bond_failure"
                break
            if damage_iteration > len(bonds):
                raise RuntimeError(f"Damage iteration did not converge at load step {step}")

        force = 0.0 if termination_reason == "loss_of_load_path_after_bond_failure" else punch_reaction()
        failed_after_step = sum(mode != "" for mode in failure_modes)
        if (
            termination_reason == "maximum_displacement_reached"
            and failed_after_step > failed_before_step
            and previous_peak_force > 0.0
            and force <= LOAD_CAPACITY_CUTOFF_FRACTION * previous_peak_force
        ):
            termination_reason = "loss_of_load_capacity_after_bond_failure"
        history.append(
            {
                "step": step,
                "displacement_m": travel,
                "force_N": force,
                "failed_tensile_bonds": sum(mode == "tension" for mode in failure_modes),
                "failed_compressive_bonds": sum(mode == "compression" for mode in failure_modes),
            }
        )
        frames.append(
            {
                "displacement": displacement[:],
                "strain": [
                    extension(index, displacement) / bond.length if alive[index] else 0.0
                    for index, bond in enumerate(bonds)
                ],
                "failed_tension": [
                    index for index, mode in enumerate(failure_modes) if mode == "tension"
                ],
                "failed_compression": [
                    index
                    for index, mode in enumerate(failure_modes)
                    if mode == "compression"
                ],
            }
        )
        if termination_reason in {
            "loss_of_load_path_after_bond_failure",
            "loss_of_load_capacity_after_bond_failure",
        }:
            break

    force_csv = output_dir / "force_displacement.csv"
    with force_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    crack_csv = output_dir / "crack_network.csv"
    with crack_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("x1_m", "z1_m", "x2_m", "z2_m", "failure_mode"),
        )
        writer.writeheader()
        for index, bond in enumerate(bonds):
            if not alive[index]:
                x1, z1 = nodes[bond.first]
                x2, z2 = nodes[bond.second]
                writer.writerow(
                    {
                        "x1_m": x1,
                        "z1_m": z1,
                        "x2_m": x2,
                        "z2_m": z2,
                        "failure_mode": failure_modes[index],
                    }
                )

    ratio = parameters.length / parameters.height
    expected_pattern = (
        "two diagonal cracks"
        if ratio <= 4.0
        else "one central crack and two root cracks"
    )
    peak = max(history, key=lambda row: float(row["force_N"]))
    summary: dict[str, object] = {
        "model": "2D square-lattice bond DEM with axial, vertical, and diagonal links",
        "parameters": parameters.__dict__,
        "mesh": {
            "nodes": node_count,
            "bonds": len(bonds),
            "nx": nx,
            "nz": nz,
            "dx_m": dx,
            "dz_m": parameters.height / (nz - 1),
            "punch_contact_nodes": len(punch_nodes),
            "equivalent_lattice_poisson_ratio": 1.0 / 3.0,
        },
        "experimental_geometry_regime": {
            "L_over_h": ratio,
            "regime": "short_beam" if ratio <= 4.0 else "long_beam",
            "reported_pattern_for_comparison": expected_pattern,
            "pattern_is_not_prescribed": True,
        },
        "first_bond_failure": first_failure,
        "peak_force_N": float(peak["force_N"]),
        "peak_force_displacement_m": float(peak["displacement_m"]),
        "load_capacity_cutoff_fraction": LOAD_CAPACITY_CUTOFF_FRACTION,
        "load_capacity_cutoff_N": (
            float(peak["force_N"]) * LOAD_CAPACITY_CUTOFF_FRACTION
        ),
        "final_failed_tensile_bonds": sum(mode == "tension" for mode in failure_modes),
        "final_failed_compressive_bonds": sum(
            mode == "compression" for mode in failure_modes
        ),
        "termination_reason": termination_reason,
        "actual_steps": len(history),
        "artifacts": [
            force_csv.name,
            crack_csv.name,
            "force_displacement.svg",
            "crack_state.svg",
            "beam_animation.html",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_force_svg(history, output_dir / "force_displacement.svg")
    write_crack_svg(nodes, bonds, alive, failure_modes, parameters, output_dir / "crack_state.svg")
    write_animation_html(
        nodes,
        bonds,
        history,
        frames,
        parameters,
        punch_nodes,
        termination_reason,
        output_dir / "beam_animation.html",
    )
    return summary


def write_force_svg(history: list[dict[str, float | int | str]], path: Path) -> None:
    width, height = 900, 520
    left, top, right, bottom = 85, 35, 30, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    max_x = max(float(row["displacement_m"]) for row in history)
    max_y = max(float(row["force_N"]) for row in history) * 1.1 or 1.0
    points = " ".join(
        f"{left + float(row['displacement_m']) / max_x * plot_width:.2f},"
        f"{top + (1.0 - float(row['force_N']) / max_y) * plot_height:.2f}"
        for row in history
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">2D lattice DEM: force-displacement</text>
<path d="M {left} {top} V {top + plot_height} H {left + plot_width}" fill="none" stroke="#333" stroke-width="2"/>
<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="2.5"/>
<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif">Center displacement (m)</text>
<text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif">Reaction force (N)</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_crack_svg(
    nodes: list[tuple[float, float]],
    bonds: list[Bond],
    alive: list[bool],
    failure_modes: list[str],
    parameters: Parameters,
    path: Path,
) -> None:
    width, height = 1200, 520
    margin_x, margin_y = 55, 55
    scale = min(
        (width - 2 * margin_x) / parameters.length,
        (height - 2 * margin_y) / parameters.height,
    )
    x_offset = (width - parameters.length * scale) / 2.0
    z_offset = (height - parameters.height * scale) / 2.0

    def point(node: tuple[float, float]) -> tuple[float, float]:
        return x_offset + node[0] * scale, height - z_offset - (node[1] + parameters.height / 2.0) * scale

    intact_lines: list[str] = []
    broken_lines: list[str] = []
    half_length = 0.35 * min(parameters.height / (parameters.layers - 1), parameters.length / 20.0) * scale
    for index, bond in enumerate(bonds):
        first, second = point(nodes[bond.first]), point(nodes[bond.second])
        if alive[index]:
            intact_lines.append(
                f'<line x1="{first[0]:.2f}" y1="{first[1]:.2f}" x2="{second[0]:.2f}" y2="{second[1]:.2f}" stroke="#d7e0e8" stroke-width="0.7"/>'
            )
            continue
        mid_x, mid_y = (first[0] + second[0]) / 2, (first[1] + second[1]) / 2
        perp_x, perp_z = -bond.normal_z, bond.normal_x
        color = "#c62828" if failure_modes[index] == "tension" else "#1565c0"
        broken_lines.append(
            f'<line x1="{mid_x - half_length * perp_x:.2f}" y1="{mid_y + half_length * perp_z:.2f}" '
            f'x2="{mid_x + half_length * perp_x:.2f}" y2="{mid_y - half_length * perp_z:.2f}" '
            f'stroke="{color}" stroke-width="2.2"/>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">2D DEM: разрушенные связи и трещиноподобные траектории</text>
{"".join(intact_lines)}
{"".join(broken_lines)}
<line x1="70" y1="485" x2="105" y2="485" stroke="#c62828" stroke-width="3"/><text x="112" y="489" font-family="sans-serif" font-size="13">Растянутый разрыв</text>
<line x1="300" y1="485" x2="335" y2="485" stroke="#1565c0" stroke-width="3"/><text x="342" y="489" font-family="sans-serif" font-size="13">Разрыв при сжатии</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_animation_html(
    nodes: list[tuple[float, float]],
    bonds: list[Bond],
    history: list[dict[str, float | int | str]],
    frames: list[dict[str, object]],
    parameters: Parameters,
    punch_nodes: set[int],
    termination_reason: str,
    path: Path,
) -> None:
    data = {
        "length": parameters.length,
        "height": parameters.height,
        "tensile_strain_limit": parameters.tensile_strength / parameters.youngs_modulus,
        "compressive_strain_limit": parameters.compressive_strength / parameters.youngs_modulus,
        "nodes": nodes,
        "bonds": [[bond.first, bond.second] for bond in bonds],
        "punch_nodes": sorted(punch_nodes),
        "termination_reason": termination_reason,
        "load_capacity_cutoff_fraction": LOAD_CAPACITY_CUTOFF_FRACTION,
        "history": history,
        "frames": frames,
    }
    payload = json.dumps(data, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2D DEM: анимация разрушения балки</title>
<style>
body{{font:15px system-ui,sans-serif;margin:20px;color:#202124}} .controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}}
canvas{{border:1px solid #ccd2d8;background:#fff;max-width:100%;height:auto}} .views{{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}}
button,input{{font:inherit}} #step{{flex:1;min-width:220px}} .legend span{{margin-right:16px}} .red{{color:#c62828}} .blue{{color:#1565c0}}
</style>
<h2>2D DEM: нагружение и разрушение балки</h2>
<div class="controls"><button id="play">Пуск</button><input id="step" type="range" min="0" max="{max(0, len(frames) - 1)}" value="0">
<span id="readout"></span></div>
<div class="views"><canvas id="beam" width="1100" height="560"></canvas><canvas id="curve" width="420" height="560"></canvas></div>
<p class="legend"><span class="red">Красный — растяжение / разрыв при растяжении</span>
<span class="blue">Синий — сжатие / разрыв при сжатии</span> Деформация увеличена в 4 раза.
Реакция пуансона вычисляется при заданном перемещении.</p>
<script>
const data={payload};
const beam=document.getElementById('beam'), bctx=beam.getContext('2d');
const curve=document.getElementById('curve'), cctx=curve.getContext('2d');
const slider=document.getElementById('step'), readout=document.getElementById('readout');
const failedTension=data.frames.map(f=>new Set(f.failed_tension));
const failedCompression=data.frames.map(f=>new Set(f.failed_compression));
const scale=Math.min(1040/data.length,390/data.height), x0=(beam.width-data.length*scale)/2, y0=beam.height/2;
function nodePoint(frame,index){{const [x,z]=data.nodes[index], ux=frame.displacement[2*index], uz=frame.displacement[2*index+1];
 return [x0+(x+4*ux)*scale,y0-(z+4*uz)*scale]}}
function drawBeam(i){{
 const frame=data.frames[i], tension=failedTension[i], compression=failedCompression[i];
 bctx.clearRect(0,0,beam.width,beam.height); bctx.lineWidth=1;
 for(let j=0;j<data.bonds.length;j++){{
  const [a,b]=data.bonds[j], p=nodePoint(frame,a), q=nodePoint(frame,b);
  if(tension.has(j)||compression.has(j)){{
   const mx=(p[0]+q[0])/2,my=(p[1]+q[1])/2, dx=q[0]-p[0],dy=q[1]-p[1],norm=Math.hypot(dx,dy)||1;
   bctx.strokeStyle=tension.has(j)?'#c62828':'#1565c0';bctx.lineWidth=2;
   bctx.beginPath();bctx.moveTo(mx-dy/norm*4,my+dx/norm*4);bctx.lineTo(mx+dy/norm*4,my-dx/norm*4);bctx.stroke();continue
  }}
  const strain=frame.strain[j], limit=strain>=0?data.tensile_strain_limit:data.compressive_strain_limit;
  const alpha=Math.min(.9,.15+.65*Math.abs(strain)/limit);
  bctx.strokeStyle=strain>0?`rgba(198,40,40,${{alpha}})`:strain<0?`rgba(21,101,192,${{alpha}})`:'#d8dee5';
  bctx.lineWidth=.8;bctx.beginPath();bctx.moveTo(...p);bctx.lineTo(...q);bctx.stroke()
 }}
 bctx.fillStyle='#202124';bctx.font='14px system-ui';bctx.fillText('Деформированная решётка и состояние связей',18,24);
 const contact=data.punch_nodes.map(index=>nodePoint(frame,index));
 const contactX=contact.reduce((sum,p)=>sum+p[0],0)/contact.length;
 const contactY=contact.reduce((sum,p)=>sum+p[1],0)/contact.length;
 const peakForce=Math.max(...data.history.map(row=>row.force_N))||1;
 const relativeForce=Math.min(1,Math.max(0,data.history[i].force_N/peakForce));
 const arrowLength=20+30*Math.sqrt(relativeForce), arrowTop=contactY-arrowLength;
 bctx.strokeStyle='#e65100';bctx.fillStyle='#e65100';bctx.lineWidth=3;
 bctx.beginPath();bctx.moveTo(contactX,arrowTop);bctx.lineTo(contactX,contactY-7);bctx.stroke();
 bctx.beginPath();bctx.moveTo(contactX,contactY);bctx.lineTo(contactX-6,contactY-9);bctx.lineTo(contactX+6,contactY-9);bctx.closePath();bctx.fill();
 for(const [x,y] of contact){{bctx.beginPath();bctx.arc(x,y,3.5,0,Math.PI*2);bctx.fill()}}
 bctx.fillStyle='#a33b00';bctx.font='12px system-ui';bctx.fillText('Пуансон',contactX+10,arrowTop+8);
}}
function drawCurve(i){{
 const left=58,top=40,right=20,bottom=58,w=curve.width-left-right,h=curve.height-top-bottom;
 const maxX=Math.max(...data.history.map(r=>r.displacement_m))||1,maxY=Math.max(...data.history.map(r=>r.force_N))*1.1||1;
 cctx.clearRect(0,0,curve.width,curve.height);cctx.strokeStyle='#333';cctx.lineWidth=1;
 cctx.beginPath();cctx.moveTo(left,top);cctx.lineTo(left,top+h);cctx.lineTo(left+w,top+h);cctx.stroke();
 cctx.strokeStyle='#1769aa';cctx.lineWidth=2;cctx.beginPath();
 data.history.forEach((r,j)=>{{const x=left+r.displacement_m/maxX*w,y=top+(1-r.force_N/maxY)*h;j?cctx.lineTo(x,y):cctx.moveTo(x,y)}});cctx.stroke();
 const row=data.history[i],x=left+row.displacement_m/maxX*w,y=top+(1-row.force_N/maxY)*h;
 cctx.fillStyle='#e65100';cctx.beginPath();cctx.arc(x,y,5,0,Math.PI*2);cctx.fill();
 cctx.fillStyle='#202124';cctx.font='14px system-ui';cctx.fillText('Реакция пуансона–перемещение',left,24);
 cctx.font='12px system-ui';cctx.fillText('Перемещение, м',left+w/2-40,curve.height-15);
 cctx.save();cctx.translate(16,top+h/2);cctx.rotate(-Math.PI/2);cctx.fillText('Реакция, Н',-35,0);cctx.restore()
}}
function formatForce(force){{return force!==0&&Math.abs(force)<0.1?force.toExponential(2):force.toFixed(1)}}
function render(){{const i=Number(slider.value),row=data.history[i],frame=data.frames[i];
 const stopped=i===data.history.length-1&&data.termination_reason==='loss_of_load_capacity_after_bond_failure';
 drawBeam(i);drawCurve(i);readout.textContent=`Шаг ${{row.step}}/${{data.history.length}} · δ=${{(row.displacement_m*1000).toFixed(3)}} мм · R пуансона=${{formatForce(row.force_N)}} Н · разрывов: ${{frame.failed_tension.length}} растяжение, ${{frame.failed_compression.length}} сжатие${{stopped?' · потеря несущей способности':''}}`}}
slider.addEventListener('input',render);render();
let timer=null;document.getElementById('play').addEventListener('click',event=>{{
 if(timer){{clearInterval(timer);timer=null;event.target.textContent='Пуск';return}}
 event.target.textContent='Пауза';timer=setInterval(()=>{{
  if(Number(slider.value)>=Number(slider.max)){{
   if(data.termination_reason!=='maximum_displacement_reached'){{clearInterval(timer);timer=null;event.target.textContent='Пуск';render();return}}
   slider.value='0'
  }}else slider.value=String(Number(slider.value)+1);
  render()
 }},100)
}});
</script></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=1.0, help="Beam length in metres")
    parser.add_argument("--height", type=float, default=0.1, help="Beam height in metres")
    parser.add_argument("--width", type=float, default=0.1, help="Out-of-plane beam width in metres")
    parser.add_argument("--youngs-modulus", type=float, default=9.0e9, help="Young's modulus in Pa")
    parser.add_argument("--tensile-strength", type=float, default=0.7e6, help="Tensile strength in Pa")
    parser.add_argument("--compressive-strength", type=float, default=3.0e6, help="Compressive strength in Pa")
    parser.add_argument("--max-displacement", type=float, default=0.005, help="Punch travel in metres")
    parser.add_argument("--steps", type=int, default=100, help="Number of displacement load steps")
    parser.add_argument("--layers", type=int, default=11, help="Odd number of lattice rows through beam depth")
    parser.add_argument("--output", type=Path, default=Path("results_2d"), help="Output directory")
    args = parser.parse_args()
    parameters = Parameters(
        length=args.length,
        height=args.height,
        width=args.width,
        youngs_modulus=args.youngs_modulus,
        tensile_strength=args.tensile_strength,
        compressive_strength=args.compressive_strength,
        max_displacement=args.max_displacement,
        steps=args.steps,
        layers=args.layers,
    )
    print(json.dumps(run_simulation(parameters, args.output), indent=2))


if __name__ == "__main__":
    main()

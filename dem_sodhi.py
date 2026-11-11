#!/usr/bin/env python3
"""Quasi-static 2D DEM of a clamped ice beam with compressive crack-face contact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Parameters:
    length: float = 1.0
    height: float = 0.1
    width: float = 0.1
    youngs_modulus: float = 9.0e9
    poisson_ratio: float = 0.30
    density: float = 917.0
    tensile_strength: float = 0.7e6
    compressive_strength: float = 3.0e6
    max_displacement: float = 0.005
    steps: int = 100
    blocks: int = 41
    fibers: int = 21


@dataclass(frozen=True)
class Spring:
    dofs: tuple[int, ...]
    coefficients: tuple[float, ...]
    stiffness: float
    area: float = 0.0
    z: float = 0.0
    x: float = 0.0
    is_fiber: bool = False


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
        raise RuntimeError("The beam has an unconstrained degree of freedom")

    preconditioned = [r / d for r, d in zip(residual, diagonal)]
    direction = preconditioned[:]
    residual_product = inner(residual, preconditioned)
    for _ in range(max(500, 4 * len(rhs))):
        product = multiply(direction)
        denominator = inner(direction, product)
        if denominator <= 0.0:
            raise RuntimeError("The beam stiffness matrix is not positive definite")
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


def build_springs(parameters: Parameters) -> tuple[list[Spring], list[int], int, float]:
    if parameters.blocks < 5 or parameters.blocks % 2 == 0:
        raise ValueError("blocks must be an odd integer of at least 5")
    if parameters.fibers < 3 or parameters.fibers % 2 == 0:
        raise ValueError("fibers must be an odd integer of at least 3")

    dx = parameters.length / (parameters.blocks - 1)
    dz = parameters.height / parameters.fibers
    shear_modulus = parameters.youngs_modulus / (2.0 * (1.0 + parameters.poisson_ratio))
    shear_correction = 5.0 / 6.0
    springs: list[Spring] = []
    for block in range(parameters.blocks - 1):
        left, right = 3 * block, 3 * (block + 1)
        interface_x = (block + 0.5) * dx
        for fiber in range(parameters.fibers):
            z = -parameters.height / 2.0 + (fiber + 0.5) * dz
            area = parameters.width * dz
            springs.append(
                Spring(
                    (left, left + 2, right, right + 2),
                    (-1.0, z, 1.0, -z),
                    parameters.youngs_modulus * area / dx,
                    area,
                    z,
                    interface_x,
                    True,
                )
            )
        springs.append(
            Spring(
                (left + 1, left + 2, right + 1, right + 2),
                (-1.0, -dx / 2.0, 1.0, -dx / 2.0),
                shear_correction * shear_modulus * parameters.width * parameters.height / dx,
            )
        )

    fixed_dofs = [0, 1, 2]
    last_block = parameters.blocks - 1
    fixed_dofs.extend((3 * last_block, 3 * last_block + 1, 3 * last_block + 2))
    center_block = (parameters.blocks - 1) // 2
    punch_dof = 3 * center_block + 1
    return springs, fixed_dofs, punch_dof, dx


def load_schedule(parameters: Parameters) -> list[float]:
    if parameters.steps < 1:
        raise ValueError("steps must be positive")
    if parameters.steps < 3:
        return [
            parameters.max_displacement * step / parameters.steps
            for step in range(1, parameters.steps + 1)
        ]
    crack_estimate = (
        parameters.tensile_strength
        * parameters.length**2
        / (12.0 * parameters.youngs_modulus * parameters.height)
    )
    crack_region = min(parameters.max_displacement, 1.5 * crack_estimate)
    compression_region = min(
        parameters.max_displacement,
        max(crack_region, 1.5 * crack_estimate * parameters.compressive_strength / parameters.tensile_strength),
    )
    crack_steps = parameters.steps // 3
    compression_steps = parameters.steps // 3
    post_steps = parameters.steps - crack_steps - compression_steps
    schedule = [
        crack_region * step / crack_steps
        for step in range(1, crack_steps + 1)
    ]
    schedule.extend(
        crack_region + (compression_region - crack_region) * step / compression_steps
        for step in range(1, compression_steps + 1)
    )
    schedule.extend(
        compression_region
        + (parameters.max_displacement - compression_region) * step / post_steps
        for step in range(1, post_steps + 1)
    )
    return schedule


def run_simulation(parameters: Parameters, output_dir: Path) -> dict[str, object]:
    if min(parameters.length, parameters.height, parameters.width) <= 0.0:
        raise ValueError("Beam dimensions must be positive")
    if parameters.youngs_modulus <= 0.0 or parameters.tensile_strength <= 0.0:
        raise ValueError("Young's modulus and tensile strength must be positive")
    if parameters.compressive_strength <= 0.0:
        raise ValueError("Compressive strength must be positive")
    if not 0.0 <= parameters.poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must be in [0, 0.5)")

    springs, fixed_dofs, punch_dof, dx = build_springs(parameters)
    dof_count = 3 * parameters.blocks
    fiber_indices = [i for i, spring in enumerate(springs) if spring.is_fiber]
    free_dofs = [dof for dof in range(dof_count) if dof not in fixed_dofs and dof != punch_dof]
    free_map = {dof: index for index, dof in enumerate(free_dofs)}
    displacement = [0.0] * dof_count
    intact = [True] * len(springs)
    tensile_failed = [False] * len(springs)
    crushed = [False] * len(springs)
    contact_active = [False] * len(springs)
    crush_contact_active = [False] * len(springs)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    frames: list[dict[str, object]] = []
    first_crack: dict[str, float] | None = None
    max_compressive_stress = 0.0
    stabilizer = parameters.youngs_modulus * parameters.width * parameters.height * 1e-10
    schedule = load_schedule(parameters)

    def extension(spring: Spring, values: list[float]) -> float:
        return sum(
            coefficient * values[dof]
            for dof, coefficient in zip(spring.dofs, spring.coefficients)
        )

    def spring_is_active(index: int) -> bool:
        return (
            not springs[index].is_fiber
            or intact[index]
            or (tensile_failed[index] and contact_active[index])
            or (crushed[index] and crush_contact_active[index])
        )

    def multiply(free_values: list[float]) -> list[float]:
        result = [stabilizer * value for value in free_values]
        for index, spring in enumerate(springs):
            if not spring_is_active(index):
                continue
            if crushed[index]:
                continue
            spring_extension = sum(
                coefficient * (free_values[free_map[dof]] if dof in free_map else 0.0)
                for dof, coefficient in zip(spring.dofs, spring.coefficients)
            )
            force = spring.stiffness * spring_extension
            for dof, coefficient in zip(spring.dofs, spring.coefficients):
                free_index = free_map.get(dof)
                if free_index is not None:
                    result[free_index] += force * coefficient
        return result

    for step, punch_displacement in enumerate(schedule, start=1):
        displacement[punch_dof] = -punch_displacement
        for dof in fixed_dofs:
            displacement[dof] = 0.0
        reference = displacement[:]
        for index in fiber_indices:
            if crushed[index]:
                crush_contact_active[index] = extension(springs[index], displacement) < 0.0
            elif tensile_failed[index]:
                contact_active[index] = extension(springs[index], displacement) < 0.0
            else:
                contact_active[index] = False
                crush_contact_active[index] = False

        for _ in range(len(fiber_indices) + 10):
            diagonal = [stabilizer] * len(free_dofs)
            known_product = [0.0] * len(free_dofs)
            for index, spring in enumerate(springs):
                if not spring_is_active(index):
                    continue
                if crushed[index]:
                    capped_force = -parameters.compressive_strength * spring.area
                    for dof, coefficient in zip(spring.dofs, spring.coefficients):
                        free_index = free_map.get(dof)
                        if free_index is not None:
                            known_product[free_index] += capped_force * coefficient
                    continue
                known_extension = sum(
                    coefficient * (displacement[dof] if dof not in free_map else 0.0)
                    for dof, coefficient in zip(spring.dofs, spring.coefficients)
                )
                known_force = spring.stiffness * known_extension
                for dof, coefficient in zip(spring.dofs, spring.coefficients):
                    free_index = free_map.get(dof)
                    if free_index is not None:
                        diagonal[free_index] += spring.stiffness * coefficient**2
                        known_product[free_index] += known_force * coefficient

            rhs = [
                -known_product[i] + stabilizer * reference[dof]
                for i, dof in enumerate(free_dofs)
            ]
            solution = conjugate_gradient(
                multiply,
                rhs,
                diagonal,
                [displacement[dof] for dof in free_dofs],
            )
            for dof, value in zip(free_dofs, solution):
                displacement[dof] = value

            broke_bond = False
            for index in fiber_indices:
                spring = springs[index]
                if crushed[index]:
                    continue
                bond_stress = spring.stiffness * extension(spring, displacement) / spring.area
                if tensile_failed[index]:
                    if contact_active[index] and bond_stress < -parameters.compressive_strength:
                        crushed[index] = True
                        contact_active[index] = False
                        crush_contact_active[index] = True
                        max_compressive_stress = max(
                            max_compressive_stress, parameters.compressive_strength
                        )
                        broke_bond = True
                    continue
                if not intact[index]:
                    continue
                if bond_stress > parameters.tensile_strength:
                    if first_crack is None:
                        pre_crack_force = 0.0
                        for active_index, active_spring in enumerate(springs):
                            if not spring_is_active(active_index):
                                continue
                            active_force = active_spring.stiffness * extension(
                                active_spring, displacement
                            )
                            for dof, coefficient in zip(
                                active_spring.dofs, active_spring.coefficients
                            ):
                                if dof == punch_dof:
                                    pre_crack_force += active_force * coefficient
                    intact[index] = False
                    tensile_failed[index] = True
                    contact_active[index] = False
                    broke_bond = True
                    if first_crack is None:
                        first_crack = {
                            "displacement_m": punch_displacement,
                            "force_N": abs(pre_crack_force),
                            "x_m": spring.x,
                            "z_m": spring.z,
                        }
                elif bond_stress < -parameters.compressive_strength:
                    intact[index] = False
                    crushed[index] = True
                    crush_contact_active[index] = True
                    max_compressive_stress = max(
                        max_compressive_stress, parameters.compressive_strength
                    )
                    broke_bond = True
            if broke_bond:
                continue

            contact_changed = False
            for index in fiber_indices:
                if crushed[index]:
                    compressed = extension(springs[index], displacement) < 0.0
                    if compressed != crush_contact_active[index]:
                        crush_contact_active[index] = compressed
                        contact_changed = True
                elif tensile_failed[index]:
                    compressed = extension(springs[index], displacement) < 0.0
                    if compressed != contact_active[index]:
                        contact_active[index] = compressed
                        contact_changed = True
            if not contact_changed:
                break
        else:
            raise RuntimeError(f"Damage/contact iteration did not converge at load step {step}")

        punch_force = 0.0
        for index, spring in enumerate(springs):
            if spring_is_active(index):
                force = (
                    -parameters.compressive_strength * spring.area
                    if crushed[index]
                    else spring.stiffness * extension(spring, displacement)
                )
                for dof, coefficient in zip(spring.dofs, spring.coefficients):
                    if dof == punch_dof:
                        punch_force += force * coefficient

        for index in fiber_indices:
            if spring_is_active(index) and not crushed[index]:
                stress = springs[index].stiffness * extension(springs[index], displacement) / springs[index].area
                max_compressive_stress = max(max_compressive_stress, -stress)

        history.append(
            {
                "step": step,
                "displacement_m": punch_displacement,
                "force_N": abs(punch_force),
                "broken_bonds": sum(tensile_failed[index] for index in fiber_indices),
                "crushed_bonds": sum(crushed[index] for index in fiber_indices),
                "active_contacts": sum(
                    contact_active[index] or crush_contact_active[index]
                    for index in fiber_indices
                ),
            }
        )
        states: list[str] = []
        stresses: list[float] = []
        for index in fiber_indices:
            if crushed[index]:
                states.append("crushed_contact" if crush_contact_active[index] else "crushed_open")
                stress = -parameters.compressive_strength if crush_contact_active[index] else 0.0
            elif tensile_failed[index]:
                states.append("compression_contact" if contact_active[index] else "open_crack")
                stress = (
                    springs[index].stiffness
                    * extension(springs[index], displacement)
                    / springs[index].area
                    if contact_active[index]
                    else 0.0
                )
            else:
                bond_stress = springs[index].stiffness * extension(springs[index], displacement) / springs[index].area
                states.append(
                    "compression"
                    if bond_stress < -1e-6
                    else "tension"
                    if bond_stress > 1e-6
                    else "neutral"
                )
                stress = bond_stress
            stresses.append(stress)
        frames.append({"displacement": displacement[:], "states": states, "stresses": stresses})

    csv_path = output_dir / "force_displacement.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    first_crack_force = (
        4.0
        * parameters.tensile_strength
        * parameters.width
        * parameters.height**2
        / (3.0 * parameters.length)
    )
    sodhi_beta_one_third = (
        8.0
        * parameters.compressive_strength
        * parameters.width
        * parameters.height**2
        / (9.0 * parameters.length)
    )
    peak = max(history, key=lambda row: float(row["force_N"]))
    post_crack = [row for row in history if int(row["broken_bonds"]) > 0]
    summary: dict[str, object] = {
        "model": "2D rigid cross-section DEM with fiber bonds and unilateral compression contact",
        "parameters": parameters.__dict__,
        "mesh": {
            "rigid_blocks": parameters.blocks,
            "fiber_bonds_per_interface": parameters.fibers,
            "interfaces": parameters.blocks - 1,
            "dx_m": dx,
        },
        "first_tensile_crack": first_crack,
        "peak_force_N": float(peak["force_N"]),
        "peak_force_displacement_m": float(peak["displacement_m"]),
        "post_crack_peak_force_N": max((float(row["force_N"]) for row in post_crack), default=0.0),
        "final_broken_bonds": int(history[-1]["broken_bonds"]),
        "final_active_compression_contacts": int(history[-1]["active_contacts"]),
        "final_crushed_bonds": sum(crushed[index] for index in fiber_indices),
        "max_compressive_bond_stress_Pa": max_compressive_stress,
        "compressive_strength_reached": max_compressive_stress >= parameters.compressive_strength,
        "sodhi_reference": {
            "first_crack_force_Ff_N": first_crack_force,
            "post_failure_force_at_beta_1_3_Fc_N": sodhi_beta_one_third,
        },
        "artifacts": [
            csv_path.name,
            "force_displacement.svg",
            "crack_state.svg",
            "beam_animation.html",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_force_svg(history, output_dir / "force_displacement.svg")
    write_crack_svg(springs, frames[-1]["states"], parameters, output_dir / "crack_state.svg")
    write_animation_html(
        springs,
        frames,
        history,
        parameters,
        output_dir / "beam_animation.html",
    )
    return summary


def write_force_svg(history: list[dict[str, float | int]], path: Path) -> None:
    width, height = 900, 520
    left, top, right, bottom = 85, 35, 30, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    max_x = max(float(row["displacement_m"]) for row in history)
    max_y = max(float(row["force_N"]) for row in history) or 1.0
    points = " ".join(
        f"{left + float(row['displacement_m']) / max_x * plot_width:.2f},"
        f"{top + (1.0 - float(row['force_N']) / max_y) * plot_height:.2f}"
        for row in history
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Ice beam DEM: force-displacement</text>
<path d="M {left} {top} V {top + plot_height} H {left + plot_width}" fill="none" stroke="#333" stroke-width="2"/>
<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="2.5"/>
<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif">Center displacement (m)</text>
<text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif">Reaction force (N)</text>
<text x="{left}" y="{height - 45}" font-family="sans-serif" font-size="12">0</text>
<text x="{left + plot_width - 35}" y="{height - 45}" font-family="sans-serif" font-size="12">{max_x:.4g}</text>
<text x="{left - 55}" y="{top + 5}" font-family="sans-serif" font-size="12">{max_y:.4g}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def bond_color(state: str) -> str:
    return {
        "tension": "#d84315",
        "compression": "#1565c0",
        "compression_contact": "#00a6a6",
        "open_crack": "#c62828",
        "crushed_contact": "#6a1b9a",
        "crushed_open": "#795548",
        "neutral": "#aeb7c2",
    }[state]


STATE_CODES = {
    "tension": "t",
    "compression": "c",
    "compression_contact": "a",
    "open_crack": "o",
    "crushed_contact": "p",
    "crushed_open": "x",
    "neutral": "n",
}


def write_crack_svg(
    springs: list[Spring],
    states: list[str],
    parameters: Parameters,
    path: Path,
) -> None:
    width, height = 1200, 330
    margin_x, margin_y = 45, 70
    scale_x = (width - 2 * margin_x) / parameters.length
    scale_z = (height - 2 * margin_y) / parameters.height
    scale = min(scale_x, scale_z)
    x_offset = (width - parameters.length * scale) / 2
    y_offset = (height - parameters.height * scale) / 2
    lines: list[str] = []
    labels = {
        "tension": "Растяжение",
        "compression": "Сжатие",
        "compression_contact": "Контакт трещины в сжатии",
        "open_crack": "Открытая трещина",
        "crushed_contact": "Раздавлено, контакт",
        "crushed_open": "Раздавлено, без контакта",
        "neutral": "Почти ненагружено",
    }
    fiber_position = 0
    for index, spring in enumerate(springs):
        if not spring.is_fiber:
            continue
        state = states[fiber_position]
        fiber_position += 1
        x = x_offset + spring.x * scale
        y = height - y_offset - (parameters.height / 2.0 + spring.z) * scale
        color = bond_color(state)
        dash = ' stroke-dasharray="3 3"' if state in ("open_crack", "crushed_open") else ""
        lines.append(
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 5:.2f}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="2.4"{dash}/>'
        )
    legend = []
    for column, state in enumerate(sorted(labels)):
        x = 55 + (column % 4) * 285
        y = 282 + (column // 4) * 23
        legend.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{bond_color(state)}" '
            f'stroke-width="4"/><text x="{x + 30}" y="{y + 4}" font-family="sans-serif" '
            f'font-size="13">{labels[state]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Связи балки: растяжение, сжатие и состояние трещины</text>
{"".join(lines)}
{"".join(legend)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_animation_html(
    springs: list[Spring],
    frames: list[dict[str, object]],
    history: list[dict[str, float | int]],
    parameters: Parameters,
    path: Path,
) -> None:
    fibers = [
        {
            "block": spring.dofs[0] // 3,
            "z": spring.z,
            "x": spring.x,
        }
        for spring in springs
        if spring.is_fiber
    ]
    frame_data = [
        {
            "displacement": frame["displacement"],
            "states": "".join(STATE_CODES[state] for state in frame["states"]),
            "stresses": [round(stress) for stress in frame["stresses"]],
            "force": row["force_N"],
            "travel": row["displacement_m"],
            "step": row["step"],
        }
        for frame, row in zip(frames, history)
    ]
    payload = json.dumps(
        {
            "frames": frame_data,
            "fibers": fibers,
            "length": parameters.length,
            "height": parameters.height,
            "blocks": parameters.blocks,
            "width": parameters.width,
        },
        separators=(",", ":"),
    )
    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DEM-модель ледяной балки — анимация</title>
<style>
body {{ margin: 0; padding: 24px; color: #17212b; font: 15px system-ui, sans-serif; background: #f3f6f9; }}
main {{ max-width: 1100px; margin: auto; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 12px #17212b18; }}
h1 {{ font-size: 21px; margin: 0 0 8px; }}
p {{ color: #506070; }}
svg {{ width: 100%; height: auto; border: 1px solid #d5dee7; border-radius: 8px; background: #fbfdff; }}
.controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 14px; }}
button {{ border: 0; border-radius: 6px; padding: 8px 14px; color: white; background: #1769aa; cursor: pointer; }}
input[type=range] {{ flex: 1; min-width: 180px; }}
#readout {{ font-variant-numeric: tabular-nums; min-width: 280px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0 0; color: #394b5a; font-size: 13px; }}
.legend span::before {{ content: ""; display: inline-block; width: 14px; margin-right: 5px; border-top: 4px solid var(--c); vertical-align: middle; }}
</style>
</head>
<body>
<main>
<h1>Нагружение ледяной балки — DEM</h1>
<p>Анимация квазистатических шагов расчёта. Деформация увеличена в 4 раза для наглядности; значения перемещения и нагрузки указаны без увеличения. Цветом показано состояние связей по высоте балки.</p>
<svg id="scene" viewBox="0 0 1550 500" role="img" aria-label="Анимация нагружения балки и график силы"></svg>
<div class="controls">
<button id="play" type="button">Пауза</button>
<input id="frame" type="range" min="0" max="{len(frame_data) - 1}" value="0">
<label for="speed">Скорость</label>
<select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="0.5">0.5×</option></select>
<output id="readout"></output>
</div>
<div class="legend">
<span style="--c:#d84315">Растяжение</span>
<span style="--c:#1565c0">Сжатие</span>
<span style="--c:#00a6a6">Контакт трещины</span>
<span style="--c:#c62828">Открытая трещина</span>
<span style="--c:#6a1b9a">Раздавлено, контакт</span>
<span style="--c:#795548">Раздавлено, разомкнуто</span>
<span>Стрелки: → растяжение · ← сжатие (длина пропорциональна |σx|)</span>
</div>
</main>
<script>
const data = {payload};
const svg = document.getElementById("scene");
const slider = document.getElementById("frame");
const button = document.getElementById("play");
const speed = document.getElementById("speed");
const readout = document.getElementById("readout");
const NS = "http://www.w3.org/2000/svg";
const colors = {{
  t: "#d84315", c: "#1565c0", a: "#00a6a6",
  o: "#c62828", p: "#6a1b9a", x: "#795548", n: "#aeb7c2"
}};
const stressTensionColor = "#c62828", stressCompressionColor = "#0d47a1";
const fills = {{
  t: "#f7b49e", c: "#a9c8ef", a: "#8bd9d5",
  o: "#ffe0dc", p: "#d6b7e7", x: "#d9c5b9", n: "#dceef8"
}};
const beamTop = 118, beamHeight = 210, left = 60, beamWidth = 980;
const scaleX = beamWidth / data.length, scaleZ = beamHeight / data.height;
const deformationScale = 4;
const fibersPerInterface = data.fibers.length / (data.blocks - 1);
const layerHeight = data.height / fibersPerInterface;
const plot = {{left: 1120, top: 125, width: 370, height: 260}};
const maxTravel = Math.max(...data.frames.map(frame => frame.travel));
const maxForce = Math.max(...data.frames.map(frame => frame.force));
let current = 0, playing = true, lastTime = 0, accumulator = 0;
const add = (name, attrs, parent = svg) => {{
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  parent.appendChild(element);
  return element;
}};
function point(frame, block, localX, localZ) {{
  const u = frame.displacement[3 * block] * deformationScale;
  const v = frame.displacement[3 * block + 1] * deformationScale;
  const angle = frame.displacement[3 * block + 2] * deformationScale;
  return [
    left + (block * data.length / (data.blocks - 1) + u + localX - localZ * angle) * scaleX,
    beamTop + beamHeight / 2 - (v + localZ + localX * angle) * scaleZ
  ];
}}
function interfaceState(frame, block, layer) {{
  const candidates = [];
  if (block > 0) candidates.push(frame.states[(block - 1) * fibersPerInterface + layer]);
  if (block < data.blocks - 1) candidates.push(frame.states[block * fibersPerInterface + layer]);
  const rank = {{o:7, p:6, x:5, a:4, t:3, c:3, n:1}};
  return candidates.sort((a,b)=>(rank[b]||0)-(rank[a]||0))[0] || "n";
}}
function draw(frameIndex) {{
  const frame = data.frames[frameIndex];
  svg.replaceChildren();
  add("rect", {{x:0,y:0,width:1550,height:500,fill:"#fbfdff"}});
  add("text", {{x:775,y:35,"text-anchor":"middle","font-size":19,fill:"#17212b"}}, svg)
    .textContent = "Шаг " + frame.step + " / " + data.frames.length;
  const beam = add("g", {{}});
  const halfBlock = data.length / (2 * (data.blocks - 1));
  for (let block = 0; block < data.blocks; block++) {{
    const halfWidth = block === 0 || block === data.blocks - 1 ? halfBlock : halfBlock;
    for (let layer = 0; layer < fibersPerInterface; layer++) {{
      const zCenter = -data.height/2 + (layer + 0.5) * layerHeight;
      const state = interfaceState(frame, block, layer);
      const corners = [
        [-halfWidth, zCenter-layerHeight/2], [-halfWidth, zCenter+layerHeight/2],
        [halfWidth, zCenter+layerHeight/2], [halfWidth, zCenter-layerHeight/2]
      ].map(([localX, localZ]) => point(frame, block, localX, localZ).join(",")).join(" ");
      add("polygon", {{points:corners,fill:fills[state] || fills.n,stroke:"#ffffff","stroke-width":0.45}}, beam);
    }}
  }}
  const stressVectors = add("g", {{"fill":"none","stroke-width":2.2,"stroke-linecap":"round"}});
  add("defs", {{}}).innerHTML = `
    <marker id="stressTensionArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="${{stressTensionColor}}"/></marker>
    <marker id="stressCompressionArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="${{stressCompressionColor}}"/></marker>`;
  const vectorBlockStride = Math.max(1, Math.ceil((data.blocks - 1) / 12));
  const vectorLayerStride = Math.max(1, Math.ceil(fibersPerInterface / 6));
  const tensionLimit = Math.max(...data.frames.flatMap(item=>item.stresses.filter(value=>value>0)), 1);
  const compressionLimit = Math.max(...data.frames.flatMap(item=>item.stresses.filter(value=>value<0).map(value=>-value)), 1);
  for (let block=0;block<data.blocks-1;block+=vectorBlockStride) {{
    for (let layer=0;layer<fibersPerInterface;layer+=vectorLayerStride) {{
      const index=block*fibersPerInterface+layer;
      const stress=frame.stresses[index];
      if (Math.abs(stress)<1) continue;
      const state=frame.states[index];
      if (state==="o" || state==="x") continue;
      const fiber=data.fibers[index];
      const anchor=point(frame,block,halfBlock,fiber.z);
      const tensile=stress>0;
      const limit=tensile?tensionLimit:compressionLimit;
      const length=5+11*Math.min(1,Math.abs(stress)/limit);
      const startX=tensile?anchor[0]-length/2:anchor[0]+length/2;
      const endX=tensile?anchor[0]+length/2:anchor[0]-length/2;
      const color=tensile?stressTensionColor:stressCompressionColor;
      const vector=add("line", {{
        x1:startX,y1:anchor[1],x2:endX,y2:anchor[1],stroke:color,
        "marker-end":tensile?"url(#stressTensionArrow)":"url(#stressCompressionArrow)"
      }},stressVectors);
      add("title",{{}},vector).textContent=`σx = ${{(stress/1e6).toFixed(3)}} MPa`;
    }}
  }}
  const cracks = add("g", {{"stroke-linecap":"round","fill":"none"}});
  for (let block = 0; block < data.blocks - 1; block++) {{
    for (let layer = 0; layer < fibersPerInterface; layer++) {{
      const index = block * fibersPerInterface + layer;
      const state = frame.states[index];
      if (state !== "o" && state !== "a" && state !== "p" && state !== "x") continue;
      const fiber = data.fibers[index];
      const leftPoint = point(frame, block, halfBlock, fiber.z);
      const rightPoint = point(frame, block+1, -halfBlock, fiber.z);
      const midX = (leftPoint[0]+rightPoint[0])/2;
      const midY = (leftPoint[1]+rightPoint[1])/2;
      const crackColor = colors[state];
      const crackWidth = state === "a" || state === "p" ? 3.5 : 2.4;
      const crackLength = state === "o" || state === "x" ? 4 : 2.5;
      add("line", {{x1:midX,y1:midY-crackLength,x2:midX,y2:midY+crackLength,stroke:crackColor,"stroke-width":crackWidth}}, cracks);
    }}
  }}
  const bounds = add("g", {{fill:"none",stroke:"#315d7a","stroke-width":2}});
  for (const localZ of [-data.height/2, data.height/2]) {{
    const pts = Array.from({{length:data.blocks}}, (_,block)=>point(frame,block,0,localZ).join(",")).join(" ");
    add("polyline", {{points:pts}}, bounds);
  }}
  const baseline = add("line", {{x1:left,y1:beamTop+beamHeight+55,x2:left+beamWidth,y2:beamTop+beamHeight+55,stroke:"#687887","stroke-width":1.5}});
  const statesLayer = add("g", {{fill:"none"}});
  for (const block of [0,data.blocks-1]) {{
    const clampX = point(frame,block,0,0)[0] + (block === 0 ? -12 : 2);
    add("rect", {{x:clampX,y:beamTop+beamHeight/2-beamHeight/2,width:10,height:beamHeight,fill:"#8a9299",stroke:"#42484e","stroke-width":1}}, statesLayer);
    for (let hatch=0;hatch<8;hatch++) {{
      const y=beamTop+20+hatch*23;
      add("line", {{x1:clampX,y1:y,x2:clampX+(block===0?-8:18),y2:y+12,stroke:"#42484e","stroke-width":1}}, statesLayer);
    }}
  }}
  const punchX = left + data.length/2 * scaleX;
  const centerTop = point(frame,(data.blocks-1)/2,0,data.height/2)[1];
  const punchY = centerTop - 8;
  add("line", {{x1:punchX,y1:65,x2:punchX,y2:punchY-8,stroke:"#c62828","stroke-width":3,"marker-end":"url(#arrow)"}});
  add("defs", {{}}).innerHTML = '<marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c62828"/></marker>';
  add("text", {{x:punchX+12,y:83,fill:"#c62828","font-size":15}}).textContent = "F";

  const chart = add("g", {{}});
  const chartBottom = plot.top + plot.height;
  const chartRight = plot.left + plot.width;
  const chartX = travel => plot.left + travel / maxTravel * plot.width;
  const chartY = force => chartBottom - force / maxForce * plot.height;
  add("text", {{x:plot.left+plot.width/2,y:72,"text-anchor":"middle","font-size":18,fill:"#17212b"}}, chart)
    .textContent = "Сила — перемещение";
  add("line", {{x1:plot.left,y1:plot.top,x2:plot.left,y2:chartBottom,stroke:"#526575","stroke-width":1.5}}, chart);
  add("line", {{x1:plot.left,y1:chartBottom,x2:chartRight,y2:chartBottom,stroke:"#526575","stroke-width":1.5}}, chart);
  for (let tick=0;tick<=4;tick++) {{
    const force=maxForce*tick/4, y=chartY(force);
    add("line", {{x1:plot.left-4,y1:y,x2:chartRight,y2:y,stroke:"#e3eaf0","stroke-width":1}}, chart);
    add("text", {{x:plot.left-8,y:y+4,"text-anchor":"end","font-size":11,fill:"#526575"}}, chart).textContent=force.toFixed(0);
    const travel=maxTravel*tick/4, x=chartX(travel);
    add("line", {{x1:x,y1:chartBottom,x2:x,y2:chartBottom+4,stroke:"#526575","stroke-width":1}}, chart);
    add("text", {{x:x,y:chartBottom+18,"text-anchor":"middle","font-size":10,fill:"#526575"}}, chart).textContent=(travel*1000).toFixed(1);
  }}
  const curve=data.frames.map(point=>`${{chartX(point.travel).toFixed(2)}},${{chartY(point.force).toFixed(2)}}`).join(" ");
  add("polyline", {{points:curve,fill:"none",stroke:"#b6c8d8","stroke-width":2}}, chart);
  const passedCurve=data.frames.slice(0,frameIndex+1).map(point=>`${{chartX(point.travel).toFixed(2)}},${{chartY(point.force).toFixed(2)}}`).join(" ");
  add("polyline", {{points:passedCurve,fill:"none",stroke:"#1769aa","stroke-width":3}}, chart);
  const markerX=chartX(frame.travel), markerY=chartY(frame.force);
  add("line", {{x1:markerX,y1:plot.top,x2:markerX,y2:chartBottom,stroke:"#c62828","stroke-width":1.2,"stroke-dasharray":"4 4"}}, chart);
  add("circle", {{cx:markerX,cy:markerY,r:6,fill:"#c62828",stroke:"white","stroke-width":2}}, chart);
  add("text", {{x:plot.left+plot.width/2,y:chartBottom+40,"text-anchor":"middle","font-size":13,fill:"#17212b"}}, chart).textContent="Перемещение, мм";
  add("text", {{x:plot.left-45,y:plot.top+plot.height/2,transform:`rotate(-90 ${{plot.left-45}} ${{plot.top+plot.height/2}})`,"text-anchor":"middle","font-size":13,fill:"#17212b"}}, chart).textContent="Сила, Н";
  const damagedSections = Array.from({{length:data.blocks-1}},(_,block)=>
    [...frame.states.slice(block*fibersPerInterface,(block+1)*fibersPerInterface)].some(s=>"oapx".includes(s))
  ).filter(Boolean).length;
  readout.textContent = `u = ${{(frame.travel*1000).toFixed(3)}} мм · F = ${{frame.force.toFixed(1)}} Н · сечения с трещинами: ${{damagedSections}}`;
  slider.value = frameIndex;
}}
function animate(time) {{
  if (playing) {{
    if (!lastTime) lastTime = time;
    accumulator += (time - lastTime) * Number(speed.value);
    if (accumulator >= 100) {{
      current = (current + Math.floor(accumulator / 100)) % data.frames.length;
      accumulator %= 100;
      draw(current);
    }}
  }}
  lastTime = time;
  requestAnimationFrame(animate);
}}
slider.addEventListener("input", () => {{ current = Number(slider.value); draw(current); }});
button.addEventListener("click", () => {{
  playing = !playing;
  button.textContent = playing ? "Пауза" : "Воспроизвести";
  lastTime = 0;
}});
draw(0);
requestAnimationFrame(animate);
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=1.0, help="Beam length in metres")
    parser.add_argument("--height", type=float, default=0.1, help="Beam height in metres")
    parser.add_argument("--width", type=float, default=0.1, help="Beam width in metres")
    parser.add_argument("--youngs-modulus", type=float, default=9.0e9, help="Young's modulus in Pa")
    parser.add_argument("--poisson-ratio", type=float, default=0.30, help="Poisson ratio")
    parser.add_argument("--density", type=float, default=917.0, help="Ice density in kg/m^3")
    parser.add_argument("--tensile-strength", type=float, default=0.7e6, help="Tensile strength in Pa")
    parser.add_argument("--compressive-strength", type=float, default=3.0e6, help="Compressive strength in Pa")
    parser.add_argument("--max-displacement", type=float, default=0.005, help="Punch travel in metres")
    parser.add_argument("--steps", type=int, default=100, help="Number of quasi-static load steps")
    parser.add_argument("--blocks", type=int, default=41, help="Odd number of rigid beam blocks")
    parser.add_argument("--fibers", type=int, default=21, help="Odd number of fiber bonds per interface")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    args = parser.parse_args()

    parameters = Parameters(
        length=args.length,
        height=args.height,
        width=args.width,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        density=args.density,
        tensile_strength=args.tensile_strength,
        compressive_strength=args.compressive_strength,
        max_displacement=args.max_displacement,
        steps=args.steps,
        blocks=args.blocks,
        fibers=args.fibers,
    )
    print(json.dumps(run_simulation(parameters, args.output), indent=2))


if __name__ == "__main__":
    main()

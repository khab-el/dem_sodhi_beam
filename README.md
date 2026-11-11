# Sodhi ice beam: discrete-element model

The supplied sketch is modeled as a clamped beam under a centered, downward
displacement-controlled punch. Each cross-section is a rigid discrete block.
Neighboring blocks are joined by longitudinal fiber springs distributed over
the beam depth and by a shear spring. A tensile fiber spring breaks at
`sigma_t`; after breaking, it can transmit compression only. The surviving
compressed fibers therefore provide a load path across the cracked section.
An intact fiber spring is removed when its compressive stress reaches
`sigma_c`; a crushed fiber carries a capped compressive force thereafter.

The formulation is a quasi-static 2D DEM reduced to rigid cross-sections, not a
full 3D particle model. It uses small-displacement, linear springs and no
friction or inertia. Crushing is idealized as a perfectly plastic compression
plateau; post-crush compaction is not modeled. Density is recorded but does not
affect a quasi-static solve. A separate three-hinge plastic limit criterion
estimates global collapse. At its first crossing, the demo imposes an
instantaneous complete break at the two interfaces adjoining the punch block,
then continues for fifty display steps to show the zero-reaction final
state. During those display steps the beam halves are held at their last
pre-collapse shape while the detached punch block moves down, avoiding an
unsupported post-fracture static equilibrium. This is a visualization
idealization, not a dynamic post-collapse simulation.

## Run

The script uses only the Python standard library:

```bash
python3 dem_sodhi.py --output results
```

Default demonstration values are `L=1 m`, `h=w=0.1 m`, `E=9 GPa`,
`nu=0.30`, `sigma_t=0.7 MPa`, `sigma_c=3 MPa`, and 5 mm punch travel in 100
quasi-static steps. The load schedule is refined near the estimated first
flexural crack and expected compression failure. Use `--length`, `--height`,
`--width`, `--youngs-modulus`,
`--poisson-ratio`, `--tensile-strength`, `--compressive-strength`,
`--max-displacement`, and `--steps` to override inputs. `--blocks` sets the
number of rigid cross-sections and `--fibers` sets the number of longitudinal
fiber bonds at each interface; both must be odd. Check mesh sensitivity by
increasing both.

The run writes:

- `force_displacement.csv` — punch reaction, displacement, broken fiber bonds,
  active compression contacts, the plastic-collapse threshold, and the
  post-failure continuation index;
- `force_displacement.svg` — load-displacement curve;
- `crack_state.svg` — final bond state: tension, compression, crack-face
  compression contact, open tensile cracks, and crushed fibers;
- `beam_animation.html` — interactive animation with play/pause, speed, and a
  step slider; the beam body follows the calculated block rotations and
  deflections, its colored layers show fiber tension/compression states, and
  sparse arrows show the calculated longitudinal normal stress `sigma_x`
  (right for tension, left for compression; length scales with magnitude).
  Vertical markers show broken, contacting, or crushed bonds. The displayed
  deformation is magnified 4x; force and displacement readouts are actual.
  An adjacent force-displacement graph tracks the current animation step;
- `summary.json` — parameters, first crack, the derived plastic moment and
  collapse load, the first collapse-threshold crossing, the idealized
  post-failure section cut, and analytical Sodhi reference loads.

The reference loads are `Ff = 4 sigma_t w h^2/(3 L)` for first flexural
cracking and `Fc = 8 sigma_c w h^2/(9 L)` for Sodhi's post-failure formula at
`beta = 1/3`. The separate plastic limit used to stop the simulation is
`Fu = 4 w h^2 sigma_t sigma_c/[L(sigma_t + sigma_c)]`, derived from a
rectangular stress block and a three-hinge mechanism at the two clamps and
midspan. The discrete reaction may overshoot `Fu` at the first crossing because
the loading is displacement-controlled and damage is brittle. After crossing,
the punch-adjacent block is intentionally disconnected from both neighboring
beam portions so its computed reaction becomes zero. This artificial
post-failure cut is used to illustrate loss of load path; it does not predict
the physical crack trajectory, fragment motion, or energy release. These are
comparison/limit values, not calibration targets. The simulation is an
educational reduced-order DEM; validate mesh convergence and material/contact
parameters before using predictions quantitatively.

For the detailed derivation of the discrete spring laws, damage/contact
transitions, assembled equilibrium equations, and load-step algorithm, see
[MATHEMATICAL_MODEL.md](./MATHEMATICAL_MODEL.md).

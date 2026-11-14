# Robotech Simulator — Agent & Developer Guidelines (`AGENTS.md`)

## 1. Project Overview
- **Robotech: SDF-1 Veritech 3D Combat & Transformation Simulator** is a standalone 100% pure Mojo 1.2 game built on top of **Flint Engine** (`../flint`).
- Keep engine-generic rendering, physics, audio, ECS, and platform capabilities inside `../flint/src/flint/`, and keep Robotech-specific mecha, AI, HUD, and mission logic inside `src/robotech/`.

## 2. Sibling Directory Layout on Any Machine
`tools/run.sh` automatically detects `../flint` relative to this repository (or honors `FLINT_ROOT=/path/to/flint`):

```text
<workspace-parent>/
  flint/      # Pure-Mojo 1.2 Game Engine (flint.*)
  robotech/   # Robotech: SDF-1 Veritech 3D Combat Simulator
```

## 3. Commands
```bash
pixi install            # Install Mojo 1.2 toolchain (or share ../flint/.pixi)
bash tools/run.sh gate  # Build + run 4 combat verification suites + render 4 PNGs in build/shots/
bash tools/run.sh play  # Launch interactive Wayland simulator
```

## 4. Next Milestone on Desktop GPU Workstation
See `docs/VISUAL_REALISM_PLAN.md` for the step-by-step plan to upgrade `RobotechRenderer` (`src/robotech/hud_cockpit.mojo` & `src/robotech/mecha_meshes.mojo`) from CPU `SimdCanvas` flat boxes to **Vulkan / Radeon GFX12 GPU PBR rendering**, high-poly `.glb` VF-1 / Zentraedi mecha models, AgX filmic tonemapping, dynamic missile point lights, HDR bloom, and volumetric Itano Circus smoke ribbons.

# Robotech: SDF-1 Veritech 3D Combat & Macross Saga Campaign Simulator

A standalone **100% pure Mojo 1.2** 3D mecha transformation, dogfight & capital-ship combat simulator inspired by **Robotech (The Macross Saga)**, built on top of [**Flint Engine**](file:///home/user/flint) (`flint.*`) with zero external C/C++ wrapper libraries.

---

## Features

- **Macross Saga 3-Mission Campaign (`[F1]` to cycle or automatic wave progression)**:
  1. **Mission 1 — Macross Island Atmospheric Intercept (Waves 1–2)**: Dogfight through a volumetric sunlit cumulus cloud deck (`draw_cloud_deck`) over the azure South Pacific Ocean and sculpted 3D South Ataria (Macross) Island caldera.
  2. **Mission 2 — High Earth Orbit SDF-1 Defense (Waves 3–4)**: Space-fold into high orbit above a curved sunlit Earth + cyan Rayleigh atmospheric limb alongside the colossal **SDF-1 Super Dimension Fortress**, unlocking Super/Strike FAST Packs against **Miriya's crimson Queadluun-Rau** power-armor aces.
  3. **Mission 3 — Zentraedi Command Fleet Assault (Wave 5+)**: Engage the colossal **Thuverl-Salan Heavy Destroyer** capital warship boss escorted by Queadluun-Rau & Glaug aces, and call in **SDF-1's Twin Main Boom Reflex Cannon (`[F]`)**!
- **Real-Time 3-Mode VF-1S Valkyrie Transformation (`[1]` / `[2]` / `[3]` / `[TAB]`)**:
  - **Fighter Mode (`[1]`)**: High-speed variable-sweep interceptor (`104 m/s` clean / `126 m/s` Super FAST Pack afterburner), bank-and-pitch aerodynamics, and evasive aileron barrel rolls (`Q`/`E`) that break Zentraedi missile locks.
  - **Guardian / GERWALK Mode (`[2]`)**: Iconic reverse-knee VTOL hover-strafing with twin vectoring foot plumes and deployed GU-11 Gunpod arm.
  - **Battloid Mode (`[3]`)**: Full towering humanoid mecha with VF-1S quad head lasers, +40% GU-11 Gunpod fire rate, 360° omni-thrusters, and a glowing hexagonal **Pinpoint Barrier Shield (`SHIFT`)**.
- **NP-BP-01 Super / Strike Valkyrie FAST Packs (`[G]`)**:
  - Attaches twin dorsal rocket booster pods + starboard dual-barrel **RO-X2A Strike Beam Cannon** (`vf1s_fast_pack.glb`), expanding micro-missile capacity from `36 -> 60`, boosting afterburner thrust, and firing twin high-velocity cyan Strike Cannon bolts alongside the GU-11 Gunpod.
- **SDF-1 Main Boom Reflex Cannon Tactical Strike (`[F]`)**:
  - Charges passively and on Zentraedi kills; at `100%`, unleashes twin colossal golden-cyan mega-particle beams (`draw_colossal_beam_3d`) directly from SDF-1's forward booms with synthesized `SFX_REFLEX_CANNON` audio.
- **Hardware Vulkan / AMDGPU RDNA GFX12 PBR Pipeline (`GpuSceneRenderer`)**:
  - 7 native binary glTF 2.0 `.glb` models (`vf1s_valkyrie.glb`, `vf1s_fast_pack.glb`, `regult_pod.glb`, `glaug_officer.glb`, `queadluun_rau.glb`, `thuverl_salan.glb`, `sdf1_macross.glb`), Cook-Torrance GGX specular, Fresnel sheen, soft-penumbra sphere shadows, 48 dynamic combat lights, AgX filmic tonemapping, 5-tap bilinear HDR bloom, and 1st-person 3D armored cockpit canopy + SDF TrueType HUD (`[V]`).
- **Instant LZ4 Checkpoints (`[F5]` / `[F9]`) & LiveLink MCP Telemetry (`--mcp`)**:
  - Bit-exact versioned `BlobWriter` + `LZ4` state save/load (`build/quicksave.flz4`) and non-blocking `AF_UNIX` LiveLink MCP telemetry (`build/flint_robotech_mcp.sock`).

---

## Quickstart

```bash
cd /home/user/robotech

# Launch live Wayland 3D Combat Simulator
bash tools/run.sh play

# Query headless Flint MCP live state JSON
./build/robotech-sim --mcp flint_state

# Run full verification gate (build + 5 combat/campaign test suites + 6 1280x720 showcase PNGs)
bash tools/run.sh gate
```

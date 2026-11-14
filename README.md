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
- **SDF-1 Macross Cruiser <-> Towering Storm Attacker Transformation & Daedalus Attack (`[H]`)**:
  - Smoothly transforms the 7-node articulated `sdf1_macross.glb` between **Battlecruiser** and towering humanoid **Storm Attacker** (`sdf1_storm_morph`), pivoting the twin 60m Reflex Cannons over the shoulders and extending the **CVS-101 Prometheus** & **SLV-111 Daedalus** carrier arms!
  - Pressing `[H]` while in Storm Attacker unleashes the **SLV-111 Daedalus Pinpoint-Barrier Attack** — ramming the enemy warship with a glowing hexagonal Pinpoint Barrier bow and launching 16 internal ramp micro-missiles!
- **Interactive CVS-101 Prometheus Hangar Briefing & 3 Playable Squadron Pilots (`[B]`, `[P]`)**:
  - Inspect rotating 3D Valkyries on the lit Prometheus hangar turntable with crisp **GPU Directional Planar Deck Shadows**, mission intel, and performance stat bars before scrambling (`SPACE` / `ENTER`):
    - **Skull-1 — Roy Focker / Rick Hunter (`VF-1S` White/Gold/Black)**: `+30%` SDF-1 Reflex Cannon charge rate.
    - **Blue-2 — Max Sterling (`VF-1J` Cobalt Ace)**: `+18%` turn/roll agility & `+12%` top Mach speed.
    - **Red-1 — Miriya Parina (`VF-1J` Crimson/Gold Ace)**: `+22%` GU-11 & RO-X2A Strike Cannon ROF & `+35%` multi-lock radar sweep speed.
- **Destructible Zentraedi Warship Subsystems & Hull Trench Skimming**:
  - Target 4 independent subsystems on the **Thuverl-Salan Heavy Destroyer**: **Port/Starboard Plasma Turrets** (silences heavy plasma fire), **Torpedo Bay** (halts 4-missile salvos), and **Command Bridge** (exposes **2.0× critical hull damage**!). Skimming within `66m` of the Destroyer's armor trench grants `1.35×` weapon damage + rapid Reflex Cannon charge.
- **Procedural 48 kHz Stereo Macross BGM (`[N]`) & VHF Radio Comms Synth**:
  - Concert-pitch stereo heroic anime brass/arp/bass BGM loop (`synth_macross_bgm_clip`) on `BUS_MUSIC` + VHF squelch/formant radio comms (`synth_radio_comms_clip`) on `BUS_SPEECH`.
- **Hardware Vulkan / AMDGPU RDNA GFX12 PBR Pipeline (`GpuSceneRenderer`)**:
  - 7 native binary glTF 2.0 `.glb` models, Cook-Torrance GGX specular, Fresnel sheen, soft-penumbra sphere shadows, GPU planar hangar/ocean/hull shadows (`draw_planar_shadow_model`), 48 dynamic combat lights, AgX filmic tonemapping, 5-tap bilinear HDR bloom, and 1st-person 3D armored cockpit canopy + SDF TrueType HUD (`[V]`).
- **Deterministic `.flrp` Flight Recorder (`[F6]` / `[F7]`), `.flpk` VFS Packager (`--pack`), LZ4 Checkpoints (`[F5]` / `[F9]`) & LiveLink MCP**:
  - Record and bit-verify full dogfight replays (`build/flight_record.flrp`, FNV-64 state-verified), bundle all `.glb`/`.png`/`.ttf` assets into a single `build/robotech_assets.flpk` archive (`bash tools/run.sh pack`), quick-save/load (`build/quicksave.flz4`), and query live `AF_UNIX` MCP telemetry.

---

## Quickstart

```bash
cd /home/user/robotech

# Launch live Wayland 3D Combat Simulator (opens in Prometheus Hangar Briefing)
bash tools/run.sh play

# Bundle all 3D GLB, sky PNG, and TTF HUD assets into a single .flpk VFS archive
bash tools/run.sh pack

# Query headless Flint MCP live state JSON
./build/robotech-sim --mcp flint_state

# Run full verification gate (build + 6 test suites + .flpk pack + 8 1280x720 showcase PNGs)
bash tools/run.sh gate
```

# Robotech: SDF-1 Veritech 3D Combat Simulator

A standalone **100% pure Mojo 1.2** 3D mecha transformation & dogfight simulator inspired by **Robotech (The Macross Saga)**, built on top of [**Flint Engine**](file:///home/user/flint) (`flint.*`) with zero external C/C++ wrapper libraries.

---

## Features

- **Real-Time 3-Mode VF-1S Valkyrie Transformation (`[1]` / `[2]` / `[3]` / `[TAB]`)**:
  - **Fighter Mode (`[1]`)**: High-speed swept-wing interceptor (`104 m/s` afterburner), bank-and-pitch aerodynamics, and evasive aileron barrel rolls (`Q`/`E`) that break Zentraedi missile locks.
  - **Guardian / GERWALK Mode (`[2]`)**: Iconic reverse-knee VTOL hover-strafing with twin vectoring foot plumes and deployed GU-11 Gunpod arm.
  - **Battloid Mode (`[3]`)**: Full towering humanoid mecha with VF-1S quad head lasers, +40% GU-11 Gunpod fire rate, 360° omni-thrusters, and a glowing **Pinpoint Barrier Shield (`SHIFT`)**.
- **Ichiro Itano ("Macross") Corkscrew Micro-Missile Swarms**:
  - Hold **`Right Mouse` / `M`** to paint multiple Zentraedi Battlepods with diamond radar locks (`LOCK X1..X4`) and release — or press **`R` / `K`** for an instant 24-missile helical salvo!
- **Zentraedi Assault Waves & Skull Squadron Wingman**:
  - Fight agile **Regult Tactical Battlepods** and armored **Glaug Officer Command Pods** alongside AI wingman **Max Sterling (`SKULL-2`)** in front of the **SDF-1 Super Dimension Fortress**.
- **3 Camera Perspectives (`[V]`)**:
  - Dynamic Spring Chase Cam, 1st-Person **VF-1 Holographic Canopy Cockpit HUD**, and Action Wingman Cam.
- **Zero-Asset Procedural 48 kHz Audio (`robotech.audio_synth`)**:
  - Pure-Mojo synthesized VF-1 hydraulic transformation servos, GU-11 55mm rotary gunpod bursts, SDF-1 lock-on chirps, missile salvos, plasma booms, and continuous thermonuclear turbine drone (`NativeAudio.set_drone`).

---

## Quickstart

```bash
cd /home/user/robotech

# Launch live Wayland 3D Combat Simulator
bash tools/run.sh play

# Run full verification gate (build + 4 combat test suites + 4 headless showcase PNGs)
bash tools/run.sh gate
```

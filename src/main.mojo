"""Robotech: SDF-1 Veritech 3D Combat & Transformation Simulator (`main.mojo`).

Powered by **Flint Engine** (`flint.*`) in 100% pure Mojo 1.2.

Controls:
  - `[1]` / `[2]` / `[3]` or `[TAB]`: Transform VF-1S Valkyrie (Fighter / Guardian / Battloid)
  - `WASD` / Arrow Keys / Mouse: Pitch, Yaw, VTOL Strafe & Throttle
  - `SPACE` / `C`: Vertical Thruster Climb / Dive (Guardian & Battloid) or Gunpod Fire (Fighter)
  - `Left Mouse` or `J`: Fire GU-11 55mm Rotary Gunpod
  - `Hold Right Mouse` or `Hold M`: Sweep Multi-Lock Radar -> Release to Launch Itano Circus!
  - `R` or `K`: Quick-Fire Full 24-Missile Itano Circus Swarm!
  - `SHIFT`: Thermonuclear Afterburner (Fighter/Guardian) or Pinpoint Barrier Shield (Battloid)
  - `Q` / `E`: Evasive Aileron Barrel Roll (breaks incoming Zentraedi missile locks!)
  - `V`: Cycle Camera (Dynamic Chase Cam / VF-1 Cockpit Canopy HUD / Action Wingman Cam)
  - `~`: Flint DevConsole  |  `F2`: Frame Profiler  |  `F3`: MemCharge Inspector  |  `ESC`: Quit
"""
from std.sys import argv
from std.math import atan2
from flint.core.math import Vec3
from flint.assets.image import write_png
from flint.platform.os import get_time
from flint.platform.input import InputState, PAD_A, PAD_B, PAD_X, PAD_Y, PAD_RB, PAD_LB, PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import SimdCanvas
from flint.devtools.overlays import FlintDevOverlays
from robotech.audio_synth import (
    RobotechAudioBank,
    SFX_TRANSFORM,
    SFX_GUNPOD,
    SFX_LOCK_BEEP,
    SFX_MISSILE_SALVO,
    SFX_EXPLOSION,
)
from robotech.sim_world import (
    RobotechSimWorld,
    PilotControls,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    EVT_TRANSFORM,
    EVT_GUNPOD,
    EVT_LOCK,
    EVT_SALVO,
    EVT_EXPLODE,
)
from robotech.hud_cockpit import RobotechRenderer, CAM_CHASE, CAM_COCKPIT, CAM_ACTION


def gather_controls(input: InputState, mode: Int) -> PilotControls:
    var c = PilotControls()

    # Mode selection: [1] Fighter (49), [2] Guardian (50), [3] Battloid (51), [TAB] cycle (258)
    if input.key_pressed(49):
        c.request_mode = MODE_FIGHTER
    elif input.key_pressed(50):
        c.request_mode = MODE_GUARDIAN
    elif input.key_pressed(51):
        c.request_mode = MODE_BATTLOID
    elif input.key_pressed(258) or input.pads[0].pressed[PAD_Y]:
        c.request_mode = (mode + 1) % 3

    var pad_lx = input.pads[0].axes[PAD_AXIS_LEFT_X]
    var pad_ly = input.pads[0].axes[PAD_AXIS_LEFT_Y]

    if mode == MODE_FIGHTER:
        if input.key_down(65) or input.key_down(263):  # A / Left
            c.yaw_in -= 1.0
        if input.key_down(68) or input.key_down(262):  # D / Right
            c.yaw_in += 1.0
        if input.key_down(87) or input.key_down(265):  # W / Up -> Climb
            c.pitch_in += 1.0
            c.throttle_in += 0.6
        if input.key_down(83) or input.key_down(264):  # S / Down -> Dive
            c.pitch_in -= 1.0
            c.throttle_in -= 0.4
    else:
        # Guardian & Battloid: WASD strafe/thrust, Arrows/Mouse aim!
        if input.key_down(87):  # W
            c.throttle_in += 1.0
        if input.key_down(83):  # S
            c.throttle_in -= 1.0
        if input.key_down(65):  # A
            c.strafe_x -= 1.0
        if input.key_down(68):  # D
            c.strafe_x += 1.0
        if input.key_down(263):  # Left Arrow
            c.yaw_in -= 1.0
        if input.key_down(262):  # Right Arrow
            c.yaw_in += 1.0
        if input.key_down(265):  # Up Arrow
            c.pitch_in += 1.0
        if input.key_down(264):  # Down Arrow
            c.pitch_in -= 1.0
        if input.key_down(32):  # SPACE -> Jump-jet climb
            c.lift_y += 1.0
        if input.key_down(67):  # C -> Descend
            c.lift_y -= 1.0

    # Smooth mouse flight / reticle steering + gamepad stick
    c.yaw_in = max(Float32(-1.5), min(Float32(1.5), c.yaw_in + input.delta_x * Float32(0.012) + pad_lx))
    c.pitch_in = max(Float32(-1.5), min(Float32(1.5), c.pitch_in - input.delta_y * Float32(0.012) - pad_ly))

    c.boost_held = input.key_down(340) or input.key_down(344) or input.pads[0].down[PAD_LB]
    c.fire_gunpod = (
        input.mouse_down(0)
        or input.key_down(74)  # J
        or (mode == MODE_FIGHTER and input.key_down(32))
        or input.pads[0].down[PAD_RB]
    )
    c.lock_held = input.mouse_down(2) or input.mouse_down(1) or input.key_down(77) or input.pads[0].down[PAD_A]  # RMB / M
    c.launch_salvo = input.mouse_released_state[2] or input.mouse_released_state[1] or input.key_released(77)
    c.quick_salvo = input.key_pressed(82) or input.key_pressed(75) or input.pads[0].pressed[PAD_B]  # R / K

    if input.key_pressed(81):  # Q
        c.barrel_roll_dir = -1
    elif input.key_pressed(69) or input.pads[0].pressed[PAD_X]:  # E
        c.barrel_roll_dir = 1

    return c


def render_showcase_shots():
    """Generates 4 deterministic 1280x720 combat showcase PNGs into `build/shots/`."""
    var w = 1280
    var h = 720
    var canvas = SimdCanvas(w, h)
    var renderer = RobotechRenderer()
    var overlays = FlintDevOverlays()
    overlays.show_profiler = True

    # Shot 1: Fighter Mode + 24-Missile Corkscrewing Itano Circus Swarm in Mid-Air!
    var sim = RobotechSimWorld(4)
    var c1 = PilotControls()
    c1.lock_held = True
    for _ in range(12):
        sim.step(c1, Float32(0.04))
    var c_salvo = PilotControls()
    c_salvo.quick_salvo = True
    c_salvo.fire_gunpod = True
    c_salvo.boost_held = True
    sim.step(c_salvo, Float32(0.03))
    var c_fly = PilotControls()
    c_fly.fire_gunpod = True
    c_fly.boost_held = True
    c_fly.yaw_in = Float32(0.08)
    for _ in range(7):
        sim.step(c_fly, Float32(0.03))

    var t0 = get_time()
    renderer.cam_mode = CAM_CHASE
    renderer.render_frame(canvas, sim, True)
    overlays.tick_us = 88
    overlays.render_us = Int((get_time() - t0) * 1_000_000.0)
    overlays.render(canvas)
    _ = write_png("build/shots/fighter_itano_circus.png", canvas.pixels, w, h)

    # Shot 2: Guardian (GERWALK) Reverse-Knee VTOL Hover Strafe!
    var c_guard = PilotControls()
    c_guard.request_mode = MODE_GUARDIAN
    sim.step(c_guard, Float32(0.04))
    var c_strafe = PilotControls()
    c_strafe.strafe_x = Float32(-0.65)
    c_strafe.fire_gunpod = True
    for _ in range(18):
        sim.step(c_strafe, Float32(0.04))
    for i in range(len(sim.enemies)):
        if sim.enemies[i].active:
            var dir_e = (sim.enemies[i].pos - sim.pos).normalized()
            sim.yaw = atan2(dir_e.x, dir_e.z)
            sim.pitch = dir_e.y * Float32(0.5)
            break
    sim.step(c_strafe, Float32(0.02))
    renderer.cam_mode = CAM_ACTION
    renderer.render_frame(canvas, sim, False)
    _ = write_png("build/shots/guardian_hover_strafe.png", canvas.pixels, w, h)

    # Shot 3: Towering VF-1S Battloid Mode + Pinpoint Barrier Dueling Zentraedi Glaug Officer!
    var c_batt = PilotControls()
    c_batt.request_mode = MODE_BATTLOID
    sim.step(c_batt, Float32(0.04))
    var c_duel = PilotControls()
    c_duel.fire_gunpod = True
    c_duel.boost_held = True  # Activates cyan Pinpoint Barrier Shield on left arm!
    for _ in range(16):
        sim.step(c_duel, Float32(0.04))
    for i in range(len(sim.enemies)):
        if sim.enemies[i].active:
            var dir_e = (sim.enemies[i].pos - sim.pos).normalized()
            sim.yaw = atan2(dir_e.x, dir_e.z)
            sim.pitch = dir_e.y * Float32(0.5)
            break
    c_duel.quick_salvo = True
    sim.step(c_duel, Float32(0.03))
    c_duel.quick_salvo = False
    for _ in range(5):
        sim.step(c_duel, Float32(0.03))
    renderer.cam_mode = CAM_ACTION
    renderer.render_frame(canvas, sim, True)
    _ = write_png("build/shots/battloid_duel_glaug.png", canvas.pixels, w, h)

    # Shot 4: First-Person VF-1 Holographic Cockpit Canopy View with Multi-Lock Reticles!
    for i in range(len(sim.enemies)):
        if sim.enemies[i].active:
            var dir_e = (sim.enemies[i].pos - sim.pos).normalized()
            sim.yaw = atan2(dir_e.x, dir_e.z)
            sim.pitch = dir_e.y * Float32(0.6)
            break
    var c_lock = PilotControls()
    c_lock.lock_held = True
    c_lock.fire_gunpod = True
    for _ in range(12):
        sim.step(c_lock, Float32(0.04))
    renderer.cam_mode = CAM_COCKPIT
    renderer.render_frame(canvas, sim, False)
    _ = write_png("build/shots/cockpit_tactical_hud.png", canvas.pixels, w, h)

    sim.shutdown()
    print("[Robotech] Wrote 4 showcase combat screenshots -> build/shots/")


def main():
    for i in range(1, len(argv())):
        if String(argv()[i]) == "--shots":
            render_showcase_shots()
            return

    var w = 1280
    var h = 720
    var sim = RobotechSimWorld()
    var renderer = RobotechRenderer()
    var canvas = SimdCanvas(w, h)
    var overlays = FlintDevOverlays()
    var audio = RobotechAudioBank(True)

    var win = WaylandWindow()
    var input = InputState()
    if not win.initialize(w, h):
        audio.close()
        sim.shutdown()
        print("[Robotech] Headless session detected — run with '--shots' to render combat PNGs.")
        return

    var last_t = get_time()
    while not win.close_requested:
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last_t)))
        last_t = now

        input.begin_frame()
        win.poll(input)
        if input.key_pressed(256):
            break

        var ctrl = PilotControls()
        if not overlays.handle_input(input):
            if input.key_pressed(86):  # V -> cycle camera
                renderer.cam_mode = (renderer.cam_mode + 1) % 3
            ctrl = gather_controls(input, sim.mode)

        var t0 = get_time()
        sim.step(ctrl, dt)
        var t1 = get_time()

        # Trigger synthesized Robotech SFX from simulation events
        if (sim.sound_events & EVT_TRANSFORM) != 0:
            audio.trigger(SFX_TRANSFORM, Float32(1.0), Float32(0.85), 4)
        if (sim.sound_events & EVT_SALVO) != 0:
            audio.trigger(SFX_MISSILE_SALVO, Float32(1.0), Float32(0.85), 3)
        if (sim.sound_events & EVT_EXPLODE) != 0:
            audio.trigger(SFX_EXPLOSION, Float32(0.95), Float32(0.80), 3)
        if (sim.sound_events & EVT_LOCK) != 0:
            audio.trigger(SFX_LOCK_BEEP, Float32(0.95) + Float32(sim.total_locks) * Float32(0.03), Float32(0.55), 2)
        if (sim.sound_events & EVT_GUNPOD) != 0:
            audio.trigger(SFX_GUNPOD, Float32(0.98), Float32(0.45), 1)

        renderer.render_frame(canvas, sim, ctrl.boost_held)
        overlays.tick_us = Int((t1 - t0) * 1_000_000.0)
        overlays.render_us = Int((get_time() - t1) * 1_000_000.0)
        overlays.fps = Int(Float32(1.0) / dt)
        overlays.render(canvas)

        win.present(canvas.pixels, w, h)
        audio.update(sim.vel.length() / Float32(105.0))

    audio.close()
    win.close()
    sim.shutdown()

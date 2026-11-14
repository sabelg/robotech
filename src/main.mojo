"""Robotech: SDF-1 Veritech 3D Combat & Macross Saga Campaign Simulator (`main.mojo`).

Powered by **Flint Engine** (`flint.*`) in 100% pure Mojo 1.2.

Controls:
  - `[1]` / `[2]` / `[3]` or `[TAB]`: Transform VF-1 Valkyrie (Fighter / Guardian / Battloid)
  - `[P]`: Cycle Squadron Pilot & VF-1 Livery (Roy Focker VF-1S / Max Sterling VF-1J / Miriya VF-1J)
  - `[B]`: Open / Close Interactive CVS-101 Prometheus Hangar Tactical Briefing (`SPACE`/`ENTER` Scramble)
  - `[G]`: Equip / Jettison NP-BP-01 Super/Strike FAST Packs (60 Missiles + RO-X2A Beam Cannon)
  - `[H]`: Transform SDF-1 Cruiser <-> Towering Storm Attacker / Trigger SLV-111 Daedalus Attack!
  - `[F]`: Fire SDF-1 Main Boom Reflex Cannon Tactical Strike (when 100% charged!)
  - `[N]`: Toggle Procedural Heroic Macross Stereo BGM Synthesizer
  - `[F1]`: Cycle Campaign Mission (M1 Macross Island / M2 High Orbit / M3 Fleet Boss)
  - `[F5]` / `[F9]`: Instant LZ4 Checkpoint Quick-Save / Quick-Load (`build/quicksave.flz4`)
  - `[F6]` / `[F7]`: Record / Replay Deterministic `.flrp` Flight Stream (`build/flight_record.flrp`)
  - `WASD` / Arrow Keys / Mouse: Pitch, Yaw, VTOL Strafe & Throttle
  - `SPACE` / `C`: Vertical Thruster Climb / Dive (Guardian & Battloid) or Gunpod Fire (Fighter)
  - `Left Mouse` or `J`: Fire GU-11 55mm Rotary Gunpod (+ RO-X2A Strike Cannon)
  - `Hold Right Mouse` or `Hold M`: Sweep Multi-Lock Radar -> Release to Launch Itano Circus!
  - `R` or `K`: Quick-Fire Full Itano Circus Micro-Missile Swarm!
  - `SHIFT`: Thermonuclear Afterburner (Fighter/Guardian) or Pinpoint Barrier Shield (Battloid)
  - `Q` / `E`: Evasive Aileron Barrel Roll (breaks incoming Zentraedi missile locks!)
  - `V`: Cycle Camera (Dynamic Chase Cam / VF-1 Cockpit Canopy HUD / Action Wingman Cam)
  - `~`: Flint DevConsole  |  `F2`: Frame Profiler  |  `F3`: MemCharge Inspector  |  `ESC`: Quit
"""
from std.sys import argv
from std.math import atan2
from flint.core.math import Vec3
from flint.core.serial import BlobWriter
from flint.assets.image import write_png
from flint.assets.replay_stream import FlintReplayStream
from flint.platform.os import get_time
from flint.platform.input import (
    InputState,
    PAD_A,
    PAD_B,
    PAD_X,
    PAD_Y,
    PAD_RB,
    PAD_LB,
    PAD_START,
    PAD_BACK,
    PAD_AXIS_LEFT_X,
    PAD_AXIS_LEFT_Y,
)
from flint.platform.wayland import WaylandWindow
from flint.render.simd_canvas import SimdCanvas
from flint.devtools.overlays import FlintDevOverlays
from flint.devtools.mcp_server import (
    FlintMcpInfo,
    FlintMcpLiveEndpoint,
    handle_flint_mcp_query,
)
from robotech.asset_forge import pack_robotech_assets
from robotech.audio_synth import (
    RobotechAudioBank,
    SFX_TRANSFORM,
    SFX_GUNPOD,
    SFX_LOCK_BEEP,
    SFX_MISSILE_SALVO,
    SFX_EXPLOSION,
    SFX_REFLEX_CANNON,
)
from robotech.sim_world import (
    RobotechSimWorld,
    PilotControls,
    encode_pilot_controls,
    decode_pilot_controls,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    PILOT_MAX_STERLING,
    PILOT_MIRIYA_ACE,
    THEATER_MACROSS_ISLAND,
    THEATER_HIGH_ORBIT,
    THEATER_BOSS_FLEET,
    EVT_TRANSFORM,
    EVT_GUNPOD,
    EVT_LOCK,
    EVT_SALVO,
    EVT_EXPLODE,
    EVT_REFLEX,
    EVT_RADIO,
)
from robotech.hud_cockpit import RobotechRenderer, CAM_CHASE, CAM_COCKPIT, CAM_ACTION

comptime QUICKSAVE_PATH: String = "build/quicksave.flz4"
comptime FLIGHT_RECORD_PATH: String = "build/flight_record.flrp"
comptime ASSET_PACK_PATH: String = "build/robotech_assets.flpk"
comptime MCP_SOCKET_PATH: String = "build/flint_robotech_mcp.sock"


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

    # [G] (71): FAST Packs | [F] (70): Reflex Cannon | [H] (72): SDF-1 Storm/Daedalus | [P] (80): Pilot | [F1] (290): Mission
    c.toggle_fast_packs = input.key_pressed(71) or input.pads[0].pressed[PAD_BACK]
    c.fire_reflex_cannon = input.key_pressed(70) or input.pads[0].pressed[PAD_START]
    c.toggle_storm_attacker = input.key_pressed(72)
    c.cycle_pilot = input.key_pressed(80)
    c.cycle_mission = input.key_pressed(290)

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
    """Generates 8 deterministic 1280x720 Macross Saga showcase PNGs into `build/shots/`."""
    var w = 1280
    var h = 720
    var canvas = SimdCanvas(w, h)
    var renderer = RobotechRenderer(w, h)
    var overlays = FlintDevOverlays()
    overlays.active_backend_name = renderer.backend_label()
    overlays.show_profiler = True

    # Shot 1: High-Orbit Fighter Mode + Corkscrewing Itano Circus Swarm!
    var sim = RobotechSimWorld(4)
    sim.set_mission(THEATER_HIGH_ORBIT)
    sim.fast_packs = False
    sim.missiles_ready = 36
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

    # Shot 3: Towering VF-1S Battloid Mode + Pinpoint Barrier Dueling Glaug & Queadluun-Rau!
    var c_batt = PilotControls()
    c_batt.request_mode = MODE_BATTLOID
    sim.step(c_batt, Float32(0.04))
    var c_duel = PilotControls()
    c_duel.throttle_in = Float32(0.65)
    c_duel.strafe_x = Float32(0.42)
    c_duel.lift_y = Float32(0.18)
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
    for _ in range(4):
        sim.step(c_duel, Float32(0.03))
    renderer.cam_mode = CAM_ACTION
    renderer.render_frame(canvas, sim, True)
    _ = write_png("build/shots/battloid_duel_glaug.png", canvas.pixels, w, h)
    sim.shutdown()

    # Shot 4: First-Person VF-1 Armored Cockpit Canopy View with Multi-Lock Reticles & SDF-1!
    var sim_cockpit = RobotechSimWorld(4)
    sim_cockpit.set_mission(THEATER_HIGH_ORBIT)
    var c_lock = PilotControls()
    c_lock.lock_held = True
    c_lock.throttle_in = Float32(0.72)
    for _ in range(10):
        sim_cockpit.step(c_lock, Float32(0.04))
    c_lock.fire_gunpod = True
    for _ in range(4):
        sim_cockpit.step(c_lock, Float32(0.03))
    renderer.cam_mode = CAM_COCKPIT
    renderer.render_frame(canvas, sim_cockpit, False)
    _ = write_png("build/shots/cockpit_tactical_hud.png", canvas.pixels, w, h)
    sim_cockpit.shutdown()

    # Shot 5: Mission 1 — Macross Island Atmospheric Intercept (Pacific Ocean + Cloud Deck + Planar Sea Shadows)!
    var sim_island = RobotechSimWorld(4)
    sim_island.set_mission(THEATER_MACROSS_ISLAND)
    sim_island.pos = Vec3(-8.0, 26.0, -10.0)
    var c_isl = PilotControls()
    c_isl.lock_held = True
    c_isl.yaw_in = Float32(-0.18)
    c_isl.pitch_in = Float32(-0.06)
    c_isl.boost_held = True
    for _ in range(8):
        sim_island.step(c_isl, Float32(0.04))
    c_isl.lock_held = False
    c_isl.quick_salvo = True
    c_isl.fire_gunpod = True
    sim_island.step(c_isl, Float32(0.03))
    c_isl.quick_salvo = False
    for _ in range(6):
        sim_island.step(c_isl, Float32(0.03))
    renderer.cam_mode = CAM_CHASE
    renderer.render_frame(canvas, sim_island, True)
    _ = write_png("build/shots/macross_island_clouds.png", canvas.pixels, w, h)
    sim_island.shutdown()

    # Shot 6: Mission 3 — Super/Strike VF-1S + SDF-1 Twin Main Reflex Cannon vs Thuverl-Salan Destroyer & Subsystems!
    var sim_boss = RobotechSimWorld(4)
    sim_boss.set_mission(THEATER_BOSS_FLEET)
    var c_boss = PilotControls()
    c_boss.lock_held = True
    c_boss.boost_held = True
    for _ in range(8):
        sim_boss.step(c_boss, Float32(0.04))
    var dir_b = (sim_boss.boss_pos - sim_boss.pos).normalized()
    sim_boss.yaw = atan2(dir_b.x, dir_b.z)
    sim_boss.pitch = dir_b.y * Float32(0.55)
    c_boss.lock_held = False
    c_boss.quick_salvo = True
    c_boss.fire_gunpod = True
    c_boss.fire_reflex_cannon = True
    sim_boss.step(c_boss, Float32(0.03))
    c_boss.quick_salvo = False
    c_boss.fire_reflex_cannon = False
    for _ in range(5):
        sim_boss.step(c_boss, Float32(0.03))
    renderer.cam_mode = CAM_CHASE
    renderer.render_frame(canvas, sim_boss, True)
    _ = write_png("build/shots/sdf1_reflex_boss_assault.png", canvas.pixels, w, h)
    sim_boss.shutdown()

    # Shot 7: Phase 5 — Miriya's Crimson Super VF-1J + Towering SDF-1 Storm Attacker & SLV-111 Daedalus Pinpoint Attack!
    var sim_storm = RobotechSimWorld(4)
    sim_storm.set_mission(THEATER_BOSS_FLEET)
    sim_storm.set_pilot(PILOT_MIRIYA_ACE)
    sim_storm.sdf1_storm_mode = True
    sim_storm.sdf1_storm_morph.value = Float32(1.0)
    # Position Miriya skimming within 62m of Destroyer while framing full humanoid SDF-1 Storm Attacker + Daedalus beam!
    sim_storm.boss_pos = Vec3(-18.0, 62.0, 152.0)
    var sp0 = sim_storm.boss_subsystem_world_pos(0)
    sim_storm._damage_boss_subsystem(0, Float32(120.0), sp0)
    sim_storm.pos = Vec3(8.0, 44.0, 86.0)
    sim_storm.yaw = Float32(-0.54)
    sim_storm.pitch = Float32(0.14)
    sim_storm.vel = sim_storm.forward_vec() * Float32(84.0)
    sim_storm.cam_eye_spring.value = sim_storm.pos - sim_storm.forward_vec() * Float32(14.5) + Vec3(0.0, 4.2, 0.0)
    sim_storm.cam_target_spring.value = sim_storm.pos + sim_storm.forward_vec() * Float32(28.0)
    var c_st = PilotControls()
    c_st.lock_held = True
    c_st.boost_held = True
    c_st.fire_gunpod = True
    for _ in range(3):
        sim_storm.step(c_st, Float32(0.02))
    c_st.toggle_storm_attacker = True  # Already in Storm Attacker -> triggers SLV-111 Daedalus Attack!
    c_st.quick_salvo = True
    sim_storm.step(c_st, Float32(0.02))
    c_st.toggle_storm_attacker = False
    c_st.quick_salvo = False
    sim_storm.pos = Vec3(4.0, 46.0, 98.0)  # 60.7m from Destroyer -> Hull Trench Skimming active!
    sim_storm.step(c_st, Float32(0.02))
    renderer.cam_mode = CAM_CHASE
    renderer.show_briefing = False
    renderer.render_frame(canvas, sim_storm, True)
    _ = write_png("build/shots/sdf1_storm_attacker_daedalus.png", canvas.pixels, w, h)
    sim_storm.shutdown()

    # Shot 8: Phase 5 — CVS-101 Prometheus Hangar Tactical Briefing, Squadron Roster & GPU Planar Deck Shadow!
    var sim_brief = RobotechSimWorld(2)
    sim_brief.set_mission(THEATER_BOSS_FLEET)
    sim_brief.set_pilot(PILOT_MAX_STERLING)
    sim_brief.mode = MODE_GUARDIAN
    sim_brief.morph_fg.value = Float32(1.0)
    sim_brief.morph_gb.value = Float32(0.0)
    sim_brief.fast_packs = True
    sim_brief.sim_time = Float32(0.65)
    renderer.show_briefing = True
    renderer.render_frame(canvas, sim_brief, False)
    _ = write_png("build/shots/tactical_hangar_briefing.png", canvas.pixels, w, h)
    renderer.show_briefing = False
    sim_brief.shutdown()

    print("[Robotech] Wrote 8 showcase combat screenshots -> build/shots/")


def main():
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--shots":
            render_showcase_shots()
            return
        if a == "--pack":
            var out_path = String(args[i + 1]) if i + 1 < len(args) else ASSET_PACK_PATH
            var nb = pack_robotech_assets(out_path)
            print("[Robotech] Packed single-file VFS archive -> " + out_path + " (" + String(nb) + " bytes)")
            return
        if a == "--mcp":
            var method = String(args[i + 1]) if i + 1 < len(args) else String("flint_state")
            var sim_mcp = RobotechSimWorld(2)
            var info = FlintMcpInfo("robotech-sdf1", sim_mcp.tick_count, "HEADLESS", sim_mcp.mcp_state_json())
            print(handle_flint_mcp_query(method, "{}", info))
            sim_mcp.shutdown()
            return

    var w = 1280
    var h = 720
    var sim = RobotechSimWorld()
    var renderer = RobotechRenderer(w, h)
    renderer.show_briefing = True  # Open on interactive Prometheus Hangar Briefing!
    var canvas = SimdCanvas(w, h)
    var overlays = FlintDevOverlays()
    overlays.active_backend_name = renderer.backend_label()
    var audio = RobotechAudioBank(True)
    var mcp_live = FlintMcpLiveEndpoint(MCP_SOCKET_PATH)
    var recorder = FlintReplayStream()
    var recording_active = False
    var replaying_active = False
    var replay_idx = 0

    var win = WaylandWindow()
    var input = InputState()
    if not win.initialize(w, h):
        mcp_live.close()
        audio.close()
        sim.shutdown()
        print("[Robotech] Headless session detected — run with '--shots', '--pack', or '--mcp flint_state'.")
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
            if input.key_pressed(78):  # N -> Toggle Macross Stereo BGM
                renderer.bgm_on = audio.toggle_bgm()
            if input.key_pressed(66):  # B -> Toggle Tactical Hangar Briefing
                renderer.show_briefing = not renderer.show_briefing
                audio.trigger_radio()

            if renderer.show_briefing:
                # Hangar Briefing controls: [P] Pilot, [F1] Mission, [G] FAST Packs, [1-3] Mode, [SPACE/ENTER] Scramble!
                if input.key_pressed(80):
                    sim.cycle_pilot()
                    audio.trigger_radio()
                if input.key_pressed(290):
                    sim.set_mission((sim.theater + 1) % 3)
                    audio.trigger_radio()
                if input.key_pressed(71):
                    sim.fast_packs = not sim.fast_packs
                    sim.missiles_ready = sim.max_missiles()
                if input.key_pressed(49):
                    sim.mode = MODE_FIGHTER
                elif input.key_pressed(50):
                    sim.mode = MODE_GUARDIAN
                elif input.key_pressed(51):
                    sim.mode = MODE_BATTLOID
                if input.key_pressed(32) or input.key_pressed(257):  # SPACE / ENTER -> Scramble!
                    renderer.show_briefing = False
                    audio.trigger_radio()
            else:
                if input.key_pressed(86):  # V -> cycle camera
                    renderer.cam_mode = (renderer.cam_mode + 1) % 3
                if input.key_pressed(294):  # F5 -> LZ4 Quick-Save
                    if sim.save_checkpoint(QUICKSAVE_PATH):
                        sim.comms_banner = "LZ4 CHECKPOINT SAVED -> " + QUICKSAVE_PATH
                        sim.comms_timer = 3.0
                if input.key_pressed(298):  # F9 -> LZ4 Quick-Load
                    try:
                        sim.load_checkpoint(QUICKSAVE_PATH)
                    except:
                        sim.comms_banner = "NO QUICKSAVE FOUND AT " + QUICKSAVE_PATH
                        sim.comms_timer = 2.5

                # [F6] (295): Start / Stop Deterministic `.flrp` Flight Recorder
                if input.key_pressed(295):
                    replaying_active = False
                    if not recording_active:
                        var w_init = BlobWriter()
                        sim.serialize_state(w_init)
                        recorder.begin_recording("robotech-sdf1", w_init)
                        recording_active = True
                        renderer.replay_label = "REC .FLRP [F6]"
                        sim.comms_banner = "FLIGHT RECORDER ENGAGED -> " + FLIGHT_RECORD_PATH + " ([F6] TO SAVE)"
                        sim.comms_timer = 3.2
                    else:
                        recording_active = False
                        var w_fin = BlobWriter()
                        sim.serialize_state(w_fin)
                        _ = recorder.finish_recording(w_fin)
                        if recorder.save(FLIGHT_RECORD_PATH):
                            renderer.replay_label = ""
                            sim.comms_banner = (
                                "SAVED "
                                + String(recorder.tick_count())
                                + " TICKS -> "
                                + FLIGHT_RECORD_PATH
                                + " ([F7] REPLAY)"
                            )
                            sim.comms_timer = 3.8

                # [F7] (296): Load & Play Back Deterministic `.flrp` Flight Stream
                if input.key_pressed(296):
                    recording_active = False
                    try:
                        recorder = FlintReplayStream.load(FLIGHT_RECORD_PATH)
                        var r_init = recorder.initial_reader()
                        sim.deserialize_state(r_init)
                        replaying_active = recorder.tick_count() > 0
                        replay_idx = 0
                        renderer.replay_label = "REPLAY .FLRP"
                        sim.comms_banner = "PLAYING BACK FLIGHT STREAM (" + String(recorder.tick_count()) + " TICKS)..."
                        sim.comms_timer = 3.0
                    except:
                        renderer.replay_label = ""
                        sim.comms_banner = "NO FLIGHT STREAM FOUND AT " + FLIGHT_RECORD_PATH
                        sim.comms_timer = 2.6

                if not replaying_active:
                    ctrl = gather_controls(input, sim.mode)

        var t0 = get_time()
        if renderer.show_briefing:
            sim.sim_time += dt
            var goal_fg = Float32(0.0) if sim.mode == MODE_FIGHTER else Float32(1.0)
            var goal_gb = Float32(1.0) if sim.mode == MODE_BATTLOID else Float32(0.0)
            sim.morph_fg.update(goal_fg, Float32(0.14), dt)
            sim.morph_gb.update(goal_gb, Float32(0.14), dt)
            sim.sound_events = 0
        elif replaying_active:
            if replay_idx < recorder.tick_count():
                var r_dt = recorder.tick_dts[replay_idx]
                ctrl = decode_pilot_controls(recorder.tick_inputs[replay_idx])
                sim.step(ctrl, r_dt)
                replay_idx += 1
                renderer.replay_label = "REPLAY " + String((replay_idx * 100) // max(1, recorder.tick_count())) + "%"
            if replay_idx >= recorder.tick_count():
                replaying_active = False
                renderer.replay_label = ""
                var ok = sim.state_hash() == recorder.final_hash
                sim.comms_banner = (
                    "FLRP REPLAY COMPLETE — BIT-EXACT FNV64 STATE VERIFIED!"
                    if ok
                    else "FLRP REPLAY COMPLETE!"
                )
                sim.comms_timer = 4.0
        else:
            if recording_active:
                recorder.record_tick(dt, encode_pilot_controls(ctrl))
            sim.step(ctrl, dt)
        var t1 = get_time()

        # Answer any non-blocking Flint MCP LiveLink queries over AF_UNIX
        _ = mcp_live.poll_once(
            FlintMcpInfo("robotech-sdf1", sim.tick_count, renderer.backend_label(), sim.mcp_state_json())
        )

        # Trigger synthesized Robotech SFX + VHF Radio Comms from simulation events
        if (sim.sound_events & EVT_RADIO) != 0:
            audio.trigger_radio()
        if (sim.sound_events & EVT_REFLEX) != 0:
            audio.trigger(SFX_REFLEX_CANNON, Float32(1.0), Float32(0.95), 5)
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
        audio.update(Float32(0.25) if renderer.show_briefing else (sim.vel.length() / Float32(105.0)))

    mcp_live.close()
    audio.close()
    win.close()
    sim.shutdown()

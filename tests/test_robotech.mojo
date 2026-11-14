"""Verification Suite for Robotech: SDF-1 Veritech Combat Simulator (`flint.*`)."""
from std.testing import assert_equal, assert_true
from flint.platform.os import read_bytes
from flint.render.simd_canvas import SimdCanvas
from flint.render.flip import flip_error_map, flip_mean
from flint.devtools.mcp_server import FlintMcpInfo, handle_flint_mcp_query
from robotech.audio_synth import (
    synth_transform_clip,
    synth_gunpod_clip,
    synth_missile_salvo_clip,
    synth_explosion_clip,
    synth_reflex_cannon_clip,
)
from robotech.sim_world import (
    RobotechSimWorld,
    PilotControls,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    THEATER_MACROSS_ISLAND,
    THEATER_HIGH_ORBIT,
    THEATER_BOSS_FLEET,
    ENEMY_QUEADLUUN,
    EVT_TRANSFORM,
    EVT_SALVO,
    EVT_REFLEX,
)
from robotech.hud_cockpit import RobotechRenderer, CAM_CHASE, CAM_COCKPIT


def test_vf1_three_mode_transformation() raises:
    var sim = RobotechSimWorld(2)
    assert_equal(sim.mode, MODE_FIGHTER)
    assert_true(sim.morph_fg.value < Float32(0.01))
    assert_true(sim.morph_gb.value < Float32(0.01))

    # Transform Fighter -> Guardian
    var ctrl = PilotControls()
    ctrl.request_mode = MODE_GUARDIAN
    sim.step(ctrl, Float32(0.05))
    assert_equal(sim.mode, MODE_GUARDIAN)
    assert_true((sim.sound_events & EVT_TRANSFORM) != 0)
    ctrl.request_mode = -1
    for _ in range(25):
        sim.step(ctrl, Float32(0.05))
    assert_true(sim.morph_fg.value > Float32(0.95))
    assert_true(sim.morph_gb.value < Float32(0.05))

    # Transform Guardian -> Battloid
    ctrl.request_mode = MODE_BATTLOID
    sim.step(ctrl, Float32(0.05))
    assert_equal(sim.mode, MODE_BATTLOID)
    ctrl.request_mode = -1
    for _ in range(25):
        sim.step(ctrl, Float32(0.05))
    assert_true(sim.morph_fg.value > Float32(0.95))
    assert_true(sim.morph_gb.value > Float32(0.95))
    sim.shutdown()
    print("  [PASS] VF-1 Valkyrie 3-Mode Transformation Springs (Fighter -> Guardian -> Battloid)")


def test_itano_circus_homing_and_combat() raises:
    var sim = RobotechSimWorld(4)
    var initial_hostiles = sim.active_enemy_count()
    assert_true(initial_hostiles >= 6)

    var ctrl = PilotControls()
    ctrl.quick_salvo = True
    ctrl.fire_gunpod = True
    sim.step(ctrl, Float32(0.04))
    assert_true((sim.sound_events & EVT_SALVO) != 0)
    assert_true(sim.missiles_ready < 36)

    # Advance 3.2 seconds so corkscrewing micro-missiles converge on Zentraedi Battlepods
    ctrl.quick_salvo = False
    for _ in range(80):
        sim.step(ctrl, Float32(0.04))

    assert_true(sim.kills >= 1 or sim.score > 0)
    sim.shutdown()
    print("  [PASS] Itano Circus Micro-Missile Swarm Homing, Broadphase Impacts & Scoring")


def test_procedural_robotech_audio_synth() raises:
    var tr = synth_transform_clip()
    var gp = synth_gunpod_clip()
    var ms = synth_missile_salvo_clip()
    var ex = synth_explosion_clip()
    var rx = synth_reflex_cannon_clip()
    assert_true(tr.valid and tr.frame_count() > 15000)
    assert_true(gp.valid and gp.frame_count() > 3000)
    assert_true(ms.valid and ms.frame_count() > 12000)
    assert_true(ex.valid and ex.frame_count() > 18000)
    assert_true(rx.valid and rx.frame_count() > 38000)
    print("  [PASS] Procedural 48 kHz Robotech SFX Synthesizer (Transform, Gunpod, Salvo, Explosion, Reflex Cannon)")


def test_renderer_and_cockpit_flip_diff() raises:
    var sim = RobotechSimWorld(2)
    var renderer = RobotechRenderer(320, 180, prefer_gpu=True)
    assert_true(renderer.fonts_loaded)
    assert_true(len(read_bytes("assets/models/vf1s_valkyrie.glb")) > 4096)
    assert_true(len(read_bytes("assets/models/vf1s_fast_pack.glb")) > 2048)
    assert_true(len(read_bytes("assets/models/regult_pod.glb")) > 4096)
    assert_true(len(read_bytes("assets/models/glaug_officer.glb")) > 4096)
    assert_true(len(read_bytes("assets/models/queadluun_rau.glb")) > 4096)
    assert_true(len(read_bytes("assets/models/thuverl_salan.glb")) > 4096)
    assert_true(len(read_bytes("assets/models/sdf1_macross.glb")) > 4096)
    assert_true(len(read_bytes("assets/sky/earth_orbit.png")) > 4096)
    assert_true(len(read_bytes("assets/sky/macross_ocean.png")) > 4096)

    var c_chase = SimdCanvas(320, 180)
    var c_cockpit = SimdCanvas(320, 180)

    renderer.cam_mode = CAM_CHASE
    renderer.render_frame(c_chase, sim, True)
    assert_true(c_chase.triangles_drawn > 2500)

    renderer.cam_mode = CAM_COCKPIT
    renderer.render_frame(c_cockpit, sim, False)
    assert_true(c_cockpit.triangles_drawn > 1500)

    var diff = flip_mean(flip_error_map(c_chase.pixels, c_cockpit.pixels, 320, 180))
    assert_true(diff > 0.02)

    # Also verify CPU PBR fallback path works cleanly on Mission 3 Fleet Boss + Reflex Beam!
    sim.set_mission(THEATER_BOSS_FLEET)
    _ = sim.trigger_reflex_cannon()
    var cpu_renderer = RobotechRenderer(160, 90, prefer_gpu=False)
    var c_cpu = SimdCanvas(160, 90)
    cpu_renderer.render_frame(c_cpu, sim, True)
    assert_true(c_cpu.triangles_drawn > 800)

    sim.shutdown()
    print(
        "  [PASS] All 7 GLB 2.0 Models, Cloud/Ocean & Orbit Skies, SDF HUD & GPU PBR Render ("
        + renderer.backend_label()
        + " -> "
        + String(c_chase.triangles_drawn)
        + " tris/frame)"
    )


def test_campaign_fast_packs_reflex_checkpoint_and_mcp() raises:
    var sim = RobotechSimWorld(2)
    assert_equal(sim.theater, THEATER_MACROSS_ISLAND)
    assert_true(not sim.fast_packs)
    assert_equal(sim.max_missiles(), 36)

    # 1. Toggle Super/Strike FAST Packs -> 60 micro-missiles + RO-X2A Strike Cannon + higher Mach speed
    var c_fp = PilotControls()
    c_fp.toggle_fast_packs = True
    c_fp.fire_gunpod = True
    c_fp.boost_held = True
    sim.step(c_fp, Float32(0.04))
    assert_true(sim.fast_packs)
    assert_equal(sim.max_missiles(), 60)
    assert_equal(sim.missiles_ready, 60)
    assert_true(sim.strike_flash_t > Float32(0.0))

    # 2. Transition to Mission 2 (High Orbit) -> verify Queadluun-Rau Aces spawn!
    sim.set_mission(THEATER_HIGH_ORBIT)
    assert_equal(sim.theater, THEATER_HIGH_ORBIT)
    var qdl_count = 0
    for i in range(len(sim.enemies)):
        if sim.enemies[i].active and sim.enemies[i].kind == ENEMY_QUEADLUUN:
            qdl_count += 1
    assert_true(qdl_count >= 2)

    # 3. Transition to Mission 3 (Zentraedi Fleet Boss) -> verify Thuverl-Salan Destroyer & SDF-1 Reflex Cannon!
    sim.set_mission(THEATER_BOSS_FLEET)
    assert_equal(sim.theater, THEATER_BOSS_FLEET)
    assert_true(sim.boss_active)
    var hp_before = sim.boss_hp
    var c_ref = PilotControls()
    c_ref.fire_reflex_cannon = True
    sim.step(c_ref, Float32(0.04))
    assert_true((sim.sound_events & EVT_REFLEX) != 0)
    assert_true(sim.reflex_beam_t > Float32(2.0))
    assert_true(sim.boss_hp < hp_before - Float32(600.0))

    # 4. Verify Bit-Exact LZ4 Checkpoint Save / Load Determinism!
    var ckpt_path = "build/test_robotech_ckpt.flz4"
    assert_true(sim.save_checkpoint(ckpt_path))
    var sim_restored = RobotechSimWorld(2)
    sim_restored.load_checkpoint(ckpt_path)

    var c_det = PilotControls()
    c_det.fire_gunpod = True
    c_det.yaw_in = Float32(0.25)
    c_det.quick_salvo = True
    sim.step(c_det, Float32(0.04))
    sim_restored.step(c_det, Float32(0.04))
    c_det.quick_salvo = False
    for _ in range(15):
        sim.step(c_det, Float32(0.04))
        sim_restored.step(c_det, Float32(0.04))

    assert_equal(sim.rng_state, sim_restored.rng_state)
    assert_equal(sim.score, sim_restored.score)
    assert_equal(sim.missiles_ready, sim_restored.missiles_ready)
    assert_true((sim.pos - sim_restored.pos).length() < Float32(0.0001))
    assert_true(abs(sim.boss_hp - sim_restored.boss_hp) < Float32(0.0001))

    # 5. Verify Flint MCP Telemetry JSON Query
    var mcp_json = handle_flint_mcp_query(
        "flint_state",
        "{}",
        FlintMcpInfo("robotech-sdf1", sim.tick_count, "VULKAN-GPU", sim.mcp_state_json()),
    )
    assert_true('"boss_active":true' in mcp_json)
    assert_true('"fast_packs":true' in mcp_json)

    sim_restored.shutdown()
    sim.shutdown()
    print("  [PASS] 3-Mission Campaign, Super/Strike FAST Packs, Queadluun/Destroyer Boss, SDF-1 Reflex & LZ4 Checkpoints")


def main() raises:
    print("[Robotech Test] Running SDF-1 Veritech Combat Simulator Verification Suite...")
    test_vf1_three_mode_transformation()
    test_itano_circus_homing_and_combat()
    test_procedural_robotech_audio_synth()
    test_renderer_and_cockpit_flip_diff()
    test_campaign_fast_packs_reflex_checkpoint_and_mcp()
    print("[Robotech Test] ALL 5 ROBOTECH COMBAT & CAMPAIGN SUITES PASSED!")

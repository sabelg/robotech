"""Verification Suite for Robotech: SDF-1 Veritech Combat Simulator (`flint.*`)."""
from std.testing import assert_equal, assert_true
from flint.core.math import Vec3
from flint.core.serial import BlobWriter
from flint.platform.os import read_bytes
from flint.assets.vfs import pack_directory, pack_read
from flint.assets.replay_stream import FlintReplayStream
from flint.render.simd_canvas import SimdCanvas
from flint.render.flip import flip_error_map, flip_mean
from flint.devtools.mcp_server import FlintMcpInfo, handle_flint_mcp_query
from robotech.asset_forge import pack_robotech_assets
from robotech.audio_synth import (
    synth_transform_clip,
    synth_gunpod_clip,
    synth_missile_salvo_clip,
    synth_explosion_clip,
    synth_reflex_cannon_clip,
    synth_radio_comms_clip,
    synth_macross_bgm_clip,
)
from robotech.sim_world import (
    RobotechSimWorld,
    PilotControls,
    encode_pilot_controls,
    decode_pilot_controls,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    PILOT_SKULL_LEADER,
    PILOT_MAX_STERLING,
    PILOT_MIRIYA_ACE,
    SUBSYS_PORT_TURRET,
    SUBSYS_BRIDGE,
    THEATER_MACROSS_ISLAND,
    THEATER_HIGH_ORBIT,
    THEATER_BOSS_FLEET,
    ENEMY_QUEADLUUN,
    EVT_TRANSFORM,
    EVT_SALVO,
    EVT_REFLEX,
    EVT_RADIO,
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


def test_phase5_storm_attacker_pilots_subsystems_flpk_and_replay() raises:
    # 1. Procedural Stereo Macross BGM + VHF Radio Voice/Squelch Synth
    var bgm = synth_macross_bgm_clip()
    var rad = synth_radio_comms_clip()
    assert_true(bgm.valid and bgm.channels == 2 and bgm.frame_count() > 280000)
    assert_true(rad.valid and rad.frame_count() > 12000)

    # 2. Squadron Pilots, SDF-1 Storm Attacker Morph, Daedalus Attack & Destroyer Subsystems!
    var sim = RobotechSimWorld(2)
    sim.set_mission(THEATER_BOSS_FLEET)
    assert_equal(sim.intact_subsystem_count(), 4)
    assert_true(not sim.boss_bridge_destroyed())

    # Cycle pilots: Skull Leader -> Max Sterling -> Miriya Parina
    sim.cycle_pilot()
    assert_equal(sim.pilot_id, PILOT_MAX_STERLING)
    assert_true((sim.sound_events & EVT_RADIO) != 0)
    sim.cycle_pilot()
    assert_equal(sim.pilot_id, PILOT_MIRIYA_ACE)

    # Transform SDF-1 into Storm Attacker ([H]) & step spring
    var c_st = PilotControls()
    c_st.toggle_storm_attacker = True
    sim.step(c_st, Float32(0.05))
    assert_true(sim.sdf1_storm_mode)
    c_st.toggle_storm_attacker = False
    for _ in range(30):
        sim.step(c_st, Float32(0.05))
    assert_true(sim.sdf1_storm_morph.value > Float32(0.85))

    # Press [H] again in Storm Attacker -> executes SLV-111 Daedalus Pinpoint-Barrier Attack!
    var boss_hp_pre_daed = sim.boss_hp
    c_st.toggle_storm_attacker = True
    sim.step(c_st, Float32(0.05))
    assert_true(sim.daedalus_attack_t > Float32(2.0))
    assert_true(sim.boss_hp < boss_hp_pre_daed - Float32(450.0))

    # Destroy Command Bridge subsystem -> verify 2.0x Critical Hull Damage!
    var bpos = sim.boss_pos
    var hp0 = sim.boss_hp
    sim._damage_boss(Float32(50.0), bpos)
    var normal_drop = hp0 - sim.boss_hp
    assert_true(abs(normal_drop - Float32(50.0)) < Float32(0.01))

    var sp_port = sim.boss_subsystem_world_pos(SUBSYS_PORT_TURRET)
    var sp_br = sim.boss_subsystem_world_pos(SUBSYS_BRIDGE)
    sim._damage_boss_subsystem(SUBSYS_PORT_TURRET, Float32(999.0), sp_port)
    sim._damage_boss_subsystem(SUBSYS_BRIDGE, Float32(999.0), sp_br)
    assert_true(sim.boss_bridge_destroyed())
    assert_equal(sim.intact_subsystem_count(), 2)

    var hp1 = sim.boss_hp
    sim._damage_boss(Float32(50.0), bpos)
    var crit_drop = hp1 - sim.boss_hp
    assert_true(abs(crit_drop - Float32(100.0)) < Float32(0.01))

    # Hull Trench Skimming within 66m of Destroyer
    sim.pos = sim.boss_pos + Vec3(0.0, 14.0, -32.0)
    var c_skim = PilotControls()
    sim.step(c_skim, Float32(0.03))
    assert_true(sim.skimming_hull)

    # 3. 7-Node Articulated SDF-1 Model + Hangar Tactical Briefing Render + Planar Deck Shadow!
    var renderer = RobotechRenderer(240, 135, prefer_gpu=True)
    assert_true(len(renderer.assets.sdf1.nodes) >= 7)
    assert_true(len(renderer.assets.vf1j_miriya.nodes) >= 14)
    var c_brief = SimdCanvas(240, 135)
    renderer.show_briefing = True
    renderer.render_frame(c_brief, sim, False)
    assert_true(c_brief.triangles_drawn > 1800)

    # 4. Single-File `.flpk` VFS Archive Bundler Round-Trip
    var flpk_path = "build/test_robotech_pack.flpk"
    var pack_bytes = pack_robotech_assets(flpk_path)
    assert_true(pack_bytes > 32768)
    var entries = pack_directory(flpk_path)
    assert_true(len(entries) >= 9)
    var sdf_packed = pack_read(flpk_path, "assets/models/sdf1_macross.glb")
    assert_true(len(sdf_packed) > 4096)

    # 5. Deterministic `.flrp` Flight Stream Recording & Bit-Exact FNV-64 Playback!
    var sim_rec = RobotechSimWorld(2)
    sim_rec.set_pilot(PILOT_SKULL_LEADER)
    var rec = FlintReplayStream()
    var w_init = BlobWriter()
    sim_rec.serialize_state(w_init)
    rec.begin_recording("robotech-sdf1", w_init)

    for tick in range(18):
        var ci = PilotControls()
        ci.yaw_in = Float32(0.22)
        ci.fire_gunpod = (tick % 2) == 0
        ci.quick_salvo = tick == 4
        ci.toggle_storm_attacker = tick == 2
        var dt_step = Float32(0.033)
        rec.record_tick(dt_step, encode_pilot_controls(ci))
        sim_rec.step(ci, dt_step)

    var w_fin = BlobWriter()
    sim_rec.serialize_state(w_fin)
    var expected_hash = rec.finish_recording(w_fin)
    var flrp_path = "build/test_flight_record.flrp"
    assert_true(rec.save(flrp_path))

    var loaded_rec = FlintReplayStream.load(flrp_path)
    assert_equal(loaded_rec.tick_count(), 18)
    var sim_play = RobotechSimWorld(2)
    var r_init = loaded_rec.initial_reader()
    sim_play.deserialize_state(r_init)
    for tick in range(loaded_rec.tick_count()):
        var cp = decode_pilot_controls(loaded_rec.tick_inputs[tick])
        sim_play.step(cp, loaded_rec.tick_dts[tick])

    assert_equal(sim_play.state_hash(), expected_hash)
    assert_equal(sim_play.score, sim_rec.score)
    assert_true((sim_play.pos - sim_rec.pos).length() < Float32(0.0001))

    sim_play.shutdown()
    sim_rec.shutdown()
    sim.shutdown()
    print("  [PASS] Phase 5: 7-Node SDF-1 Storm Attacker + Daedalus, 3 Pilots, 4 Boss Subsystems (2x Crit), Hangar Briefing, .flpk & Bit-Exact .flrp Replay!")


def main() raises:
    print("[Robotech Test] Running SDF-1 Veritech Combat Simulator Verification Suite...")
    test_vf1_three_mode_transformation()
    test_itano_circus_homing_and_combat()
    test_procedural_robotech_audio_synth()
    test_renderer_and_cockpit_flip_diff()
    test_campaign_fast_packs_reflex_checkpoint_and_mcp()
    test_phase5_storm_attacker_pilots_subsystems_flpk_and_replay()
    print("[Robotech Test] ALL 6 ROBOTECH COMBAT, HANGAR & REPLAY SUITES PASSED!")

"""Verification Suite for Robotech: SDF-1 Veritech Combat Simulator (`flint.*`)."""
from std.testing import assert_equal, assert_true
from flint.render.simd_canvas import SimdCanvas
from flint.render.flip import flip_error_map, flip_mean
from robotech.audio_synth import (
    synth_transform_clip,
    synth_gunpod_clip,
    synth_missile_salvo_clip,
    synth_explosion_clip,
)
from robotech.sim_world import (
    RobotechSimWorld,
    PilotControls,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    EVT_TRANSFORM,
    EVT_SALVO,
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
    assert_true(tr.valid and tr.frame_count() > 15000)
    assert_true(gp.valid and gp.frame_count() > 3000)
    assert_true(ms.valid and ms.frame_count() > 12000)
    assert_true(ex.valid and ex.frame_count() > 18000)
    print("  [PASS] Procedural 48 kHz Robotech SFX Synthesizer (Transform, Gunpod, Salvo, Explosion)")


def test_renderer_and_cockpit_flip_diff() raises:
    var sim = RobotechSimWorld(2)
    var renderer = RobotechRenderer()
    var c_chase = SimdCanvas(320, 180)
    var c_cockpit = SimdCanvas(320, 180)

    renderer.cam_mode = CAM_CHASE
    renderer.render_frame(c_chase, sim, True)
    assert_true(c_chase.triangles_drawn > 50)

    renderer.cam_mode = CAM_COCKPIT
    renderer.render_frame(c_cockpit, sim, False)

    var diff = flip_mean(flip_error_map(c_chase.pixels, c_cockpit.pixels, 320, 180))
    assert_true(diff > 0.02)
    sim.shutdown()
    print("  [PASS] 3D Mecha + SDF-1 Cruiser Render & Chase-vs-Cockpit LDR-FLIP Verification")


def main() raises:
    print("[Robotech Test] Running SDF-1 Veritech Combat Simulator Verification Suite...")
    test_vf1_three_mode_transformation()
    test_itano_circus_homing_and_combat()
    test_procedural_robotech_audio_synth()
    test_renderer_and_cockpit_flip_diff()
    print("[Robotech Test] ALL 4 ROBOTECH COMBAT SUITES PASSED!")

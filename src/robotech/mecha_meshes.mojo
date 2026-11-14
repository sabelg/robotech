"""Articulated GLB 2.0 Mecha Rigging & Macross Saga Theaters (`flint.render.scene_renderer`).

Drives continuous 3-mode transformation + Super/Strike FAST Packs on `vf1s_valkyrie.glb`,
reverse-knee stride kinematics on `regult_pod.glb`, `glaug_officer.glb` & `queadluun_rau.glb`,
capital-ship rendering for `thuverl_salan.glb` & `sdf1_macross.glb`, twin colossal SDF-1
Main Boom Reflex Cannon beams, and both Atmospheric Macross Island (ocean + cumulus cloud deck)
and High Earth Orbit combat theaters.
"""
from std.math import sin, cos
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.render.scene_renderer import (
    GpuSceneRenderer,
    FX_SOFT,
    FX_BOLT,
    FX_STRIPES,
)
from robotech.asset_forge import (
    RobotechAssetPack,
    VF1_NODE_TORSO,
    VF1_NODE_NOSE,
    VF1_NODE_HEAD,
    VF1_NODE_WING_L,
    VF1_NODE_WING_R,
    VF1_NODE_TAIL_L,
    VF1_NODE_TAIL_R,
    VF1_NODE_THIGH_L,
    VF1_NODE_SHIN_L,
    VF1_NODE_THIGH_R,
    VF1_NODE_SHIN_R,
    VF1_NODE_ARM_L,
    VF1_NODE_ARM_R,
    VF1_NODE_GUNPOD,
)
from robotech.sim_world import THEATER_MACROSS_ISLAND, THEATER_BOSS_FLEET


@fieldwise_init
struct MechaPose(Copyable, ImplicitlyCopyable, Movable):
    var pos: Vec3
    var fwd: Vec3
    var right: Vec3
    var up: Vec3
    var scale: Float32

    @staticmethod
    def from_angles(pos: Vec3, yaw: Float32, pitch: Float32, roll: Float32, scale: Float32 = 1.0) -> Self:
        var cy = cos(yaw)
        var sy = sin(yaw)
        var cp = cos(pitch)
        var sp = sin(pitch)
        var cr = cos(roll)
        var sr = sin(roll)

        var fwd = Vec3(sy * cp, sp, cy * cp).normalized()
        var flat_right = Vec3(-cy, Float32(0.0), sy).normalized()
        var base_up = flat_right.cross(fwd).normalized()
        var right = (flat_right * cr + base_up * sr).normalized()
        var up = right.cross(fwd).normalized()
        return Self(pos, fwd, right, up, scale)

    @always_inline
    def world_pt(self, lx: Float32, ly: Float32, lz: Float32) -> Vec3:
        return (
            self.pos
            + self.right * (lx * self.scale)
            + self.up * (ly * self.scale)
            + self.fwd * (lz * self.scale)
        )

    @always_inline
    def world_dir(self, lx: Float32, ly: Float32, lz: Float32) -> Vec3:
        return (self.right * lx + self.up * ly + self.fwd * lz).normalized()

    def to_mat4(self) -> Mat4:
        var m = Mat4()
        m.set(0, 0, self.right.x * self.scale)
        m.set(1, 0, self.right.y * self.scale)
        m.set(2, 0, self.right.z * self.scale)
        m.set(0, 1, self.up.x * self.scale)
        m.set(1, 1, self.up.y * self.scale)
        m.set(2, 1, self.up.z * self.scale)
        m.set(0, 2, self.fwd.x * self.scale)
        m.set(1, 2, self.fwd.y * self.scale)
        m.set(2, 2, self.fwd.z * self.scale)
        m.set(0, 3, self.pos.x)
        m.set(1, 3, self.pos.y)
        m.set(2, 3, self.pos.z)
        return m^


@always_inline
def _lerp(a: Float32, b: Float32, t: Float32) -> Float32:
    return a + (b - a) * t


@always_inline
def _lerp3(a: Vec3, b: Vec3, t: Float32) -> Vec3:
    return a + (b - a) * t


@always_inline
def _quat_axis(axis: Vec3, radians: Float32) -> Quat:
    var half = radians * Float32(0.5)
    var s = sin(half)
    var n = axis.normalized()
    return Quat(n.x * s, n.y * s, n.z * s, cos(half)).normalized()


def pose_and_draw_vf1(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    skull_leader: Bool,
    pose: MechaPose,
    morph_fg: Float32,
    morph_gb: Float32,
    afterburner: Float32 = 0.65,
    gun_flash: Bool = False,
    shield_active: Bool = False,
    fast_packs: Bool = False,
    strike_flash: Bool = False,
):
    """Articulates the 14-node `vf1s_valkyrie.glb` hierarchy (+ optional NP-BP-01 Super/Strike FAST Packs) and draws PBR + VFX."""
    ref model = assets.vf1s_skull if skull_leader else assets.vf1j_max
    if len(model.nodes) >= 14:
        var fg = max(Float32(0.0), min(Float32(1.0), morph_fg))
        var gb = max(Float32(0.0), min(Float32(1.0), morph_gb))

        # 0. Torso Core: rises upright in Battloid
        model.nodes[VF1_NODE_TORSO].translation = Vec3(0.0, _lerp(0.0, 0.86, gb), 0.0)
        model.nodes[VF1_NODE_TORSO].rotation = _quat_axis(Vec3(1, 0, 0), _lerp(0.0, -0.10, gb))

        # 1. Nosecone & Canopy: forward in Fighter/Guardian, folds down into chestplate in Battloid
        model.nodes[VF1_NODE_NOSE].translation = Vec3(0.0, _lerp(0.02, 0.56, gb), _lerp(1.18, 0.52, gb))
        model.nodes[VF1_NODE_NOSE].rotation = _quat_axis(Vec3(1, 0, 0), _lerp(0.0, 1.62, gb))

        # 2. VF-1S Head + Quad Lasers: emerges smoothly as morph_gb -> 1
        var head_s = max(Float32(0.02), clamp01((gb - 0.18) / 0.82))
        model.nodes[VF1_NODE_HEAD].translation = Vec3(0.0, _lerp(0.10, 0.64, gb), _lerp(0.05, 0.12, gb))
        model.nodes[VF1_NODE_HEAD].scale = Vec3(head_s, head_s, head_s)

        # 3 & 4. Variable-Sweep NACA Wings: sweep back at high Mach, fold into backpack in Battloid
        var mach_sweep = clamp01((afterburner - 0.5) * 1.4) * (1.0 - fg) * Float32(0.32)
        var wing_yaw = _lerp(mach_sweep, Float32(0.95), gb)
        var wing_dihedral = _lerp(Float32(0.0), Float32(0.68), gb)
        model.nodes[VF1_NODE_WING_L].translation = Vec3(-0.64, _lerp(0.08, 0.24, gb), _lerp(-0.12, -0.58, gb))
        model.nodes[VF1_NODE_WING_R].translation = Vec3(0.64, _lerp(0.08, 0.24, gb), _lerp(-0.12, -0.58, gb))
        model.nodes[VF1_NODE_WING_L].rotation = _quat_axis(Vec3(0, 1, 0), wing_yaw).multiply(_quat_axis(Vec3(0, 0, 1), wing_dihedral))
        model.nodes[VF1_NODE_WING_R].rotation = _quat_axis(Vec3(0, 1, 0), -wing_yaw).multiply(_quat_axis(Vec3(0, 0, 1), -wing_dihedral))

        # 5 & 6. Twin Tailfins
        model.nodes[VF1_NODE_TAIL_L].translation = Vec3(-0.52, _lerp(0.22, 0.30, gb), _lerp(-0.95, -0.82, gb))
        model.nodes[VF1_NODE_TAIL_R].translation = Vec3(0.52, _lerp(0.22, 0.30, gb), _lerp(-0.95, -0.82, gb))
        model.nodes[VF1_NODE_TAIL_L].rotation = _quat_axis(Vec3(0, 0, 1), _lerp(0.18, 0.48, gb))
        model.nodes[VF1_NODE_TAIL_R].rotation = _quat_axis(Vec3(0, 0, 1), _lerp(-0.18, -0.48, gb))

        # 7..10. Engine Nacelles (Fighter: +90 deg X -> aft) -> Reverse-Knee GERWALK -> Heroic Battloid Legs
        var thigh_pitch = _lerp(Float32(1.5708), _lerp(Float32(-0.82), Float32(-0.18), gb), fg)
        var shin_pitch = _lerp(Float32(0.0), _lerp(Float32(1.18), Float32(0.24), gb), fg)
        var stance_splay = _lerp(Float32(0.0), Float32(0.11), fg)
        model.nodes[VF1_NODE_THIGH_L].translation = Vec3(_lerp(-0.54, -0.60, fg), _lerp(-0.08, -0.22, gb), _lerp(-0.22, 0.02, gb))
        model.nodes[VF1_NODE_THIGH_R].translation = Vec3(_lerp(0.54, 0.60, fg), _lerp(-0.08, -0.22, gb), _lerp(-0.22, 0.02, gb))
        model.nodes[VF1_NODE_THIGH_L].rotation = _quat_axis(Vec3(0, 0, 1), -stance_splay).multiply(_quat_axis(Vec3(1, 0, 0), thigh_pitch))
        model.nodes[VF1_NODE_THIGH_R].rotation = _quat_axis(Vec3(0, 0, 1), stance_splay).multiply(_quat_axis(Vec3(1, 0, 0), thigh_pitch))
        model.nodes[VF1_NODE_SHIN_L].rotation = _quat_axis(Vec3(1, 0, 0), shin_pitch)
        model.nodes[VF1_NODE_SHIN_R].rotation = _quat_axis(Vec3(1, 0, 0), shin_pitch)

        # 11 & 12. Articulated Arms + Shoulder Pauldrons
        var arm_s = max(Float32(0.02), clamp01((fg - 0.12) / 0.88))
        var arm_y = _lerp(-0.26, 0.34, gb)
        model.nodes[VF1_NODE_ARM_L].translation = Vec3(-1.06, arm_y, 0.10)
        model.nodes[VF1_NODE_ARM_R].translation = Vec3(1.06, arm_y, 0.10)
        model.nodes[VF1_NODE_ARM_L].scale = Vec3(arm_s, arm_s, arm_s)
        model.nodes[VF1_NODE_ARM_R].scale = Vec3(arm_s, arm_s, arm_s)
        model.nodes[VF1_NODE_ARM_L].rotation = _quat_axis(Vec3(1, 0, 0), _lerp(Float32(0.10), Float32(-0.22) if shield_active else Float32(0.05), gb))
        model.nodes[VF1_NODE_ARM_R].rotation = _quat_axis(Vec3(1, 0, 0), _lerp(Float32(0.05), Float32(-0.14), gb))

        # 13. GU-11 55mm Rotary Gunpod: belly-mounted in Fighter, held in right hand in Guardian/Battloid
        model.nodes[VF1_NODE_GUNPOD].translation = _lerp3(
            Vec3(0.0, -0.40, -0.25),
            Vec3(1.06, arm_y - 0.56, 0.88),
            fg,
        )
        model.nodes[VF1_NODE_GUNPOD].rotation = _quat_axis(Vec3(1, 0, 0), _lerp(0.0, -0.08 * gb, fg))
        model.transforms_revision += 1

    var world_mat = pose.to_mat4()
    scene.draw_model(model, world_mat)

    var globals = model.global_transforms()

    # Optional NP-BP-01 Super/Strike FAST Pack Dorsal Boosters + RO-X2A Dual Strike Beam Cannon
    if fast_packs and len(globals) > VF1_NODE_TORSO:
        var torso_world = world_mat.multiply(globals[VF1_NODE_TORSO])
        scene.draw_model(assets.fast_pack, torso_world)
        var fp_len = (Float32(2.6) + afterburner * Float32(2.5)) * pose.scale
        var fp_rad = (Float32(0.24) + afterburner * Float32(0.09)) * pose.scale
        var fp_dir = torso_world.transform_vector(Vec3(0.0, 0.0, -1.0)).normalized()
        for side in range(2):
            var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
            var bell_pos = torso_world.transform_point(Vec3(sgn * Float32(0.46), Float32(0.42), Float32(-1.95)))
            scene.draw_thruster_plume(
                bell_pos,
                fp_dir,
                fp_len,
                fp_rad,
                UInt32(0xFFFFFFF2),
                UInt32(0xFF28EAFF),
                alpha=240,
                emissive=2.25 + afterburner * 0.9,
            )
        if strike_flash:
            for b_idx in range(2):
                var bx = Float32(0.39) if b_idx == 0 else Float32(0.53)
                var s0 = torso_world.transform_point(Vec3(bx, Float32(0.79), Float32(1.76)))
                var s1 = torso_world.transform_point(Vec3(bx, Float32(0.79), Float32(8.20)))
                scene.draw_billboard(s0, 0.44 * pose.scale, FX_SOFT, 0xFFB8FFFF, 245, emissive=2.8)
                scene.draw_beam_3d(s0, s1, 0.14 * pose.scale, FX_BOLT, 0xFF38EEFF, 235, emissive=2.7)

    # Volumetric Thermonuclear Afterburner Plumes + Mach Shock Diamonds from actual nozzle sockets
    if len(globals) > VF1_NODE_SHIN_R:
        var plume_len = (_lerp(Float32(3.8), Float32(1.65), morph_fg) + afterburner * Float32(2.6)) * pose.scale
        var plume_rad = (_lerp(Float32(0.30), Float32(0.22), morph_fg) + afterburner * Float32(0.10)) * pose.scale
        for side in range(2):
            var shin_idx = VF1_NODE_SHIN_L if side == 0 else VF1_NODE_SHIN_R
            var shin_world = world_mat.multiply(globals[shin_idx])
            var noz_pos = shin_world.transform_point(Vec3(0.0, -1.62, 0.04))
            var noz_dir = shin_world.transform_vector(Vec3(0.0, -1.0, 0.02)).normalized()
            scene.draw_thruster_plume(
                noz_pos,
                noz_dir,
                plume_len,
                plume_rad,
                UInt32(0xFFFFF8DC),
                UInt32(0xFF38D4FF) if skull_leader else UInt32(0xFF4CA8FF),
                alpha=235,
                emissive=1.95 + afterburner * 0.85,
            )

    # GU-11 55mm Rotary Gunpod Muzzle Starburst + Volumetric Tracer Cone
    if gun_flash and len(globals) > VF1_NODE_GUNPOD:
        var gp_world = world_mat.multiply(globals[VF1_NODE_GUNPOD])
        var muz = gp_world.transform_point(Vec3(0.0, 0.02, 2.12))
        var muz_end = gp_world.transform_point(Vec3(0.0, 0.02, 6.40))
        scene.draw_billboard(muz, 0.58 * pose.scale, FX_SOFT, 0xFFFFF2A0, 240, emissive=2.6)
        scene.draw_beam_3d(muz, muz_end, 0.18 * pose.scale, FX_BOLT, 0xFFFFE068, 225, emissive=2.4)

    # Translucent Hexagonal Pinpoint Barrier Energy Disc on Left Forearm (Battloid + SHIFT)
    if shield_active and len(globals) > VF1_NODE_ARM_L:
        var arm_world = world_mat.multiply(globals[VF1_NODE_ARM_L])
        var shield_pos = arm_world.transform_point(Vec3(-0.32, -0.45, 0.85))
        var shield_mat = trs(shield_pos, _quat_axis(pose.up, Float32(1.5708)), Vec3(pose.scale, pose.scale, pose.scale))
        scene.draw_model(assets.barrier_hex, shield_mat, emissive_boost=0.65)
        scene.draw_billboard(shield_pos, 1.55 * pose.scale, FX_SOFT, 0xFF4CEEFF, 115, emissive=1.8)


def pose_and_draw_regult(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    pose: MechaPose,
    stride_phase: Float32,
    firing: Bool = False,
):
    if len(assets.regult.nodes) >= 5:
        var sw_l = sin(stride_phase) * Float32(0.32)
        var sw_r = sin(stride_phase + Float32(3.14159)) * Float32(0.32)
        assets.regult.nodes[1].rotation = _quat_axis(Vec3(1, 0, 0), sw_l)
        assets.regult.nodes[2].rotation = _quat_axis(Vec3(1, 0, 0), -sw_l * Float32(1.35))
        assets.regult.nodes[3].rotation = _quat_axis(Vec3(1, 0, 0), sw_r)
        assets.regult.nodes[4].rotation = _quat_axis(Vec3(1, 0, 0), -sw_r * Float32(1.35))
        assets.regult.transforms_revision += 1

    scene.draw_model(assets.regult, pose.to_mat4())

    if firing:
        for side in range(2):
            var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
            var m0 = pose.world_pt(sgn * 0.44, 1.68, 2.28)
            var m1 = pose.world_pt(sgn * 0.44, 1.68, 7.60)
            scene.draw_billboard(m0, 0.42 * pose.scale, FX_SOFT, 0xFFFF6888, 235, emissive=2.5)
            scene.draw_beam_3d(m0, m1, 0.16 * pose.scale, FX_BOLT, 0xFFFF2E58, 225, emissive=2.3)


def pose_and_draw_glaug(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    pose: MechaPose,
    stride_phase: Float32,
    firing: Bool = False,
):
    if len(assets.glaug.nodes) >= 5:
        var sw_l = sin(stride_phase) * Float32(0.28)
        var sw_r = sin(stride_phase + Float32(3.14159)) * Float32(0.28)
        assets.glaug.nodes[1].rotation = _quat_axis(Vec3(1, 0, 0), sw_l)
        assets.glaug.nodes[2].rotation = _quat_axis(Vec3(1, 0, 0), -sw_l * Float32(1.25))
        assets.glaug.nodes[3].rotation = _quat_axis(Vec3(1, 0, 0), sw_r)
        assets.glaug.nodes[4].rotation = _quat_axis(Vec3(1, 0, 0), -sw_r * Float32(1.25))
        assets.glaug.transforms_revision += 1

    scene.draw_model(assets.glaug, pose.to_mat4())

    if firing:
        var c0 = pose.world_pt(0.0, 1.66, 3.48)
        var c1 = pose.world_pt(0.0, 1.66, 10.8)
        scene.draw_billboard(c0, 0.72 * pose.scale, FX_SOFT, 0xFFFFB048, 245, emissive=2.8)
        scene.draw_beam_3d(c0, c1, 0.28 * pose.scale, FX_BOLT, 0xFFFF5832, 235, emissive=2.6)


def pose_and_draw_queadluun(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    pose: MechaPose,
    stride_phase: Float32,
    firing: Bool = False,
):
    """Articulates Miriya's crimson `queadluun_rau.glb` Quadrono Ace power armor with twin emerald-cyan thruster plumes."""
    if len(assets.queadluun.nodes) >= 5:
        var sw_l = sin(stride_phase) * Float32(0.22) - Float32(0.28)
        var sw_r = sin(stride_phase + Float32(3.14159)) * Float32(0.22) - Float32(0.28)
        assets.queadluun.nodes[1].rotation = _quat_axis(Vec3(1, 0, 0), sw_l)
        assets.queadluun.nodes[2].rotation = _quat_axis(Vec3(1, 0, 0), Float32(0.54))
        assets.queadluun.nodes[3].rotation = _quat_axis(Vec3(1, 0, 0), sw_r)
        assets.queadluun.nodes[4].rotation = _quat_axis(Vec3(1, 0, 0), Float32(0.54))
        assets.queadluun.transforms_revision += 1

    scene.draw_model(assets.queadluun, pose.to_mat4())

    var aft_dir = pose.world_dir(0.0, -0.14, -1.0)
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var noz = pose.world_pt(sgn * Float32(1.18), Float32(2.08), Float32(-1.32))
        scene.draw_thruster_plume(
            noz,
            aft_dir,
            3.2 * pose.scale,
            0.28 * pose.scale,
            0xFFE8FFF8,
            0xFF38FFB2,
            alpha=225,
            emissive=2.3,
        )
        if firing:
            var g0 = pose.world_pt(sgn * Float32(1.24), Float32(1.22), Float32(1.94))
            var g1 = pose.world_pt(sgn * Float32(1.24), Float32(1.22), Float32(8.80))
            scene.draw_billboard(g0, 0.48 * pose.scale, FX_SOFT, 0xFFFF88D8, 240, emissive=2.6)
            scene.draw_beam_3d(g0, g1, 0.17 * pose.scale, FX_BOLT, 0xFFFF38A8, 230, emissive=2.5)


def draw_thuverl_salan_boss(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    pose: MechaPose,
    firing: Bool = False,
):
    """Renders the colossal Zentraedi Thuverl-Salan Heavy Destroyer capital warship + plasma drives."""
    scene.draw_model(assets.thuverl_salan, pose.to_mat4())

    var aft = pose.world_dir(0.0, 0.0, -1.0)
    for k in range(4):
        var dx = Float32(-8.4) + Float32(k) * Float32(5.6)
        var noz = pose.world_pt(dx, Float32(-0.6), Float32(-47.5))
        scene.draw_thruster_plume(
            noz,
            aft,
            19.0 * pose.scale,
            3.2 * pose.scale,
            0xFFE8FFEE,
            0xFF42FF88,
            alpha=225,
            emissive=2.45,
        )

    if firing:
        var maw = pose.world_pt(0.0, 1.2, 55.5)
        var maw_end = pose.world_pt(0.0, 1.2, 92.0)
        scene.draw_billboard(maw, 5.2 * pose.scale, FX_SOFT, 0xFFFF6898, 240, emissive=2.9)
        scene.draw_beam_3d(maw, maw_end, 1.45 * pose.scale, FX_BOLT, 0xFFFF2E6E, 235, emissive=2.8)


def draw_macross_environment(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    player_pos: Vec3,
    to_sun: Vec3,
    theater: Int,
    sim_time: Float32 = 0.0,
    reflex_beam_t: Float32 = 0.0,
    reflex_target_pos: Vec3 = Vec3(),
):
    """Renders Mission 1 Atmospheric Macross Island (Pacific Ocean + Cloud Deck) or High Orbit / Fleet Boss Sector + SDF-1 Reflex Cannon!"""
    var cam = scene.fb.camera

    if theater == THEATER_MACROSS_ISLAND:
        # 1A. Sunlit Tropical Pacific Sky Glow + High Cirrus Wisps
        var sun_pos = cam + to_sun * Float32(420.0)
        scene.draw_billboard(sun_pos, 56.0, FX_SOFT, 0xFFFFDF9E, 105, emissive=1.55)
        scene.draw_billboard(sun_pos, 18.0, FX_SOFT, 0xFFFFFCF0, 245, emissive=2.95)

        # 2A. Curved Sunlit South Pacific Ocean + Sculpted 3D South Ataria (Macross) Island Caldera
        var sea_mat = trs(Vec3(player_pos.x * Float32(0.35), Float32(0.0), player_pos.z * Float32(0.35)), Quat(), Vec3(1.0, 1.0, 1.0))
        scene.draw_model(assets.macross_ocean, sea_mat)

        # 3A. Volumetric Cumulus Cloud Deck (`flint.render.scene_renderer.draw_cloud_deck`)
        scene.draw_cloud_deck(
            player_pos,
            altitude=52.0,
            span=410.0,
            count=26,
            sunlit_color=0xFFF6FAFF,
            shadow_color=0xFF7D9EC6,
            alpha=138,
        )
    else:
        # 1B. Deep-space Macross nebula clouds & crisp orbital starfield on far celestial sphere
        for n_i in range(6):
            var ang = Float32(n_i) * Float32(1.0472) + Float32(0.35)
            var elev = Float32(0.16) + Float32(n_i % 3) * Float32(0.14)
            var neb_pos = cam + Vec3(cos(ang) * cos(elev), sin(elev), sin(ang) * cos(elev)) * Float32(460.0)
            var neb_col = (
                UInt32(0xFF4A1E44)
                if theater == THEATER_BOSS_FLEET and (n_i & 1) == 0
                else (UInt32(0xFF1A3A6E) if (n_i & 1) == 0 else UInt32(0xFF3A1E5C))
            )
            scene.draw_billboard(neb_pos, 95.0, FX_SOFT, neb_col, 56, emissive=0.58)

        for i in range(110):
            var h1 = Float32((i * 73 + 19) % 360) * Float32(0.0174533)
            var h2 = Float32((i * 41 + 7) % 80) * Float32(0.013) - Float32(0.04)
            var star_pos = cam + Vec3(cos(h1) * cos(h2), sin(h2), sin(h1) * cos(h2)) * Float32(440.0)
            var scol = UInt32(0xFFD8EEFF) if (i & 3) == 0 else (UInt32(0xFFFFE4B8) if (i & 7) == 1 else UInt32(0xFF7D9BC4))
            var srad = Float32(1.35) if (i & 7) == 0 else Float32(0.75)
            scene.draw_billboard(star_pos, srad, FX_SOFT, scol, 220, emissive=1.85)

        # 2B. Blinding Orbital Solar Corona in Key Light Direction
        var sun_pos = cam + to_sun * Float32(420.0)
        scene.draw_billboard(sun_pos, 46.0, FX_SOFT, 0xFFFFB868, 95, emissive=1.8)
        scene.draw_billboard(sun_pos, 16.5, FX_SOFT, 0xFFFFF8E8, 245, emissive=3.2)

        # 3B. Curved Sunlit Orbital Earth Horizon + Cyan Rayleigh Atmosphere Below Combat Sector
        var earth_mat = trs(Vec3(player_pos.x, -82.0, player_pos.z + 110.0), Quat(), Vec3(1.0, 1.0, 1.0))
        scene.draw_model(assets.earth, earth_mat, no_fog=True)

        # 4B. Subtle Translucent Tactical Orbital Combat Reference Grid at Y=0
        var snap_x = Float32(Int(player_pos.x / Float32(28.0)) * 28)
        var snap_z = Float32(Int(player_pos.z / Float32(28.0)) * 28)
        for g in range(-5, 6):
            var off = Float32(g * 28)
            var col = UInt32(0xFF2E82B8) if g == 0 else UInt32(0xFF184268)
            var alpha = UInt8(95) if g == 0 else UInt8(48)
            scene.draw_beam_3d(
                Vec3(snap_x + off, 0.0, snap_z - 140.0),
                Vec3(snap_x + off, 0.0, snap_z + 140.0),
                0.09,
                FX_BOLT,
                col,
                alpha,
                emissive=0.85,
            )
            scene.draw_beam_3d(
                Vec3(snap_x - 140.0, 0.0, snap_z + off),
                Vec3(snap_x + 140.0, 0.0, snap_z + off),
                0.09,
                FX_BOLT,
                col,
                alpha,
                emissive=0.85,
            )

        # Distant Zentraedi Command Fleet escort cruisers in Mission 3!
        if theater == THEATER_BOSS_FLEET:
            for fl_i in range(2):
                var fx = Float32(-195.0) if fl_i == 0 else Float32(210.0)
                var fy = Float32(84.0) + Float32(fl_i) * Float32(22.0)
                var fz = player_pos.z + Float32(340.0) + Float32(fl_i) * Float32(35.0)
                var f_pose = MechaPose.from_angles(Vec3(player_pos.x * Float32(0.25) + fx, fy, fz), Float32(3.05), Float32(-0.04), Float32(0.0), Float32(0.72))
                scene.draw_model(assets.thuverl_salan, f_pose.to_mat4(), no_fog=True)

    # 5. Colossal SDF-1 Super Dimension Fortress Battlecruiser Overhead/Flank
    var sdf_pos = (
        Vec3(player_pos.x * Float32(0.25) - Float32(138.0), Float32(82.0), player_pos.z + Float32(76.0))
        if theater == THEATER_BOSS_FLEET or reflex_beam_t > Float32(0.0)
        else Vec3(
            player_pos.x * Float32(0.25) - Float32(52.0),
            Float32(56.0) if theater == THEATER_MACROSS_ISLAND else Float32(66.0),
            player_pos.z + Float32(275.0),
        )
    )
    var sdf_yaw = Float32(0.68) if (theater == THEATER_BOSS_FLEET or reflex_beam_t > Float32(0.0)) else Float32(-2.52)
    var sdf_pose = MechaPose.from_angles(
        sdf_pos,
        sdf_yaw,
        Float32(-0.08),
        Float32(0.06),
        Float32(1.05),
    )
    scene.draw_model(assets.sdf1, sdf_pose.to_mat4(), no_fog=True)
    for k in range(4):
        var fx = Float32(-9.0) + Float32(k) * Float32(6.0)
        var noz = sdf_pose.world_pt(fx, -1.0, -47.0)
        var aft = sdf_pose.world_dir(0.0, 0.0, -1.0)
        scene.draw_thruster_plume(noz, aft, 22.0, 3.8, 0xFFE8FFFF, 0xFF38D4FF, alpha=220, emissive=2.4)

    # 6. Twin Colossal SDF-1 Main Boom Reflex Cannon Beams (`draw_colossal_beam_3d`)!
    if reflex_beam_t > Float32(0.0):
        var fade = min(Float32(1.0), reflex_beam_t / Float32(0.45))
        var pulse = Float32(1.0) + sin(sim_time * Float32(28.0)) * Float32(0.12)
        var b_alpha = UInt8(max(40, min(250, Int(fade * Float32(246.0)))))
        var boom_l = sdf_pose.world_pt(-8.6, 3.6, 94.0)
        var boom_r = sdf_pose.world_pt(8.6, 3.6, 94.0)
        scene.draw_colossal_beam_3d(
            boom_l,
            reflex_target_pos,
            radius=2.55 * pulse,
            core_color=0xFFFFFCE2,
            sheath_color=0xFF42EAFF,
            scroll=sim_time * Float32(5.2),
            alpha=b_alpha,
            emissive=3.35,
        )
        scene.draw_colossal_beam_3d(
            boom_r,
            reflex_target_pos,
            radius=2.55 * pulse,
            core_color=0xFFFFFCE2,
            sheath_color=0xFF42EAFF,
            scroll=sim_time * Float32(5.2) + Float32(0.5),
            alpha=b_alpha,
            emissive=3.35,
        )
        # Cataclysmic golden-cyan Reflex convergence sphere at impact point
        scene.draw_billboard(reflex_target_pos, 18.5 * pulse * fade, FX_SOFT, 0xFF42EAFF, UInt8(Int(b_alpha) * 3 // 4), emissive=2.8)
        scene.draw_billboard(reflex_target_pos, 9.5 * pulse * fade, FX_SOFT, 0xFFFFFCE8, b_alpha, emissive=3.6)


@always_inline
def clamp01(v: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), v))

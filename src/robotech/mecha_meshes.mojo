"""Articulated GLB 2.0 Mecha Rigging & Orbital Macross Environment (`flint.render.scene_renderer`).

Drives continuous 3-mode transformation on `vf1s_valkyrie.glb`, reverse-knee
stride kinematics on `regult_pod.glb` & `glaug_officer.glb`, volumetric
afterburner Mach shock-diamonds, Pinpoint Barrier shields, and the orbital
SDF-1 Macross + sunlit Earth horizon scene.
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
):
    """Articulates the 14-node `vf1s_valkyrie.glb` hierarchy across Fighter, Guardian & Battloid and draws PBR + VFX."""
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

    # Volumetric Thermonuclear Afterburner Plumes + Mach Shock Diamonds from actual nozzle sockets
    var globals = model.global_transforms()
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


def draw_orbital_macross_environment(
    mut scene: GpuSceneRenderer,
    mut assets: RobotechAssetPack,
    player_pos: Vec3,
    to_sun: Vec3,
):
    """Renders celestial stardome, solar corona, curved sunlit orbital Earth + Rayleigh limb, combat deck, and SDF-1 Cruiser."""
    var cam = scene.fb.camera

    # 1. Deep-space Macross nebula clouds & crisp orbital starfield on far celestial sphere
    for n_i in range(6):
        var ang = Float32(n_i) * Float32(1.0472) + Float32(0.35)
        var elev = Float32(0.16) + Float32(n_i % 3) * Float32(0.14)
        var neb_pos = cam + Vec3(cos(ang) * cos(elev), sin(elev), sin(ang) * cos(elev)) * Float32(460.0)
        var neb_col = UInt32(0xFF1A3A6E) if (n_i & 1) == 0 else UInt32(0xFF3A1E5C)
        scene.draw_billboard(neb_pos, 95.0, FX_SOFT, neb_col, 52, emissive=0.55)

    for i in range(110):
        var h1 = Float32((i * 73 + 19) % 360) * Float32(0.0174533)
        var h2 = Float32((i * 41 + 7) % 80) * Float32(0.013) - Float32(0.04)
        var star_pos = cam + Vec3(cos(h1) * cos(h2), sin(h2), sin(h1) * cos(h2)) * Float32(440.0)
        var scol = UInt32(0xFFD8EEFF) if (i & 3) == 0 else (UInt32(0xFFFFE4B8) if (i & 7) == 1 else UInt32(0xFF7D9BC4))
        var srad = Float32(1.35) if (i & 7) == 0 else Float32(0.75)
        scene.draw_billboard(star_pos, srad, FX_SOFT, scol, 220, emissive=1.85)

    # 2. Blinding Orbital Solar Corona in Key Light Direction
    var sun_pos = cam + to_sun * Float32(420.0)
    scene.draw_billboard(sun_pos, 46.0, FX_SOFT, 0xFFFFB868, 95, emissive=1.8)
    scene.draw_billboard(sun_pos, 16.5, FX_SOFT, 0xFFFFF8E8, 245, emissive=3.2)

    # 3. Curved Sunlit Orbital Earth Horizon + Cyan Rayleigh Atmosphere Below Combat Sector
    var earth_mat = trs(Vec3(player_pos.x, -82.0, player_pos.z + 110.0), Quat(), Vec3(1.0, 1.0, 1.0))
    scene.draw_model(assets.earth, earth_mat, no_fog=True)

    # 4. Subtle Translucent Tactical Orbital Combat Reference Grid at Y=0
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

    # 5. Colossal SDF-1 Super Dimension Fortress Battlecruiser in High Orbit Ahead
    var sdf_pose = MechaPose.from_angles(
        Vec3(player_pos.x * Float32(0.25) - Float32(52.0), Float32(66.0), player_pos.z + Float32(275.0)),
        Float32(-2.52),
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


@always_inline
def clamp01(v: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), v))

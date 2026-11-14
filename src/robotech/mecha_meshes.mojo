"""Procedural 3D Robotech Mecha & SDF-1 Cruiser Geometry for `flint.render.simd_canvas`.

Renders:
  - VF-1S / VF-1J Valkyrie Veritech with continuous 3-mode transformation:
      Fighter (`morph_fg=0, morph_gb=0`) -> Guardian/GERWALK (`morph_fg=1, morph_gb=0`)
      -> Battloid (`morph_fg=1, morph_gb=1`)
  - Zentraedi Regult Tactical Battlepod (ovoid pod, crimson cyclops eye, dorsal cannons, ostrich legs)
  - Zentraedi Glaug Officer's Pod (command chassis, dorsal heavy plasma cannon, twin particle arms)
  - SDF-1 Super Dimension Fortress cruiser & holographic combat training sector
"""
from std.math import sin, cos
from flint.core.math import Vec3
from flint.render.simd_canvas import Camera3D, SimdCanvas


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
def _lerp(a: Float32, b: Float32, t: Float32) -> Float32:
    return a + (b - a) * t


@always_inline
def _lerp3(a: Vec3, b: Vec3, t: Float32) -> Vec3:
    return a + (b - a) * t


def draw_taper_segment(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    pose: MechaPose,
    p0: Vec3,
    hw0: Float32,
    hh0: Float32,
    p1: Vec3,
    hw1: Float32,
    hh1: Float32,
    color: UInt32,
    sun: Vec3,
):
    """Renders an 8-corner, 12-triangle 3D tapered hull/limb segment in local pose coordinates."""
    var a0 = pose.world_pt(p0.x - hw0, p0.y - hh0, p0.z)
    var a1 = pose.world_pt(p0.x + hw0, p0.y - hh0, p0.z)
    var a2 = pose.world_pt(p0.x + hw0, p0.y + hh0, p0.z)
    var a3 = pose.world_pt(p0.x - hw0, p0.y + hh0, p0.z)

    var b0 = pose.world_pt(p1.x - hw1, p1.y - hh1, p1.z)
    var b1 = pose.world_pt(p1.x + hw1, p1.y - hh1, p1.z)
    var b2 = pose.world_pt(p1.x + hw1, p1.y + hh1, p1.z)
    var b3 = pose.world_pt(p1.x - hw1, p1.y + hh1, p1.z)

    # Top & Bottom
    canvas.draw_tri_3d_lit(cam, a3, a2, b2, color, sun)
    canvas.draw_tri_3d_lit(cam, a3, b2, b3, color, sun)
    canvas.draw_tri_3d_lit(cam, a0, b1, a1, color, sun)
    canvas.draw_tri_3d_lit(cam, a0, b0, b1, color, sun)
    # Right & Left
    canvas.draw_tri_3d_lit(cam, a1, b1, b2, color, sun)
    canvas.draw_tri_3d_lit(cam, a1, b2, a2, color, sun)
    canvas.draw_tri_3d_lit(cam, a0, a3, b3, color, sun)
    canvas.draw_tri_3d_lit(cam, a0, b3, b0, color, sun)
    # End caps
    canvas.draw_tri_3d_lit(cam, a0, a1, a2, color, sun)
    canvas.draw_tri_3d_lit(cam, a0, a2, a3, color, sun)
    canvas.draw_tri_3d_lit(cam, b0, b2, b1, color, sun)
    canvas.draw_tri_3d_lit(cam, b0, b3, b2, color, sun)


def draw_box_local(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    pose: MechaPose,
    cx: Float32,
    cy: Float32,
    cz: Float32,
    hx: Float32,
    hy: Float32,
    hz: Float32,
    color: UInt32,
    sun: Vec3,
):
    draw_taper_segment(
        canvas, cam, pose,
        Vec3(cx, cy, cz - hz), hx, hy,
        Vec3(cx, cy, cz + hz), hx, hy,
        color, sun,
    )


def draw_veritech_valkyrie(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    pose: MechaPose,
    morph_fg: Float32,
    morph_gb: Float32,
    trim_color: UInt32 = 0xFFD038,
    accent_color: UInt32 = 0xE83A3A,
    afterburner: Float32 = 0.6,
    gun_flash: Bool = False,
    shield_active: Bool = False,
    sun: Vec3 = Vec3(0.42, 0.82, 0.38),
):
    """Renders an articulated VF-1 Valkyrie across Fighter, Guardian (GERWALK) & Battloid."""
    var hull_white = UInt32(0xEEF3FA)
    var joint_dark = UInt32(0x2B3545)
    var gunmetal = UInt32(0x4E5F75)
    var canopy_amber = UInt32(0xFFB838)
    var visor_green = UInt32(0x38FFB0)
    var thruster_cyan = UInt32(0x4CE8FF)

    # 1. Central Torso / Airframe Spine
    var torso_y = _lerp(Float32(0.0), Float32(0.75), morph_gb)
    var torso_len = _lerp(Float32(1.45), Float32(0.68), morph_gb)
    draw_box_local(canvas, cam, pose, 0.0, torso_y, 0.0, 0.72, _lerp(0.28, 0.72, morph_gb), torso_len, hull_white, sun)
    # Skull Squadron collar stripe
    draw_box_local(canvas, cam, pose, 0.0, torso_y + _lerp(0.29, 0.68, morph_gb), 0.15, 0.66, 0.09, 0.42, trim_color, sun)
    draw_box_local(canvas, cam, pose, 0.0, torso_y + _lerp(0.31, 0.73, morph_gb), -0.15, 0.50, 0.07, 0.25, joint_dark, sun)

    # 2. Nosecone & Canopy (folds down into Battloid chestplate as morph_gb -> 1)
    var nose_base = Vec3(0.0, _lerp(0.04, 0.85, morph_gb), _lerp(1.25, 0.62, morph_gb))
    var nose_tip = Vec3(0.0, _lerp(-0.04, -0.38, morph_gb), _lerp(4.15, 0.92, morph_gb))
    draw_taper_segment(
        canvas, cam, pose,
        nose_base, _lerp(0.46, 0.62, morph_gb), _lerp(0.26, 0.24, morph_gb),
        nose_tip, 0.08, 0.07,
        hull_white, sun,
    )
    # Canopy bubble (visible in Fighter & Guardian, armored heatshield in Battloid)
    var canopy_col = canopy_amber if morph_gb < Float32(0.55) else accent_color
    draw_taper_segment(
        canvas, cam, pose,
        Vec3(0.0, nose_base.y + _lerp(0.24, 0.05, morph_gb), nose_base.z + _lerp(0.2, 0.12, morph_gb)),
        0.26, 0.16,
        Vec3(0.0, _lerp(0.10, -0.05, morph_gb), _lerp(2.75, 0.88, morph_gb)),
        0.10, 0.06,
        canopy_col, sun,
    )

    # 3. Battloid VF-1S Head Unit + Quad Laser Antennas (emerges as morph_gb -> 1)
    if morph_gb > Float32(0.22):
        var hy = torso_y + Float32(0.95) * morph_gb
        draw_box_local(canvas, cam, pose, 0.0, hy, 0.08, 0.30, 0.26, 0.30, hull_white, sun)
        draw_box_local(canvas, cam, pose, 0.0, hy + 0.02, 0.36, 0.24, 0.10, 0.06, visor_green, sun)
        # Iconic VF-1S swept head laser cannons
        canvas.draw_line_3d(cam, pose.world_pt(-0.32, hy + 0.08, 0.05), pose.world_pt(-0.55, hy + 0.42, -0.35), trim_color, 2)
        canvas.draw_line_3d(cam, pose.world_pt(0.32, hy + 0.08, 0.05), pose.world_pt(0.55, hy + 0.42, -0.35), trim_color, 2)

    # 4. Variable-Sweep Wings (wide in Fighter/Guardian, folded into dorsal pack in Battloid)
    var wing_span = _lerp(Float32(3.85), Float32(1.35), morph_gb)
    var wing_sweep_z = _lerp(Float32(-1.45) - afterburner * Float32(0.55), Float32(-1.30), morph_gb)
    var wing_tip_y = _lerp(Float32(0.04), Float32(-0.85), morph_gb)
    var wing_root_y = _lerp(Float32(0.08), torso_y + Float32(0.35), morph_gb)
    var wing_root_z = _lerp(Float32(-0.15), Float32(-0.62), morph_gb)
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var r0 = pose.world_pt(sgn * 0.68, wing_root_y, wing_root_z + 0.55)
        var r1 = pose.world_pt(sgn * 0.68, wing_root_y, wing_root_z - 0.65)
        var t0 = pose.world_pt(sgn * wing_span, wing_tip_y, wing_sweep_z + 0.25)
        var t1 = pose.world_pt(sgn * wing_span, wing_tip_y, wing_sweep_z - 0.32)
        canvas.draw_tri_3d_lit(cam, r0, t0, t1, hull_white, sun)
        canvas.draw_tri_3d_lit(cam, r0, t1, r1, hull_white, sun)
        canvas.draw_line_3d(cam, r0, t0, accent_color, 2)
        # Twin angled tailfins
        var fin_base = pose.world_pt(sgn * 0.55, wing_root_y + 0.15, wing_root_z - 0.75)
        var fin_top = pose.world_pt(sgn * 0.88, wing_root_y + _lerp(0.98, 0.55, morph_gb), wing_root_z - 1.35)
        var fin_back = pose.world_pt(sgn * 0.55, wing_root_y + 0.15, wing_root_z - 1.45)
        canvas.draw_tri_3d_lit(cam, fin_base, fin_top, fin_back, trim_color, sun)

    # 5. Twin Engine Nacelles -> Reverse-Knee GERWALK Legs -> Upright Battloid Legs
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var lx = sgn * _lerp(Float32(0.52), Float32(0.58), morph_fg)
        if morph_fg < Float32(0.15):
            # Pure Fighter streamlined engine nacelles + glowing afterburners
            draw_box_local(canvas, cam, pose, lx, -0.06, -1.55, 0.30, 0.26, 1.35, hull_white, sun)
            draw_box_local(canvas, cam, pose, lx, -0.06, -3.02, 0.22, 0.20, 0.18, thruster_cyan, sun)
            if afterburner > Float32(0.2):
                var plume_len = Float32(3.2) + afterburner * Float32(2.6)
                canvas.draw_line_3d(
                    cam,
                    pose.world_pt(lx, -0.06, -3.15),
                    pose.world_pt(lx, -0.06, -plume_len),
                    thruster_cyan, 2,
                )
        else:
            # Articulated Thigh -> Reverse-Knee (Guardian) or Straight Leg (Battloid)
            var hip = Vec3(lx, _lerp(-0.15, -0.05, morph_gb), _lerp(-0.35, 0.0, morph_gb))
            var knee = Vec3(lx, _lerp(-0.95, -1.25, morph_gb), _lerp(0.95, 0.18, morph_gb))
            var ankle = Vec3(lx, _lerp(-2.05, -2.55, morph_gb), _lerp(-0.35, -0.05, morph_gb))
            draw_taper_segment(canvas, cam, pose, hip, 0.24, 0.24, knee, 0.20, 0.20, hull_white, sun)
            draw_taper_segment(canvas, cam, pose, knee, 0.28, 0.28, ankle, 0.32, 0.26, hull_white, sun)
            draw_box_local(canvas, cam, pose, ankle.x, ankle.y - 0.14, ankle.z + 0.18, 0.28, 0.14, 0.42, joint_dark, sun)
            # VTOL foot thruster plume
            canvas.draw_line_3d(
                cam,
                pose.world_pt(ankle.x, ankle.y - 0.25, ankle.z),
                pose.world_pt(ankle.x, ankle.y - 1.15, ankle.z - 0.25),
                thruster_cyan, 2,
            )

    # 6. Deployed Arms, GU-11 55mm Gunpod & Optional Pinpoint Barrier (Guardian & Battloid)
    if morph_fg > Float32(0.2):
        var sh_y = _lerp(-0.22, torso_y + 0.42, morph_gb)
        var sh_z = _lerp(0.15, 0.05, morph_gb)
        for side in range(2):
            var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
            var sh = Vec3(sgn * 1.06, sh_y, sh_z)
            var hand = Vec3(sgn * 1.02, sh_y - _lerp(0.45, 0.88, morph_gb), sh_z + _lerp(1.35, 1.05, morph_gb))
            draw_box_local(canvas, cam, pose, sh.x, sh.y, sh.z, 0.28, 0.28, 0.30, hull_white, sun)
            draw_box_local(canvas, cam, pose, sh.x, sh.y + 0.29, sh.z, 0.24, 0.06, 0.24, accent_color, sun)
            draw_taper_segment(canvas, cam, pose, sh, 0.18, 0.18, hand, 0.16, 0.16, hull_white, sun)

            if side == 1:
                # Iconic GU-11 Gunpod held in right manipulator
                draw_taper_segment(
                    canvas, cam, pose,
                    Vec3(hand.x, hand.y - 0.08, hand.z - 0.45), 0.14, 0.16,
                    Vec3(hand.x, hand.y - 0.04, hand.z + 1.85), 0.09, 0.10,
                    gunmetal, sun,
                )
                if gun_flash:
                    canvas.draw_line_3d(
                        cam,
                        pose.world_pt(hand.x, hand.y - 0.04, hand.z + 1.90),
                        pose.world_pt(hand.x, hand.y - 0.04, hand.z + 4.80),
                        0xFFF078, 3,
                    )
            elif shield_active:
                # Glowing Cyan Pinpoint Barrier Disc on Left Arm
                var sc = Vec3(hand.x - 0.28, hand.y + 0.10, hand.z + 0.35)
                draw_box_local(canvas, cam, pose, sc.x, sc.y, sc.z, 0.08, 0.95, 0.95, 0x38E8FF, sun)
    else:
        # Gunpod slung cleanly under Fighter belly
        draw_box_local(canvas, cam, pose, 0.0, -0.36, -0.35, 0.13, 0.12, 1.15, gunmetal, sun)
        if gun_flash:
            canvas.draw_line_3d(cam, pose.world_pt(0.0, -0.36, 0.85), pose.world_pt(0.0, -0.36, 4.2), 0xFFF078, 3)


def draw_regult_battlepod(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    pose: MechaPose,
    stride_phase: Float32,
    firing: Bool = False,
    sun: Vec3 = Vec3(0.42, 0.82, 0.38),
):
    """Renders a Zentraedi Regult Tactical Battlepod (blue-violet ovoid pod + crimson cyclops + ostrich legs)."""
    var pod_blue = UInt32(0x6C86BD)
    var pod_dark = UInt32(0x3C4C70)
    var leg_grey = UInt32(0x8899B8)
    var eye_red = UInt32(0xFF2848)

    # Ovoid main pod hull (front dome + rear booster)
    draw_taper_segment(
        canvas, cam, pose,
        Vec3(0.0, 0.55, -1.15), 0.95, 0.85,
        Vec3(0.0, 0.45, 0.65), 1.18, 1.05,
        pod_blue, sun,
    )
    draw_taper_segment(
        canvas, cam, pose,
        Vec3(0.0, 0.45, 0.65), 1.18, 1.05,
        Vec3(0.0, 0.38, 1.75), 0.48, 0.44,
        pod_blue, sun,
    )
    # Glowing crimson Zentraedi Cyclops Sensor Eye
    draw_box_local(canvas, cam, pose, 0.0, 0.42, 1.78, 0.32, 0.26, 0.12, eye_red, sun)

    # Twin top-mounted anti-aircraft laser cannons
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        draw_box_local(canvas, cam, pose, sgn * 0.42, 1.62, -0.15, 0.14, 0.22, 0.45, pod_dark, sun)
        var muz = pose.world_pt(sgn * 0.42, 1.72, 2.05)
        canvas.draw_line_3d(cam, pose.world_pt(sgn * 0.42, 1.72, 0.25), muz, leg_grey, 2)
        if firing:
            canvas.draw_line_3d(cam, muz, pose.world_pt(sgn * 0.42, 1.72, 5.2), 0xFF4268, 2)

    # Reverse-jointed ostrich legs
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var swing = sin(stride_phase + (Float32(0.0) if side == 0 else Float32(3.14159))) * Float32(0.35)
        var hip = Vec3(sgn * 0.92, -0.15, 0.0)
        var rknee = Vec3(sgn * 0.98, -1.15, -0.95 + swing)
        var foot = Vec3(sgn * 0.95, -2.55, 0.45 - swing)
        draw_taper_segment(canvas, cam, pose, hip, 0.20, 0.20, rknee, 0.16, 0.16, leg_grey, sun)
        draw_taper_segment(canvas, cam, pose, rknee, 0.16, 0.16, foot, 0.14, 0.14, leg_grey, sun)
        draw_box_local(canvas, cam, pose, foot.x, foot.y - 0.10, foot.z + 0.20, 0.28, 0.10, 0.45, pod_dark, sun)


def draw_glaug_officer(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    pose: MechaPose,
    stride_phase: Float32,
    firing: Bool = False,
    sun: Vec3 = Vec3(0.42, 0.82, 0.38),
):
    """Renders a Zentraedi Glaug Officer's Command Pod (crimson armor, dorsal rail-cannon, twin beam arms)."""
    var crimson = UInt32(0xCC2E46)
    var slate = UInt32(0x2B3446)
    var gold_eye = UInt32(0xFFD248)

    # Wide flattened shark-command hull
    draw_taper_segment(
        canvas, cam, pose,
        Vec3(0.0, 0.50, -1.55), 1.35, 0.65,
        Vec3(0.0, 0.38, 1.95), 0.55, 0.35,
        crimson, sun,
    )
    draw_box_local(canvas, cam, pose, 0.0, 0.42, 1.98, 0.42, 0.16, 0.12, gold_eye, sun)

    # Iconic long overhead dorsal heavy plasma cannon
    draw_box_local(canvas, cam, pose, 0.0, 1.45, -0.55, 0.28, 0.36, 0.75, slate, sun)
    draw_taper_segment(
        canvas, cam, pose,
        Vec3(0.0, 1.65, -0.10), 0.16, 0.16,
        Vec3(0.0, 1.68, 3.25), 0.10, 0.10,
        slate, sun,
    )
    if firing:
        canvas.draw_line_3d(cam, pose.world_pt(0.0, 1.68, 3.30), pose.world_pt(0.0, 1.68, 8.0), 0xFF6038, 3)

    # Outstretched twin heavy particle-beam arms
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var sh = Vec3(sgn * 1.35, 0.42, 0.15)
        var elbow = Vec3(sgn * 2.15, 0.25, 0.65)
        var claw = Vec3(sgn * 2.05, 0.15, 2.25)
        draw_taper_segment(canvas, cam, pose, sh, 0.24, 0.22, elbow, 0.20, 0.18, slate, sun)
        draw_taper_segment(canvas, cam, pose, elbow, 0.22, 0.20, claw, 0.14, 0.14, crimson, sun)
        var swing = sin(stride_phase + (Float32(0.0) if side == 0 else Float32(3.14159))) * Float32(0.30)
        var hip = Vec3(sgn * 0.95, -0.15, -0.15)
        var knee = Vec3(sgn * 1.05, -1.35, -1.05 + swing)
        var foot = Vec3(sgn * 1.00, -2.85, 0.45 - swing)
        draw_taper_segment(canvas, cam, pose, hip, 0.24, 0.24, knee, 0.20, 0.20, slate, sun)
        draw_taper_segment(canvas, cam, pose, knee, 0.22, 0.22, foot, 0.20, 0.20, crimson, sun)


def draw_sdf1_environment(
    mut canvas: SimdCanvas,
    cam: Camera3D,
    player_pos: Vec3,
    sun: Vec3 = Vec3(0.42, 0.82, 0.38),
):
    """Renders deep-space starfield, SDF-1 holographic combat grid, cover pylons & the SDF-1 Macross Cruiser."""
    # 1. Deterministic deep-space starfield projected on sky sphere around camera
    for i in range(96):
        var h1 = Float32((i * 73 + 19) % 360) * Float32(0.0174533)
        var h2 = Float32((i * 41 + 7) % 75) * Float32(0.012) + Float32(0.06)
        var star_pos = cam.eye + Vec3(cos(h1) * cos(h2), sin(h2), sin(h1) * cos(h2)) * Float32(420.0)
        var sp = cam.project(star_pos)
        if sp[3]:
            var sx = Int(sp[0])
            var sy = Int(sp[1])
            var scol = UInt32(0xCFE8FF) if (i & 3) == 0 else UInt32(0x6D86A8)
            canvas.draw_rect(sx, sy, 2 if (i & 7) == 0 else 1, 2 if (i & 7) == 0 else 1, scol)

    # 2. Holographic SDF-1 Combat Deck Grid (follows snapped around player X/Z)
    var snap_x = Float32(Int(player_pos.x / Float32(20.0)) * 20)
    var snap_z = Float32(Int(player_pos.z / Float32(20.0)) * 20)
    for g in range(-8, 9):
        var off = Float32(g * 20)
        var col = UInt32(0x23486E) if g == 0 else UInt32(0x142942)
        canvas.draw_line_3d(
            cam,
            Vec3(snap_x + off, 0.0, snap_z - 160.0),
            Vec3(snap_x + off, 0.0, snap_z + 160.0),
            col, 1,
        )
        canvas.draw_line_3d(
            cam,
            Vec3(snap_x - 160.0, 0.0, snap_z + off),
            Vec3(snap_x + 160.0, 0.0, snap_z + off),
            col, 1,
        )

    # 3. Colossal SDF-1 Super Dimension Fortress Battlecruiser Silhouette in Orbit Ahead
    var sdf_pose = MechaPose.from_angles(
        Vec3(player_pos.x * Float32(0.25) - Float32(55.0), Float32(68.0), player_pos.z + Float32(285.0)),
        Float32(-2.55), Float32(-0.08), Float32(0.05), Float32(1.0),
    )
    # Central cruiser hull + Bridge tower
    draw_box_local(canvas, cam, sdf_pose, 0.0, 0.0, 0.0, 14.0, 8.5, 42.0, 0x5D6D84, sun)
    draw_box_local(canvas, cam, sdf_pose, 0.0, 15.0, -6.0, 5.5, 9.0, 8.0, 0x8295B0, sun)
    draw_box_local(canvas, cam, sdf_pose, 0.0, 21.0, 2.2, 4.2, 1.5, 0.8, 0x48F5B0, sun)
    # Twin forward Macross Main Boom Cannons
    draw_box_local(canvas, cam, sdf_pose, -8.5, 3.5, 58.0, 3.6, 3.8, 28.0, 0xDCE4F0, sun)
    draw_box_local(canvas, cam, sdf_pose, 8.5, 3.5, 58.0, 3.6, 3.8, 28.0, 0xDCE4F0, sun)
    # Port CVS-101 Prometheus (Amber) & Starboard SLV-111 Daedalus (Sea-Green) Carrier Arms
    draw_box_local(canvas, cam, sdf_pose, -23.0, -3.0, 8.0, 5.2, 4.5, 28.0, 0xD49B38, sun)
    draw_box_local(canvas, cam, sdf_pose, 23.0, -3.0, 8.0, 5.2, 4.5, 28.0, 0x3C8E7A, sun)
    # Glowing Reflex Furnace Main Thrusters
    draw_box_local(canvas, cam, sdf_pose, 0.0, -1.0, -43.5, 10.5, 5.5, 1.5, 0x4CE8FF, sun)

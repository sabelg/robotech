"""Robotech Phase 1 Asset Forge & Native GLB 2.0 Loader (`flint.assets.*`).

Generates (if not yet cached on disk) and loads standard binary glTF 2.0 `.glb`
mecha/cruiser models with smooth curved surfaces, UVs, PBR roughness/metallic
materials, and embedded weathered PNG textures:
  - `assets/models/vf1s_valkyrie.glb` (14-node articulated VF-1 Valkyrie)
  - `assets/models/regult_pod.glb`    (5-node Zentraedi Regult Battlepod)
  - `assets/models/glaug_officer.glb` (5-node Zentraedi Glaug Command Pod)
  - `assets/models/sdf1_macross.glb`  (Colossal SDF-1 Super Dimension Fortress)
  - `assets/sky/earth_orbit.png`      (Orbital Earth ocean/cloud/archipelago map)
"""
from std.math import sin, cos, sqrt
from flint.platform.os import read_bytes
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.image import decode_png, write_png
from flint.assets.model import NativeModel, NativeMesh, NativeTexture
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes, PI_F32
from flint.assets.glb_writer import write_native_glb
from flint.assets.glb_native import load_native_glb
from flint.render.particle_atlas import value_noise


# Articulated node indices inside `vf1s_valkyrie.glb`
comptime VF1_NODE_TORSO: Int = 0
comptime VF1_NODE_NOSE: Int = 1
comptime VF1_NODE_HEAD: Int = 2
comptime VF1_NODE_WING_L: Int = 3
comptime VF1_NODE_WING_R: Int = 4
comptime VF1_NODE_TAIL_L: Int = 5
comptime VF1_NODE_TAIL_R: Int = 6
comptime VF1_NODE_THIGH_L: Int = 7
comptime VF1_NODE_SHIN_L: Int = 8
comptime VF1_NODE_THIGH_R: Int = 9
comptime VF1_NODE_SHIN_R: Int = 10
comptime VF1_NODE_ARM_L: Int = 11
comptime VF1_NODE_ARM_R: Int = 12
comptime VF1_NODE_GUNPOD: Int = 13


def _pack_argb(a: Int, r: Int, g: Int, b: Int) -> UInt32:
    var ca = UInt32(max(0, min(255, a)))
    var cr = UInt32(max(0, min(255, r)))
    var cg = UInt32(max(0, min(255, g)))
    var cb = UInt32(max(0, min(255, b)))
    return (ca << 24) | (cr << 16) | (cg << 8) | cb


def _fbm(x: Float32, y: Float32, seed: Int) -> Float32:
    return (
        value_noise(x, y, seed) * 0.55
        + value_noise(x * 2.13 + 3.7, y * 2.13 + 1.9, seed + 11) * 0.30
        + value_noise(x * 4.41 + 8.2, y * 4.41 + 5.3, seed + 23) * 0.15
    )


def make_vf1_livery_tex(skull_livery: Bool = True) -> NativeTexture:
    """256x256 weathered UN Spacy aerospace plating with panel seams, rivets, Squadron stripes & UN Spacy kite roundel."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)

    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var wear = _fbm(u * 14.0, v * 14.0, 101)
            var streak = value_noise(u * 6.0, v * 38.0, 137)

            # Base warm Macross aerospace white
            var r = Float32(236.0) - (1.0 - wear) * 18.0 - (1.0 - streak) * 8.0
            var g = Float32(241.0) - (1.0 - wear) * 17.0 - (1.0 - streak) * 7.0
            var b = Float32(248.0) - (1.0 - wear) * 14.0 - (1.0 - streak) * 5.0

            # UV [0..0.5, 0.52..0.74]: Squadron livery band (Skull Leader Gold/Black or Skull-2 Max Cobalt/White)
            if u < 0.50 and v >= 0.52 and v <= 0.74:
                if v < 0.64:
                    if skull_livery:
                        r = 252.0; g = 204.0; b = 46.0
                    else:
                        r = 46.0; g = 136.0; b = 248.0
                elif v < 0.67:
                    r = 238.0; g = 242.0; b = 248.0
                else:
                    if skull_livery:
                        r = 28.0; g = 34.0; b = 44.0
                    else:
                        r = 26.0; g = 74.0; b = 168.0
                r *= 0.88 + wear * 0.14
                g *= 0.88 + wear * 0.14
                b *= 0.88 + wear * 0.14

            # UV [0..0.5, 0.76..1.0]: Brushed dark titanium gunmetal / mechanical joints
            elif u < 0.50 and v >= 0.76:
                var brush = value_noise(u * 42.0, v * 8.0, 151)
                r = 66.0 + brush * 26.0
                g = 76.0 + brush * 28.0
                b = 92.0 + brush * 30.0

            # UV [0.52..1.0, 0.52..1.0]: Wing upper/lower skin with UN Spacy crimson/white kite roundel
            elif u >= 0.50 and v >= 0.50:
                var cx = (u - 0.75) * 4.0
                var cy = (v - 0.75) * 4.0
                var rad = sqrt(cx * cx + cy * cy)
                if rad < 0.56:
                    if rad > 0.44:
                        r = 32.0; g = 40.0; b = 52.0
                    elif abs(cx) * 1.35 + abs(cy) < 0.42:
                        r = 246.0; g = 248.0; b = 252.0
                    else:
                        r = 216.0; g = 42.0; b = 52.0

            # Aerospace panel seams + micro-rivets
            var gx = x % 32
            var gy = y % 32
            if gx == 0 or gy == 0:
                r *= 0.74; g *= 0.75; b *= 0.78
            elif gx == 1 or gy == 1:
                r = min(Float32(255.0), r * 1.05)
                g = min(Float32(255.0), g * 1.05)
                b = min(Float32(255.0), b * 1.05)
            elif (gx == 4 or gx == 28) and (gy % 8 == 4):
                r *= 0.80; g *= 0.81; b *= 0.83

            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_regult_armor_tex() -> NativeTexture:
    """256x256 Zentraedi biomechanical lavender-blue battlepod armor with panel grooves & scorch wear."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var mottling = _fbm(u * 11.0, v * 11.0, 211)
            var r = Float32(102.0) + mottling * 32.0
            var g = Float32(128.0) + mottling * 34.0
            var b = Float32(186.0) + mottling * 36.0
            if v > 0.68:
                # Leg & cannon brushed alloy
                r = 124.0 + mottling * 28.0
                g = 138.0 + mottling * 28.0
                b = 164.0 + mottling * 30.0
            var seam_u = abs((u * 6.0) - Float32(Int(u * 6.0)) - 0.5)
            var seam_v = abs((v * 6.0) - Float32(Int(v * 6.0)) - 0.5)
            if seam_u > 0.465 or seam_v > 0.465:
                r *= 0.68; g *= 0.70; b *= 0.74
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_glaug_command_tex() -> NativeTexture:
    """256x256 Zentraedi Glaug Officer crimson/obsidian command armor."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var n = _fbm(u * 12.0, v * 12.0, 307)
            var r = Float32(184.0) + n * 44.0
            var g = Float32(36.0) + n * 22.0
            var b = Float32(62.0) + n * 26.0
            if v > 0.62:
                # Dark obsidian rail-cannon & leg plating
                r = 44.0 + n * 22.0
                g = 52.0 + n * 24.0
                b = 68.0 + n * 28.0
            elif v > 0.48 and v <= 0.56:
                # Zentraedi officer gold command trim
                r = 236.0; g = 192.0; b = 58.0
            if (x % 28 == 0) or (y % 28 == 0):
                r *= 0.70; g *= 0.70; b *= 0.72
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_sdf1_cruiser_tex() -> NativeTexture:
    """256x256 SDF-1 battlecruiser hull plating with illuminated city/bridge deck portholes."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var n = _fbm(u * 16.0, v * 16.0, 409)
            var r = Float32(98.0) + n * 34.0
            var g = Float32(112.0) + n * 36.0
            var b = Float32(134.0) + n * 38.0
            if v > 0.50 and u < 0.50:
                # Prometheus amber carrier deck
                r = 204.0 + n * 26.0; g = 148.0 + n * 22.0; b = 52.0 + n * 16.0
            elif v > 0.50 and u >= 0.50:
                # Daedalus sea-green assault carrier deck
                r = 56.0 + n * 22.0; g = 138.0 + n * 26.0; b = 122.0 + n * 24.0
            if (x % 16 == 0) or (y % 16 == 0):
                r *= 0.72; g *= 0.74; b *= 0.78
            # Illuminated cruiser deck windows
            if v <= 0.48 and (x % 8 == 3) and (y % 6 == 2):
                var lit = value_noise(Float32(x) * 0.7, Float32(y) * 1.3, 443)
                if lit > 0.54:
                    r = 165.0; g = 238.0; b = 255.0
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_earth_orbit_tex() -> NativeTexture:
    """512x256 orbital Earth Pacific Ocean, Macross Island archipelago & swirling cyclone cloud bands."""
    var w = 512
    var h = 256
    var tex = NativeTexture()
    tex.width = w
    tex.height = h
    tex.pixels = List[UInt32](length=w * h, fill=0xFFFFFFFF)
    for y in range(h):
        var v = Float32(y) / Float32(h)
        for x in range(w):
            var u = Float32(x) / Float32(w)
            var cont = _fbm(u * 5.2 + 1.4, v * 3.6 + 0.8, 503)
            var detail = _fbm(u * 18.0, v * 14.0, 521)
            # Deep cobalt/sapphire orbital ocean
            var r = Float32(14.0) + cont * 18.0
            var g = Float32(48.0) + cont * 44.0
            var b = Float32(118.0) + cont * 62.0
            # Shallow turquoise atolls & emerald/ochre landmass
            if cont > 0.56:
                if cont < 0.595:
                    r = 34.0; g = 142.0; b = 168.0
                else:
                    var lush = clamp_f((cont - 0.595) * 4.5)
                    r = 46.0 + lush * 76.0 + detail * 24.0
                    g = 104.0 + (1.0 - lush * 0.35) * 42.0 + detail * 18.0
                    b = 52.0 + detail * 16.0
            # Swirling tropospheric cloud deck
            var swirl = _fbm(u * 9.5 + sin(v * 6.2) * 0.45, v * 7.5 + cos(u * 6.2) * 0.35, 577)
            var cloud = clamp_f((swirl - 0.47) * 3.1)
            cloud = cloud * cloud * (3.0 - 2.0 * cloud)
            r = r * (1.0 - cloud) + 244.0 * cloud
            g = g * (1.0 - cloud) + 248.0 * cloud
            b = b * (1.0 - cloud) + 255.0 * cloud

            # Atmospheric Rayleigh cyan scatter toward horizon edge
            var limb = clamp_f((1.0 - v) * 1.15)
            limb = limb * limb * 0.32
            r = r * (1.0 - limb) + 88.0 * limb
            g = g * (1.0 - limb) + 196.0 * limb
            b = b * (1.0 - limb) + 255.0 * limb

            tex.pixels[y * w + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def clamp_f(v: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), v))


def build_vf1_valkyrie_model(skull_livery: Bool = True) -> NativeModel:
    """Builds a 14-node articulated curved glTF 2.0 VF-1S Valkyrie model."""
    var model = NativeModel()
    model.textures.append(make_vf1_livery_tex(skull_livery))

    # 0. Torso Spine + Twin Air Intake Cowls + Skull Collar
    var torso_hull = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.36, metallic=0.16, texture=0)
    torso_hull.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.70, 0.30, 1.38), bevel=0.10, u0=0.02, v0=0.02, u1=0.48, v1=0.48)
    torso_hull.add_beveled_box(Vec3(0.0, 0.24, -0.35), Vec3(0.54, 0.16, 0.88), bevel=0.07, u0=0.02, v0=0.53, u1=0.48, v1=0.72)
    for s in range(2):
        var sx = Float32(-0.56) if s == 0 else Float32(0.56)
        torso_hull.add_beveled_box(Vec3(sx, -0.10, 0.52), Vec3(0.28, 0.22, 0.62), bevel=0.06, u0=0.02, v0=0.04, u1=0.46, v1=0.46)
    var torso_dark = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.36, metallic=0.72, texture=0)
    for s in range(2):
        var sx = Float32(-0.56) if s == 0 else Float32(0.56)
        torso_dark.add_cylinder(Vec3(sx, -0.10, 0.95), Vec3(sx, -0.10, 1.16), 0.19, 0.21, 14, u0=0.04, v0=0.78, u1=0.46, v1=0.98)
    var torso_prims = List[NativeMesh]()
    torso_prims.append(torso_hull^.finish())
    torso_prims.append(torso_dark^.finish())
    var n_torso = append_node_meshes(model, "torso_core", torso_prims)

    # 1. Curved Parabolic Ogive Nosecone + Glossy Amber Canopy + Pitot Probe
    var nose_hull = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.35, metallic=0.16, texture=0)
    var nz = List[Float32]()
    var nrx = List[Float32]()
    var nry = List[Float32]()
    var ncy = List[Float32]()
    nz.append(0.0);  nrx.append(0.48); nry.append(0.32); ncy.append(0.02)
    nz.append(0.75); nrx.append(0.44); nry.append(0.30); ncy.append(0.01)
    nz.append(1.55); nrx.append(0.35); nry.append(0.24); ncy.append(-0.01)
    nz.append(2.25); nrx.append(0.22); nry.append(0.16); ncy.append(-0.03)
    nz.append(2.75); nrx.append(0.09); nry.append(0.07); ncy.append(-0.05)
    nz.append(3.05); nrx.append(0.015); nry.append(0.015); ncy.append(-0.06)
    nose_hull.add_lathe_z(nz, nrx, nry, ncy, segments=20, u0=0.02, v0=0.04, u1=0.48, v1=0.72)
    nose_hull.add_cylinder(Vec3(0.0, -0.06, 3.02), Vec3(0.0, -0.06, 3.42), 0.018, 0.005, 8, u0=0.05, v0=0.80, u1=0.45, v1=0.95)

    var canopy_glass = MeshBuilder(
        base_color=0xD8FFB83C,
        roughness=0.08,
        metallic=0.05,
        emissive_color=Vec3(1.0, 0.72, 0.25),
        emissive_strength=0.28,
        alpha_mode=2,
    )
    canopy_glass.add_ellipsoid(Vec3(0.0, 0.22, 0.88), Vec3(0.28, 0.23, 0.78), rings=12, segments=18, theta_start=0.0, theta_end=PI_F32 * 0.62)
    var nose_prims = List[NativeMesh]()
    nose_prims.append(nose_hull^.finish())
    nose_prims.append(canopy_glass^.finish())
    _ = append_node_meshes(model, "nose_radome", nose_prims, translation=Vec3(0.0, 0.02, 1.18), parent=n_torso)

    # 2. VF-1S Head Unit + Glowing Emerald Visor + 4 Swept Quad Laser Cannons
    var head_hull = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.35, metallic=0.18, texture=0)
    head_hull.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.32, 0.28, 0.36), rings=10, segments=16, u0=0.04, v0=0.04, u1=0.46, v1=0.46)
    head_hull.add_beveled_box(Vec3(0.0, 0.16, -0.04), Vec3(0.14, 0.14, 0.28), bevel=0.04, u0=0.04, v0=0.54, u1=0.46, v1=0.72)
    for side in range(2):
        var sx = Float32(-1.0) if side == 0 else Float32(1.0)
        head_hull.add_cylinder(Vec3(sx * 0.28, 0.08, 0.04), Vec3(sx * 0.56, 0.44, -0.34), 0.036, 0.018, 8, u0=0.04, v0=0.54, u1=0.46, v1=0.63)
        head_hull.add_cylinder(Vec3(sx * 0.26, 0.0, 0.06), Vec3(sx * 0.48, 0.28, -0.28), 0.030, 0.015, 8, u0=0.04, v0=0.78, u1=0.46, v1=0.96)
    var head_visor = MeshBuilder(
        base_color=0xFF38FFB4,
        roughness=0.10,
        metallic=0.10,
        emissive_color=Vec3(0.22, 1.0, 0.72),
        emissive_strength=1.95,
    )
    head_visor.add_ellipsoid(Vec3(0.0, 0.02, 0.26), Vec3(0.24, 0.10, 0.14), rings=8, segments=12)
    var head_prims = List[NativeMesh]()
    head_prims.append(head_hull^.finish())
    head_prims.append(head_visor^.finish())
    _ = append_node_meshes(model, "head_vf1s", head_prims, translation=Vec3(0.0, 0.10, 0.05), scale=Vec3(0.01, 0.01, 0.01), parent=n_torso)

    # 3 & 4. Variable-Sweep NACA Airfoil Wings (with UN Spacy Roundel UVs + Wingtip Nav Lights + Hardpoints)
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var wb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.36, metallic=0.16, texture=0)
        wb.add_airfoil(
            Vec3(0.0, 0.0, 0.55),
            1.28,
            0.11,
            Vec3(sgn * 3.25, -0.03, -0.85),
            0.56,
            0.07,
            chord_steps=8,
            span_steps=5,
            u0=0.52,
            v0=0.52,
            u1=0.98,
            v1=0.98,
        )
        # Underwing micro-missile pylon
        wb.add_beveled_box(Vec3(sgn * 1.45, -0.14, -0.25), Vec3(0.10, 0.12, 0.48), bevel=0.03, u0=0.04, v0=0.78, u1=0.44, v1=0.96)
        var nav_col = UInt32(0xFFFF3848) if side == 0 else UInt32(0xFF38FF98)
        var nav_em = Vec3(1.0, 0.2, 0.25) if side == 0 else Vec3(0.2, 1.0, 0.55)
        var nav = MeshBuilder(base_color=nav_col, roughness=0.12, metallic=0.0, emissive_color=nav_em, emissive_strength=2.2)
        nav.add_ellipsoid(Vec3(sgn * 3.28, -0.03, -1.10), Vec3(0.07, 0.05, 0.14), 6, 8)
        var w_prims = List[NativeMesh]()
        w_prims.append(wb^.finish())
        w_prims.append(nav^.finish())
        _ = append_node_meshes(
            model,
            "wing_l" if side == 0 else "wing_r",
            w_prims,
            translation=Vec3(sgn * 0.64, 0.08, -0.12),
            parent=n_torso,
        )

    # 5 & 6. Twin Angled Tailfins
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var tb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.25, texture=0)
        tb.add_airfoil(
            Vec3(0.0, 0.0, 0.35),
            0.82,
            0.09,
            Vec3(sgn * 0.38, 0.96, -0.35),
            0.42,
            0.06,
            up_dir=Vec3(1.0, 0.0, 0.0),
            chord_dir=Vec3(0.0, 0.0, -1.0),
            chord_steps=6,
            span_steps=3,
            u0=0.04,
            v0=0.53,
            u1=0.48,
            v1=0.73,
        )
        var t_prims = List[NativeMesh]()
        t_prims.append(tb^.finish())
        _ = append_node_meshes(
            model,
            "tail_l" if side == 0 else "tail_r",
            t_prims,
            translation=Vec3(sgn * 0.52, 0.22, -0.95),
            parent=n_torso,
        )

    # 7..10. Articulated Thighs & Shins / Engine Nacelles + Titanium Vector Nozzles
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var th_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.36, metallic=0.18, texture=0)
        th_b.add_beveled_box(Vec3(0.0, -0.52, 0.0), Vec3(0.24, 0.56, 0.26), bevel=0.07, u0=0.04, v0=0.04, u1=0.46, v1=0.46)
        th_b.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.22, 0.22, 0.22), 8, 12, u0=0.04, v0=0.78, u1=0.46, v1=0.96)
        var th_prims = List[NativeMesh]()
        th_prims.append(th_b^.finish())
        var n_thigh = append_node_meshes(
            model,
            "thigh_l" if side == 0 else "thigh_r",
            th_prims,
            translation=Vec3(sgn * 0.54, -0.06, -0.32),
            parent=n_torso,
        )

        var sh_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.36, metallic=0.18, texture=0)
        sh_b.add_beveled_box(Vec3(0.0, -0.68, 0.0), Vec3(0.30, 0.72, 0.32), bevel=0.08, u0=0.04, v0=0.04, u1=0.48, v1=0.72)
        # Knee & ankle armor
        sh_b.add_ellipsoid(Vec3(0.0, 0.04, 0.22), Vec3(0.22, 0.24, 0.18), 8, 12, u0=0.04, v0=0.53, u1=0.46, v1=0.72)
        var noz_metal = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.18, metallic=0.88, texture=0)
        noz_metal.add_cylinder(Vec3(0.0, -1.32, 0.0), Vec3(0.0, -1.62, 0.04), 0.25, 0.21, 16, u0=0.04, v0=0.78, u1=0.46, v1=0.98)
        noz_metal.add_beveled_box(Vec3(0.0, -1.52, 0.22), Vec3(0.25, 0.10, 0.36), bevel=0.03, u0=0.04, v0=0.78, u1=0.46, v1=0.98)
        var noz_glow = MeshBuilder(
            base_color=0xFF5CEBFF,
            roughness=0.10,
            metallic=0.0,
            emissive_color=Vec3(0.35, 0.92, 1.0),
            emissive_strength=2.3,
        )
        noz_glow.add_ellipsoid(Vec3(0.0, -1.58, 0.02), Vec3(0.17, 0.07, 0.17), 8, 12)
        var sh_prims = List[NativeMesh]()
        sh_prims.append(sh_b^.finish())
        sh_prims.append(noz_metal^.finish())
        sh_prims.append(noz_glow^.finish())
        _ = append_node_meshes(
            model,
            "shin_l" if side == 0 else "shin_r",
            sh_prims,
            translation=Vec3(0.0, -1.02, 0.0),
            parent=n_thigh,
        )

    # 11 & 12. Articulated Shoulder Pauldrons & Forearms
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var ab = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.36, metallic=0.18, texture=0)
        ab.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.28, 0.28, 0.30), bevel=0.06, u0=0.04, v0=0.53, u1=0.46, v1=0.72)
        ab.add_beveled_box(Vec3(0.0, -0.48, 0.38), Vec3(0.19, 0.24, 0.52), bevel=0.05, u0=0.04, v0=0.04, u1=0.46, v1=0.46)
        ab.add_ellipsoid(Vec3(0.0, -0.52, 0.88), Vec3(0.14, 0.14, 0.16), 8, 10, u0=0.04, v0=0.78, u1=0.46, v1=0.96)
        var a_prims = List[NativeMesh]()
        a_prims.append(ab^.finish())
        _ = append_node_meshes(
            model,
            "arm_l" if side == 0 else "arm_r",
            a_prims,
            translation=Vec3(sgn * 1.02, 0.15, 0.05),
            scale=Vec3(0.01, 0.01, 0.01),
            parent=n_torso,
        )

    # 13. GU-11 55mm 3-Barrel Rotary Gunpod
    var gp = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.22, metallic=0.82, texture=0)
    gp.add_beveled_box(Vec3(0.0, 0.0, -0.25), Vec3(0.15, 0.18, 0.72), bevel=0.04, u0=0.04, v0=0.78, u1=0.46, v1=0.98)
    gp.add_cylinder(Vec3(0.0, 0.02, 0.42), Vec3(0.0, 0.02, 1.75), 0.12, 0.10, 14, u0=0.04, v0=0.78, u1=0.46, v1=0.98)
    for b_i in range(3):
        var ang = Float32(b_i) * 2.094395
        var bx = cos(ang) * 0.055
        var by = 0.02 + sin(ang) * 0.055
        gp.add_cylinder(Vec3(bx, by, 1.70), Vec3(bx, by, 2.08), 0.026, 0.024, 8, u0=0.04, v0=0.78, u1=0.46, v1=0.98)
    var gp_prims = List[NativeMesh]()
    gp_prims.append(gp^.finish())
    _ = append_node_meshes(model, "gunpod_gu11", gp_prims, translation=Vec3(0.0, -0.38, -0.25), parent=n_torso)

    return model^


def build_regult_pod_model() -> NativeModel:
    """Builds a 5-node articulated Zentraedi Regult Tactical Battlepod."""
    var model = NativeModel()
    model.textures.append(make_regult_armor_tex())

    var hull = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.34, metallic=0.28, texture=0)
    hull.add_ellipsoid(Vec3(0.0, 0.48, 0.15), Vec3(1.22, 1.08, 1.58), rings=16, segments=24, u0=0.0, v0=0.0, u1=1.0, v1=0.65)
    # Rear rocket booster bell
    hull.add_cylinder(Vec3(0.0, 0.52, -1.25), Vec3(0.0, 0.55, -1.78), 0.72, 0.52, 16, u0=0.0, v0=0.70, u1=1.0, v1=0.98)
    # Twin dorsal anti-aircraft cannons + chin lasers
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        hull.add_beveled_box(Vec3(sgn * 0.44, 1.56, -0.10), Vec3(0.16, 0.22, 0.46), bevel=0.04, u0=0.0, v0=0.70, u1=1.0, v1=0.98)
        hull.add_cylinder(Vec3(sgn * 0.44, 1.66, 0.32), Vec3(sgn * 0.44, 1.68, 2.25), 0.075, 0.048, 10, u0=0.0, v0=0.70, u1=1.0, v1=0.98)
        hull.add_cylinder(Vec3(sgn * 0.34, -0.18, 1.35), Vec3(sgn * 0.34, -0.22, 2.05), 0.055, 0.035, 8, u0=0.0, v0=0.70, u1=1.0, v1=0.98)

    var eye_ring = MeshBuilder(base_color=0xFF222A3A, roughness=0.22, metallic=0.80)
    eye_ring.add_cylinder(Vec3(0.0, 0.44, 1.52), Vec3(0.0, 0.44, 1.76), 0.42, 0.38, 16)
    var eye_lens = MeshBuilder(
        base_color=0xFFFF2448,
        roughness=0.06,
        metallic=0.05,
        emissive_color=Vec3(1.0, 0.16, 0.32),
        emissive_strength=2.15,
    )
    eye_lens.add_ellipsoid(Vec3(0.0, 0.44, 1.72), Vec3(0.32, 0.28, 0.16), 10, 16)

    var pod_prims = List[NativeMesh]()
    pod_prims.append(hull^.finish())
    pod_prims.append(eye_ring^.finish())
    pod_prims.append(eye_lens^.finish())
    var n_pod = append_node_meshes(model, "pod_hull", pod_prims)

    # Reverse-jointed ostrich legs (nodes 1,2 = left thigh/shin; 3,4 = right thigh/shin)
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var th = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.30, metallic=0.62, texture=0)
        th.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.24, 0.24, 0.24), 8, 12, u0=0.0, v0=0.72, u1=1.0, v1=0.98)
        th.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.05, -0.92), 0.20, 0.16, 12, u0=0.0, v0=0.72, u1=1.0, v1=0.98)
        var th_prims = List[NativeMesh]()
        th_prims.append(th^.finish())
        var n_th = append_node_meshes(model, "leg_l" if side == 0 else "leg_r", th_prims, translation=Vec3(sgn * 0.92, -0.15, 0.0), parent=n_pod)

        var sh = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.30, metallic=0.62, texture=0)
        sh.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.20, 0.20, 0.20), 8, 12, u0=0.0, v0=0.72, u1=1.0, v1=0.98)
        sh.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.42, 1.35), 0.16, 0.13, 12, u0=0.0, v0=0.72, u1=1.0, v1=0.98)
        sh.add_beveled_box(Vec3(0.0, -1.50, 1.58), Vec3(0.28, 0.10, 0.48), bevel=0.03, u0=0.0, v0=0.72, u1=1.0, v1=0.98)
        var sh_prims = List[NativeMesh]()
        sh_prims.append(sh^.finish())
        _ = append_node_meshes(model, "shin_l" if side == 0 else "shin_r", sh_prims, translation=Vec3(0.0, -1.05, -0.92), parent=n_th)

    return model^


def build_glaug_officer_model() -> NativeModel:
    """Builds a 5-node articulated Zentraedi Glaug Officer's Command Pod."""
    var model = NativeModel()
    model.textures.append(make_glaug_command_tex())

    var hull = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.26, metallic=0.38, texture=0)
    hull.add_ellipsoid(Vec3(0.0, 0.44, 0.18), Vec3(1.48, 0.72, 1.92), rings=16, segments=24, u0=0.0, v0=0.0, u1=1.0, v1=0.56)
    # Dorsal heavy impact plasma cannon turret + long rail barrel
    hull.add_beveled_box(Vec3(0.0, 1.38, -0.45), Vec3(0.34, 0.36, 0.82), bevel=0.06, u0=0.0, v0=0.64, u1=1.0, v1=0.98)
    hull.add_cylinder(Vec3(0.0, 1.62, 0.25), Vec3(0.0, 1.66, 3.45), 0.17, 0.10, 14, u0=0.0, v0=0.64, u1=1.0, v1=0.98)
    # Twin particle-beam claw arms
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        hull.add_cylinder(Vec3(sgn * 1.32, 0.42, 0.15), Vec3(sgn * 2.18, 0.24, 0.68), 0.24, 0.19, 12, u0=0.0, v0=0.64, u1=1.0, v1=0.98)
        hull.add_cylinder(Vec3(sgn * 2.18, 0.24, 0.68), Vec3(sgn * 2.04, 0.14, 2.38), 0.21, 0.13, 12, u0=0.0, v0=0.04, u1=1.0, v1=0.45)

    var visor = MeshBuilder(
        base_color=0xFFFFD648,
        roughness=0.08,
        metallic=0.15,
        emissive_color=Vec3(1.0, 0.82, 0.22),
        emissive_strength=2.05,
    )
    visor.add_ellipsoid(Vec3(0.0, 0.42, 1.98), Vec3(0.46, 0.18, 0.18), 8, 14)

    var cmd_prims = List[NativeMesh]()
    cmd_prims.append(hull^.finish())
    cmd_prims.append(visor^.finish())
    var n_cmd = append_node_meshes(model, "command_hull", cmd_prims)

    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var th = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.65, texture=0)
        th.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.25, -0.95), 0.25, 0.20, 12, u0=0.0, v0=0.64, u1=1.0, v1=0.98)
        var th_prims = List[NativeMesh]()
        th_prims.append(th^.finish())
        var n_th = append_node_meshes(model, "leg_l" if side == 0 else "leg_r", th_prims, translation=Vec3(sgn * 0.98, -0.15, -0.12), parent=n_cmd)

        var sh = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.45, texture=0)
        sh.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.55, 1.45), 0.22, 0.18, 12, u0=0.0, v0=0.04, u1=1.0, v1=0.46)
        sh.add_beveled_box(Vec3(0.0, -1.64, 1.68), Vec3(0.32, 0.12, 0.54), bevel=0.04, u0=0.0, v0=0.64, u1=1.0, v1=0.98)
        var sh_prims = List[NativeMesh]()
        sh_prims.append(sh^.finish())
        _ = append_node_meshes(model, "shin_l" if side == 0 else "shin_r", sh_prims, translation=Vec3(0.0, -1.25, -0.95), parent=n_th)

    return model^


def build_sdf1_macross_model() -> NativeModel:
    """Builds the colossal SDF-1 Super Dimension Fortress battlecruiser GLB 2.0 model."""
    var model = NativeModel()
    model.textures.append(make_sdf1_cruiser_tex())

    var cruiser = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.38, metallic=0.52, texture=0)
    # Main central armored battlecruiser hull + dorsal spine + command tower
    cruiser.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(14.5, 8.8, 44.0), bevel=1.8, u0=0.0, v0=0.0, u1=1.0, v1=0.46)
    cruiser.add_beveled_box(Vec3(0.0, 10.5, -4.0), Vec3(8.2, 4.2, 22.0), bevel=1.1, u0=0.0, v0=0.0, u1=1.0, v1=0.46)
    cruiser.add_beveled_box(Vec3(0.0, 18.5, -6.0), Vec3(5.2, 7.5, 7.2), bevel=0.8, u0=0.0, v0=0.0, u1=1.0, v1=0.46)
    cruiser.add_cylinder(Vec3(-3.2, 25.5, -6.0), Vec3(-3.2, 32.5, -6.0), 0.45, 0.15, 8, u0=0.0, v0=0.0, u1=0.4, v1=0.4)
    cruiser.add_cylinder(Vec3(3.2, 25.5, -6.0), Vec3(3.2, 32.5, -6.0), 0.45, 0.15, 8, u0=0.0, v0=0.0, u1=0.4, v1=0.4)

    # Twin forward Macross Main Boom Cannons
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        cruiser.add_beveled_box(Vec3(sgn * 8.6, 3.6, 58.0), Vec3(3.8, 4.0, 28.5), bevel=0.8, u0=0.0, v0=0.0, u1=0.8, v1=0.44)
        cruiser.add_cylinder(Vec3(sgn * 8.6, 3.6, 85.0), Vec3(sgn * 8.6, 3.6, 94.0), 2.4, 1.8, 12, u0=0.0, v0=0.0, u1=0.5, v1=0.44)

    # Port CVS-101 Prometheus (Amber) & Starboard SLV-111 Daedalus (Sea-Green) Carrier Arms
    cruiser.add_beveled_box(Vec3(-23.5, -3.2, 9.0), Vec3(5.5, 4.8, 29.0), bevel=0.9, u0=0.04, v0=0.54, u1=0.46, v1=0.96)
    cruiser.add_beveled_box(Vec3(23.5, -3.2, 9.0), Vec3(5.5, 4.8, 29.0), bevel=0.9, u0=0.54, v0=0.54, u1=0.96, v1=0.96)

    # Glowing emerald Command Bridge windows & Reflex Furnace main thrusters
    var bridge_glow = MeshBuilder(
        base_color=0xFF4CFFBA,
        roughness=0.10,
        metallic=0.0,
        emissive_color=Vec3(0.30, 1.0, 0.72),
        emissive_strength=2.1,
    )
    bridge_glow.add_beveled_box(Vec3(0.0, 22.0, 1.4), Vec3(4.2, 1.4, 0.6), bevel=0.15)

    var furnace_glow = MeshBuilder(
        base_color=0xFF58EBFF,
        roughness=0.10,
        metallic=0.0,
        emissive_color=Vec3(0.35, 0.92, 1.0),
        emissive_strength=2.4,
    )
    for k in range(4):
        var fx = Float32(-9.0) + Float32(k) * 6.0
        furnace_glow.add_cylinder(Vec3(fx, -1.0, -43.5), Vec3(fx, -1.0, -47.0), 2.8, 3.4, 14)

    var sdf_prims = List[NativeMesh]()
    sdf_prims.append(cruiser^.finish())
    sdf_prims.append(bridge_glow^.finish())
    sdf_prims.append(furnace_glow^.finish())
    _ = append_node_meshes(model, "sdf1_root", sdf_prims)
    return model^


def build_orbital_earth_model(mut sky_tex: NativeTexture) -> NativeModel:
    """Curved orbital Earth globe cap + translucent cyan Rayleigh atmospheric limb."""
    var model = NativeModel()
    model.textures.append(sky_tex.copy())

    var earth_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.54, metallic=0.05, texture=0)
    earth_b.add_ellipsoid(
        Vec3(0.0, -680.0, 0.0),
        Vec3(980.0, 620.0, 980.0),
        rings=20,
        segments=36,
        theta_start=0.0,
        theta_end=PI_F32 * 0.48,
    )

    var atmo_b = MeshBuilder(
        base_color=0x2834B8FF,
        roughness=0.25,
        metallic=0.0,
        emissive_color=Vec3(0.22, 0.68, 1.0),
        emissive_strength=0.24,
        alpha_mode=2,
    )
    atmo_b.add_ellipsoid(
        Vec3(0.0, -680.0, 0.0),
        Vec3(1002.0, 636.0, 1002.0),
        rings=16,
        segments=32,
        theta_start=PI_F32 * 0.24,
        theta_end=PI_F32 * 0.49,
    )

    var prims = List[NativeMesh]()
    prims.append(earth_b^.finish())
    prims.append(atmo_b^.finish())
    _ = append_node_meshes(model, "earth_orbit", prims)
    return model^


def build_cockpit_canopy_interior() -> NativeModel:
    """Physical 3D VF-1 cockpit canopy frame, glare-shield coaming, forward nosecone & holographic HUD combiner.

    Authored around pilot eye `(0.0, 0.42, 0.65)` with 62-deg vertical FOV so the armored canopy rails,
    instrument glare-shield, forward Skull Squadron nosecone, and holographic combiner glass frame the view.
    """
    var model = NativeModel()
    var frame = MeshBuilder(base_color=0xFF1A222E, roughness=0.30, metallic=0.80)
    # Lower dashboard glare-shield coaming & side instrument consoles
    frame.add_beveled_box(Vec3(0.0, 0.02, 1.36), Vec3(0.56, 0.11, 0.34), bevel=0.03)
    frame.add_beveled_box(Vec3(-0.52, 0.04, 1.18), Vec3(0.16, 0.09, 0.42), bevel=0.02)
    frame.add_beveled_box(Vec3(0.52, 0.04, 1.18), Vec3(0.16, 0.09, 0.42), bevel=0.02)
    # Combiner projector mount brackets
    frame.add_cylinder(Vec3(-0.21, 0.12, 1.38), Vec3(-0.21, 0.24, 1.40), 0.016, 0.012, 8)
    frame.add_cylinder(Vec3(0.21, 0.12, 1.38), Vec3(0.21, 0.24, 1.40), 0.016, 0.012, 8)

    # Left & Right forward canopy arch struts + curved overhead bow + aft rails
    frame.add_cylinder(Vec3(-0.56, 0.08, 1.16), Vec3(-0.34, 0.74, 1.54), 0.026, 0.022, 10)
    frame.add_cylinder(Vec3(0.56, 0.08, 1.16), Vec3(0.34, 0.74, 1.54), 0.026, 0.022, 10)
    frame.add_cylinder(Vec3(-0.34, 0.74, 1.54), Vec3(0.0, 0.79, 1.58), 0.022, 0.022, 10)
    frame.add_cylinder(Vec3(0.0, 0.79, 1.58), Vec3(0.34, 0.74, 1.54), 0.022, 0.022, 10)
    frame.add_cylinder(Vec3(-0.62, 0.78, 0.82), Vec3(-0.34, 0.74, 1.54), 0.024, 0.022, 8)
    frame.add_cylinder(Vec3(0.62, 0.78, 0.82), Vec3(0.34, 0.74, 1.54), 0.024, 0.022, 8)

    # Emissive MFD status indicator strips on left/right glare-shield
    var console_glow = MeshBuilder(
        base_color=0xFF38FFB8,
        roughness=0.12,
        metallic=0.0,
        emissive_color=Vec3(0.22, 1.0, 0.72),
        emissive_strength=1.45,
    )
    console_glow.add_beveled_box(Vec3(-0.34, 0.135, 1.28), Vec3(0.09, 0.008, 0.04), bevel=0.003)
    console_glow.add_beveled_box(Vec3(0.34, 0.135, 1.28), Vec3(0.09, 0.008, 0.04), bevel=0.003)

    # Forward VF-1S Valkyrie Fighter nosecone & gold Skull Squadron anti-glare dorsal trim
    var nose_ext = MeshBuilder(base_color=0xFFDCE4EE, roughness=0.36, metallic=0.18)
    nose_ext.add_cylinder(Vec3(0.0, -0.04, 1.50), Vec3(0.0, 0.04, 3.85), 0.28, 0.025, 16)
    var nose_trim = MeshBuilder(base_color=0xFFFFC828, roughness=0.32, metallic=0.22)
    nose_trim.add_beveled_box(Vec3(0.0, 0.11, 2.25), Vec3(0.085, 0.015, 0.62), bevel=0.006)

    # Physical translucent emerald holographic HUD combiner glass
    var hud_glass = MeshBuilder(
        base_color=0x2238FFB8,
        roughness=0.06,
        metallic=0.08,
        emissive_color=Vec3(0.18, 1.0, 0.72),
        emissive_strength=0.32,
        alpha_mode=2,
    )
    hud_glass.add_beveled_box(Vec3(0.0, 0.36, 1.42), Vec3(0.22, 0.15, 0.006), bevel=0.002)

    var prims = List[NativeMesh]()
    prims.append(frame^.finish())
    prims.append(console_glow^.finish())
    prims.append(nose_ext^.finish())
    prims.append(nose_trim^.finish())
    prims.append(hud_glass^.finish())
    _ = append_node_meshes(model, "vf1_cockpit", prims)
    return model^


def make_fast_pack_tex() -> NativeTexture:
    """256x256 UN Spacy NP-BP-01 Super/Strike FAST Pack slate-blue titanium plating & yellow hazard trim."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var wear = _fbm(u * 14.0, v * 14.0, 411)
            # Signature Macross Super Valkyrie slate-cobalt propellant armor
            var r = Float32(66.0) + wear * 24.0
            var g = Float32(92.0) + wear * 28.0
            var b = Float32(126.0) + wear * 32.0
            if v > 0.66 and v < 0.76:
                # Skull Squadron yellow/white vernier caution band
                r = 246.0 * (0.88 + wear * 0.12)
                g = 198.0 * (0.88 + wear * 0.12)
                b = 48.0 * (0.88 + wear * 0.12)
            elif v >= 0.76:
                # Dark RO-X2A Strike Beam Cannon gunmetal
                r = 44.0 + wear * 20.0
                g = 52.0 + wear * 22.0
                b = 64.0 + wear * 24.0
            var gx = x % 32
            var gy = y % 32
            if gx == 0 or gy == 0:
                r *= 0.72; g *= 0.74; b *= 0.76
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_queadluun_tex() -> NativeTexture:
    """256x256 Miriya Parina Sterling Ace crimson & gold Queadluun-Rau biomechanical carapace."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var n = _fbm(u * 13.0, v * 13.0, 509)
            var r = Float32(204.0) + n * 42.0
            var g = Float32(26.0) + n * 20.0
            var b = Float32(58.0) + n * 24.0
            if v > 0.54 and v < 0.66:
                # Zentraedi Quadrono Ace gold filigree trim
                r = 244.0 + n * 10.0
                g = 194.0 + n * 22.0
                b = 62.0 + n * 18.0
            elif v >= 0.66:
                # Deep plum/obsidian joint & thruster armor
                r = 78.0 + n * 24.0
                g = 34.0 + n * 16.0
                b = 68.0 + n * 24.0
            var seam = abs((u * 5.0) - Float32(Int(u * 5.0)) - 0.5)
            if seam > 0.465:
                r *= 0.72; g *= 0.72; b *= 0.75
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_thuverl_salan_tex() -> NativeTexture:
    """256x256 Zentraedi Thuverl-Salan Heavy Destroyer bio-metallic emerald/violet hull plating."""
    var sz = 256
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var m = _fbm(u * 16.0, v * 10.0, 613)
            var cell = value_noise(u * 28.0, v * 18.0, 641)
            # Iconic Zentraedi capital warship dark jade-emerald & violet organic armor
            var r = Float32(48.0) + m * 34.0 + cell * 14.0
            var g = Float32(102.0) + m * 46.0 + cell * 18.0
            var b = Float32(88.0) + m * 38.0 + cell * 16.0
            if v > 0.58:
                # Violet armored dorsal blisters & weapon sponson trim
                r = Float32(96.0) + m * 36.0
                g = Float32(58.0) + m * 24.0
                b = Float32(124.0) + m * 40.0
            if (x % 24 == 0) or (y % 24 == 0):
                r *= 0.68; g *= 0.70; b *= 0.72
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def make_macross_ocean_tex() -> NativeTexture:
    """512x512 sunlit South Pacific azure ocean swells, turquoise coral lagoons & whitecap foam."""
    var sz = 512
    var tex = NativeTexture()
    tex.width = sz
    tex.height = sz
    tex.pixels = List[UInt32](length=sz * sz, fill=0xFFFFFFFF)
    for y in range(sz):
        var v = Float32(y) / Float32(sz)
        for x in range(sz):
            var u = Float32(x) / Float32(sz)
            var swell = _fbm(u * 24.0, v * 24.0, 733)
            var ripple = value_noise(u * 68.0, v * 52.0, 757)
            var r = Float32(18.0) + swell * 28.0 + ripple * 16.0
            var g = Float32(96.0) + swell * 58.0 + ripple * 24.0
            var b = Float32(178.0) + swell * 54.0 + ripple * 22.0
            # Turquoise shallow reef ring around South Ataria Island
            var dx = (u - 0.54) * 3.4
            var dy = (v - 0.58) * 3.4
            var d = sqrt(dx * dx + dy * dy)
            if d < 0.62:
                var sh = (0.62 - d) / 0.62
                r += sh * 38.0
                g += sh * 86.0
                b += sh * 42.0
                if abs(d - 0.36) < 0.035 or ripple > 0.86:
                    r = min(Float32(252.0), r + 85.0)
                    g = min(Float32(254.0), g + 75.0)
                    b = min(Float32(255.0), b + 65.0)
            elif ripple > 0.88 and swell > 0.58:
                # Open-ocean sunlit whitecaps
                r += 64.0; g += 68.0; b += 62.0
            tex.pixels[y * sz + x] = _pack_argb(255, Int(r), Int(g), Int(b))
    return tex^


def build_vf1_fast_pack_model() -> NativeModel:
    """Twin dorsal NP-BP-01 Super Booster Pods + starboard dual-barrel RO-X2A Strike Beam Cannon (`vf1s_fast_pack.glb`)."""
    var model = NativeModel()
    model.textures.append(make_fast_pack_tex())

    var pack = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.35, metallic=0.36, texture=0)
    var dark = MeshBuilder(base_color=0xFF232B38, roughness=0.40, metallic=0.72)
    var glow = MeshBuilder(
        base_color=0xFF5CF2FF,
        roughness=0.08,
        metallic=0.0,
        emissive_color=Vec3(0.32, 0.94, 1.0),
        emissive_strength=2.35,
    )

    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var px = sgn * Float32(0.46)
        # Sculpted dorsal NP-BP-01 booster pod + caution band + forward micro-missile honeycomb bay
        pack.add_beveled_box(Vec3(px, 0.42, -0.48), Vec3(0.28, 0.24, 1.12), bevel=0.07, u0=0.04, v0=0.04, u1=0.96, v1=0.62)
        pack.add_beveled_box(Vec3(px, 0.58, -0.62), Vec3(0.22, 0.10, 0.78), bevel=0.04, u0=0.05, v0=0.66, u1=0.95, v1=0.75)
        dark.add_beveled_box(Vec3(px, 0.43, 0.65), Vec3(0.22, 0.17, 0.08), bevel=0.02)
        # Main aft rocket bell + vernier core
        dark.add_cylinder(Vec3(px, 0.42, -1.56), Vec3(px, 0.42, -1.94), 0.23, 0.28, 14)
        glow.add_cylinder(Vec3(px, 0.42, -1.86), Vec3(px, 0.42, -1.96), 0.20, 0.22, 12)
        # Conformal leg propellant armor saddle plates
        pack.add_beveled_box(Vec3(sgn * 0.78, -0.04, -0.42), Vec3(0.11, 0.18, 0.74), bevel=0.04, u0=0.04, v0=0.08, u1=0.88, v1=0.58)

    # Starboard RO-X2A Dual-Barrel Strike Beam Cannon Turret (Signature DYRL / Strike Valkyrie!)
    pack.add_beveled_box(Vec3(0.46, 0.76, -0.22), Vec3(0.18, 0.13, 0.48), bevel=0.04, u0=0.06, v0=0.78, u1=0.94, v1=0.96)
    for b_idx in range(2):
        var bx = Float32(0.39) if b_idx == 0 else Float32(0.53)
        dark.add_cylinder(Vec3(bx, 0.79, 0.24), Vec3(bx, 0.79, 1.72), 0.052, 0.038, 10)
        glow.add_cylinder(Vec3(bx, 0.79, 1.68), Vec3(bx, 0.79, 1.76), 0.032, 0.036, 8)

    var prims = List[NativeMesh]()
    prims.append(pack^.finish())
    prims.append(dark^.finish())
    prims.append(glow^.finish())
    _ = append_node_meshes(model, "vf1s_fast_pack", prims)
    return model^


def build_queadluun_rau_model() -> NativeModel:
    """5-node Miriya Parina Sterling Ace Queadluun-Rau Heavy Female Power Armor (`queadluun_rau.glb`)."""
    var model = NativeModel()
    model.textures.append(make_queadluun_tex())

    var body = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.32, metallic=0.34, texture=0)
    # Organic curved crimson torso carapace + integrated helm + swept dorsal missile/thruster pods
    body.add_ellipsoid(Vec3(0.0, 1.55, 0.05), Vec3(0.95, 1.12, 1.08), rings=14, segments=20, u0=0.02, v0=0.02, u1=0.98, v1=0.52)
    body.add_ellipsoid(Vec3(0.0, 2.48, 0.22), Vec3(0.48, 0.42, 0.62), rings=10, segments=16, u0=0.10, v0=0.54, u1=0.90, v1=0.65)
    body.add_cylinder(Vec3(0.0, 2.78, 0.12), Vec3(0.0, 3.45, -0.38), 0.09, 0.02, 8, u0=0.1, v0=0.55, u1=0.9, v1=0.65)

    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        # High-mounted swept shoulder/dorsal thruster & micro-missile pods
        body.add_ellipsoid(Vec3(sgn * 1.18, 2.08, -0.28), Vec3(0.46, 0.68, 0.92), rings=10, segments=14, u0=0.04, v0=0.04, u1=0.96, v1=0.52)
        # Triple-barrel forearm pulse-laser gauntlets
        body.add_ellipsoid(Vec3(sgn * 1.24, 1.22, 0.56), Vec3(0.28, 0.26, 0.72), rings=8, segments=12, u0=0.05, v0=0.68, u1=0.95, v1=0.96)
        body.add_cylinder(Vec3(sgn * 1.24, 1.22, 1.18), Vec3(sgn * 1.24, 1.22, 1.92), 0.11, 0.07, 10, u0=0.1, v0=0.70, u1=0.9, v1=0.95)

    var visor = MeshBuilder(
        base_color=0xFF38FFB2,
        roughness=0.08,
        metallic=0.0,
        emissive_color=Vec3(0.22, 1.0, 0.70),
        emissive_strength=2.3,
    )
    visor.add_ellipsoid(Vec3(0.0, 2.46, 0.74), Vec3(0.30, 0.14, 0.18), rings=8, segments=12)
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        visor.add_cylinder(Vec3(sgn * 1.18, 2.08, -1.12), Vec3(sgn * 1.18, 2.08, -1.32), 0.24, 0.28, 10)

    var torso_prims = List[NativeMesh]()
    torso_prims.append(body^.finish())
    torso_prims.append(visor^.finish())
    var n_torso = append_node_meshes(model, "torso", torso_prims)

    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        var th = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.34, metallic=0.36, texture=0)
        th.add_ellipsoid(Vec3(0.0, -0.58, 0.16), Vec3(0.34, 0.68, 0.38), rings=8, segments=12, u0=0.06, v0=0.06, u1=0.94, v1=0.52)
        var th_prims = List[NativeMesh]()
        th_prims.append(th^.finish())
        var n_th = append_node_meshes(model, "thigh_l" if side == 0 else "thigh_r", th_prims, translation=Vec3(sgn * 0.56, 0.85, 0.0), parent=n_torso)

        var sh = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.34, metallic=0.38, texture=0)
        sh.add_ellipsoid(Vec3(0.0, -0.68, -0.18), Vec3(0.38, 0.78, 0.46), rings=10, segments=14, u0=0.06, v0=0.54, u1=0.94, v1=0.96)
        var sh_prims = List[NativeMesh]()
        sh_prims.append(sh^.finish())
        _ = append_node_meshes(model, "shin_l" if side == 0 else "shin_r", sh_prims, translation=Vec3(0.0, -1.05, 0.20), parent=n_th)

    return model^


def build_thuverl_salan_destroyer_model() -> NativeModel:
    """Colossal Zentraedi Thuverl-Salan Heavy Destroyer capital warship (`thuverl_salan.glb`)."""
    var model = NativeModel()
    model.textures.append(make_thuverl_salan_tex())

    var hull = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.38, metallic=0.46, texture=0)
    # Elongated organic bulbous Zentraedi warship hull + predatory armored prow + command dorsal crest
    hull.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(13.5, 9.2, 46.0), rings=16, segments=26, u0=0.0, v0=0.0, u1=1.0, v1=0.56)
    hull.add_ellipsoid(Vec3(0.0, 1.4, 34.0), Vec3(8.6, 5.8, 22.0), rings=12, segments=20, u0=0.0, v0=0.0, u1=1.0, v1=0.56)
    hull.add_ellipsoid(Vec3(0.0, 8.8, -6.0), Vec3(5.6, 6.4, 18.5), rings=10, segments=16, u0=0.05, v0=0.60, u1=0.95, v1=0.96)

    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        # Lateral heavy torpedo & plasma cannon sponsons
        hull.add_ellipsoid(Vec3(sgn * 13.8, -1.2, 6.0), Vec3(5.2, 4.2, 21.0), rings=10, segments=16, u0=0.05, v0=0.60, u1=0.95, v1=0.96)
        # Dorsal heavy beam cannon turrets
        hull.add_cylinder(Vec3(sgn * 5.8, 7.6, 14.0), Vec3(sgn * 5.8, 8.8, 23.5), 1.35, 0.72, 10, u0=0.1, v0=0.62, u1=0.9, v1=0.95)

    var ports = MeshBuilder(
        base_color=0xFFFF5888,
        roughness=0.14,
        metallic=0.0,
        emissive_color=Vec3(1.0, 0.28, 0.52),
        emissive_strength=1.65,
    )
    # Forward heavy converging beam maw slit & lateral torpedo apertures
    ports.add_ellipsoid(Vec3(0.0, 1.2, 54.8), Vec3(2.4, 0.85, 1.4), rings=8, segments=12)
    ports.add_cylinder(Vec3(-13.8, -1.2, 25.8), Vec3(-13.8, -1.2, 27.2), 1.15, 0.95, 10)
    ports.add_cylinder(Vec3(13.8, -1.2, 25.8), Vec3(13.8, -1.2, 27.2), 1.15, 0.95, 10)

    var drives = MeshBuilder(
        base_color=0xFF88FF9E,
        roughness=0.10,
        metallic=0.0,
        emissive_color=Vec3(0.45, 1.0, 0.62),
        emissive_strength=2.4,
    )
    for k in range(4):
        var dx = Float32(-8.4) + Float32(k) * Float32(5.6)
        drives.add_cylinder(Vec3(dx, -0.6, -43.0), Vec3(dx, -0.6, -47.5), 2.6, 3.3, 12)

    var prims = List[NativeMesh]()
    prims.append(hull^.finish())
    prims.append(ports^.finish())
    prims.append(drives^.finish())
    _ = append_node_meshes(model, "thuverl_salan_root", prims)
    return model^


def build_macross_island_ocean_model(mut ocean_tex: NativeTexture) -> NativeModel:
    """Curved sunlit South Pacific ocean horizon + sculpted 3D South Ataria (Macross) Island caldera."""
    var model = NativeModel()
    model.textures.append(ocean_tex.copy())

    var sea = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.26, metallic=0.12, texture=0)
    sea.add_ellipsoid(
        Vec3(0.0, -620.0, 0.0),
        Vec3(1080.0, 618.5, 1080.0),
        rings=20,
        segments=36,
        theta_start=0.0,
        theta_end=PI_F32 * 0.44,
    )

    # Sculpted South Ataria Island emerald volcanic domes & golden coral beach apron in forward sector
    var beach = MeshBuilder(base_color=0xFFE4CC9E, roughness=0.72, metallic=0.02)
    beach.add_ellipsoid(Vec3(42.0, -3.2, 198.0), Vec3(74.0, 5.8, 62.0), rings=12, segments=26, theta_start=0.0, theta_end=PI_F32 * 0.52)
    beach.add_ellipsoid(Vec3(-26.0, -3.0, 226.0), Vec3(46.0, 5.2, 38.0), rings=10, segments=22, theta_start=0.0, theta_end=PI_F32 * 0.52)

    var island = MeshBuilder(base_color=0xFF2B7D42, roughness=0.64, metallic=0.04)
    island.add_ellipsoid(Vec3(46.0, -1.5, 202.0), Vec3(54.0, 24.5, 44.0), rings=14, segments=24, theta_start=0.0, theta_end=PI_F32 * 0.52)
    island.add_ellipsoid(Vec3(10.0, -1.5, 214.0), Vec3(34.0, 17.0, 28.0), rings=12, segments=20, theta_start=0.0, theta_end=PI_F32 * 0.52)
    island.add_ellipsoid(Vec3(-26.0, -1.5, 228.0), Vec3(32.0, 14.0, 26.0), rings=12, segments=20, theta_start=0.0, theta_end=PI_F32 * 0.52)

    # Macross City & UN Spacy airfield towers along the bay
    var city = MeshBuilder(base_color=0xFFDCE4EE, roughness=0.42, metallic=0.25)
    for b_i in range(6):
        var bx = Float32(18.0) + Float32(b_i) * Float32(8.5)
        var bz = Float32(158.0) + Float32((b_i * 3) % 4) * Float32(6.5)
        var bh = Float32(3.5) + Float32((b_i * 5) % 4) * Float32(2.2)
        city.add_beveled_box(Vec3(bx, 1.2 + bh, bz), Vec3(2.6, bh, 2.6), bevel=0.3)

    var prims = List[NativeMesh]()
    prims.append(sea^.finish())
    prims.append(beach^.finish())
    prims.append(island^.finish())
    prims.append(city^.finish())
    _ = append_node_meshes(model, "macross_island_ocean", prims)
    return model^


def _load_or_forge_glb(path: String, var forged: NativeModel) -> NativeModel:
    var existing = read_bytes(path)
    if len(existing) > 128:
        var loaded = load_native_glb(path)
        if loaded.loaded():
            return loaded^
    _ = write_native_glb(path, forged)
    var reloaded = load_native_glb(path)
    if reloaded.loaded():
        return reloaded^
    return forged^


struct RobotechAssetPack(Movable):
    var vf1s_skull: NativeModel
    var vf1j_max: NativeModel
    var fast_pack: NativeModel
    var regult: NativeModel
    var glaug: NativeModel
    var queadluun: NativeModel
    var thuverl_salan: NativeModel
    var sdf1: NativeModel
    var earth: NativeModel
    var macross_ocean: NativeModel
    var cockpit: NativeModel
    var barrier_hex: NativeModel

    def __init__(out self):
        self.vf1s_skull = _load_or_forge_glb("assets/models/vf1s_valkyrie.glb", build_vf1_valkyrie_model(skull_livery=True))

        # Wingman Max Sterling's VF-1J shares the exact rig with Cobalt Blue livery texture
        self.vf1j_max = build_vf1_valkyrie_model(skull_livery=False)
        self.fast_pack = _load_or_forge_glb("assets/models/vf1s_fast_pack.glb", build_vf1_fast_pack_model())

        self.regult = _load_or_forge_glb("assets/models/regult_pod.glb", build_regult_pod_model())
        self.glaug = _load_or_forge_glb("assets/models/glaug_officer.glb", build_glaug_officer_model())
        self.queadluun = _load_or_forge_glb("assets/models/queadluun_rau.glb", build_queadluun_rau_model())
        self.thuverl_salan = _load_or_forge_glb("assets/models/thuverl_salan.glb", build_thuverl_salan_destroyer_model())
        self.sdf1 = _load_or_forge_glb("assets/models/sdf1_macross.glb", build_sdf1_macross_model())

        var sky_bytes = read_bytes("assets/sky/earth_orbit.png")
        var sky_tex = NativeTexture()
        if len(sky_bytes) > 64:
            var dec = decode_png(sky_bytes, 1024)
            if dec.valid:
                sky_tex.width = dec.width
                sky_tex.height = dec.height
                sky_tex.pixels = dec.pixels.copy()
        if sky_tex.width == 0:
            sky_tex = make_earth_orbit_tex()
            _ = write_png("assets/sky/earth_orbit.png", sky_tex.pixels, sky_tex.width, sky_tex.height)
        self.earth = build_orbital_earth_model(sky_tex)

        var ocean_bytes = read_bytes("assets/sky/macross_ocean.png")
        var ocean_tex = NativeTexture()
        if len(ocean_bytes) > 64:
            var dec_o = decode_png(ocean_bytes, 1024)
            if dec_o.valid:
                ocean_tex.width = dec_o.width
                ocean_tex.height = dec_o.height
                ocean_tex.pixels = dec_o.pixels.copy()
        if ocean_tex.width == 0:
            ocean_tex = make_macross_ocean_tex()
            _ = write_png("assets/sky/macross_ocean.png", ocean_tex.pixels, ocean_tex.width, ocean_tex.height)
        self.macross_ocean = build_macross_island_ocean_model(ocean_tex)

        self.cockpit = build_cockpit_canopy_interior()

        var hex_b = MeshBuilder(
            base_color=0x784CEEFF,
            roughness=0.06,
            metallic=0.20,
            emissive_color=Vec3(0.32, 0.94, 1.0),
            emissive_strength=1.85,
            alpha_mode=2,
            double_sided=True,
        )
        hex_b.add_cylinder(Vec3(-0.04, 0.0, 0.0), Vec3(0.04, 0.0, 0.0), 1.35, 1.35, segments=6)
        var hex_prims = List[NativeMesh]()
        hex_prims.append(hex_b^.finish())
        self.barrier_hex = NativeModel()
        _ = append_node_meshes(self.barrier_hex, "pinpoint_barrier", hex_prims)

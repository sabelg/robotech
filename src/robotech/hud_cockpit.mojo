"""Robotech GPU PBR Scene Compositor & SDF-1 Veritech Tactical Cockpit HUD.

Powered by Flint's hardware Vulkan / AMDGPU RDNA GFX12 compute pipeline
(`GpuSceneRenderer`, with seamless CPU PBR fallback):
  - High-detail articulated GLB 2.0 mecha, Super/Strike FAST Packs, Queadluun-Rau & Thuverl-Salan Destroyer
  - Cook-Torrance GGX microfacet specular, Fresnel canopy sheen & soft-penumbra sphere shadows
  - Atmospheric Macross Island (Pacific Ocean + volumetric cumulus cloud deck) & High Orbit SDF-1 theaters
  - Colossal twin SDF-1 Main Boom Reflex Cannon mega-particle beams (`draw_colossal_beam_3d`)
  - 48 dynamic combat point lights + AgX filmic tonemapping + 5-tap bilinear HDR bloom
  - Anti-aliased SDF TrueType (`assets/fonts/cockpit_hud.ttf`) tactical cockpit HUD
"""
from std.ffi import external_call
from std.math import sin, cos, sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.assets.model import NativeModel, NativeMesh
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.render.lighting import LightingProfile
from flint.render.simd_canvas import Camera3D, SimdCanvas
from flint.render.rasterizer import dot
from flint.render.scene_renderer import (
    GpuSceneRenderer,
    FX_SOFT,
    FX_BOLT,
    FX_SCORCH,
)
from robotech.asset_forge import RobotechAssetPack
from robotech.mecha_meshes import (
    MechaPose,
    pose_and_draw_vf1,
    draw_vf1_planar_shadow,
    pose_and_draw_regult,
    pose_and_draw_glaug,
    pose_and_draw_queadluun,
    draw_thuverl_salan_boss,
    draw_macross_environment,
)
from robotech.sim_world import (
    RobotechSimWorld,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    PILOT_SKULL_LEADER,
    PILOT_MAX_STERLING,
    PILOT_MIRIYA_ACE,
    THEATER_MACROSS_ISLAND,
    THEATER_HIGH_ORBIT,
    THEATER_BOSS_FLEET,
    SUBSYS_BRIDGE,
    ENEMY_GLAUG,
    ENEMY_QUEADLUUN,
    BOLT_UN_STRIKE_BEAM,
    BOLT_BOSS_HEAVY,
    TEAM_UN_SPACY,
    MAX_ENEMIES,
    MAX_MISSILES,
    MIS_TRAIL_LEN,
    MAX_BOLTS,
    MAX_SPARKS,
    MAX_FIREBALLS,
)

comptime CAM_CHASE: Int = 0
comptime CAM_COCKPIT: Int = 1
comptime CAM_ACTION: Int = 2


def _build_hangar_deck_model() -> NativeModel:
    """Builds the 3D UN Spacy CVS-101 Prometheus Flight Hangar Turntable Deck + Overhead Gantry Struts."""
    var model = NativeModel()
    var deck = MeshBuilder(base_color=0xFF42546E, roughness=0.48, metallic=0.30)
    deck.add_beveled_box(Vec3(0.0, -0.25, 0.0), Vec3(18.0, 0.25, 18.0), bevel=0.06)
    # Raised sunlit turntable platform on left showcase bay (`+X` when viewed from `+Z`)
    deck.add_cylinder(Vec3(2.55, -0.02, 0.0), Vec3(2.55, 0.02, 0.0), 4.6, 4.4, 28)
    # Rear hangar bulkhead pillars & overhead gantry truss
    for k in range(-2, 3):
        var gx = Float32(k) * Float32(4.2)
        deck.add_beveled_box(Vec3(gx, 3.6, -8.5), Vec3(0.32, 3.8, 0.32), bevel=0.04)
    deck.add_beveled_box(Vec3(0.0, 7.2, -8.5), Vec3(11.0, 0.35, 0.45), bevel=0.05)

    var hazard = MeshBuilder(base_color=0xFFF0BE32, roughness=0.36, metallic=0.20)
    hazard.add_beveled_box(Vec3(2.55, 0.022, 4.85), Vec3(4.4, 0.015, 0.16), bevel=0.005)
    hazard.add_beveled_box(Vec3(2.55, 0.022, -4.85), Vec3(4.4, 0.015, 0.16), bevel=0.005)

    var strip = MeshBuilder(
        base_color=0xFF4CEEFF,
        roughness=0.10,
        metallic=0.0,
        emissive_color=Vec3(0.28, 0.92, 1.0),
        emissive_strength=1.85,
    )
    for side in range(2):
        var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
        strip.add_beveled_box(Vec3(2.55 + sgn * 4.95, 0.024, 0.0), Vec3(0.09, 0.015, 5.8), bevel=0.004)
        strip.add_beveled_box(Vec3(sgn * 4.2, 6.8, -8.1), Vec3(1.4, 0.08, 0.08), bevel=0.01)

    var prims = List[NativeMesh]()
    prims.append(deck^.finish())
    prims.append(hazard^.finish())
    prims.append(strip^.finish())
    _ = append_node_meshes(model, "prometheus_hangar", prims)
    return model^


struct RobotechRenderer(Movable):
    var scene: GpuSceneRenderer
    var assets: RobotechAssetPack
    var hangar_deck: NativeModel
    var cam_mode: Int
    var show_briefing: Bool
    var bgm_on: Bool
    var replay_label: String
    var fonts_loaded: Bool

    def __init__(out self, width: Int = 1280, height: Int = 720, prefer_gpu: Bool = True):
        self.scene = GpuSceneRenderer(width, height, prefer_gpu=prefer_gpu)
        self.assets = RobotechAssetPack()
        self.hangar_deck = _build_hangar_deck_model()
        self.cam_mode = CAM_CHASE
        self.show_briefing = False
        self.bgm_on = True
        self.replay_label = String()
        self.scene.fb.min_text_em = 11.0
        self.fonts_loaded = self.scene.fb.load_fonts(
            "assets/fonts/cockpit_hud.ttf",
            "assets/fonts/title_hud.ttf",
            ui_cache_dsf="build/cockpit_hud.dsf",
            display_cache_dsf="build/title_hud.dsf",
        )

    def backend_label(self) -> String:
        var b = self.scene.active_backend_name()
        if b == "vulkan":
            return "VULKAN-GPU (PBR+GGX)"
        if b == "amdgpu":
            return "AMDGPU-ISA (PBR+GGX)"
        return "CPU-PBR"

    def _ensure_size(mut self, w: Int, h: Int):
        if w <= 0 or h <= 0 or (w == self.scene.fb.width and h == self.scene.fb.height):
            return
        var use_gpu = self.scene.using_gpu()
        self.scene = GpuSceneRenderer(w, h, prefer_gpu=use_gpu)
        self.scene.fb.min_text_em = 11.0
        self.fonts_loaded = self.scene.fb.load_fonts(
            "assets/fonts/cockpit_hud.ttf",
            "assets/fonts/title_hud.ttf",
            ui_cache_dsf="build/cockpit_hud.dsf",
            display_cache_dsf="build/title_hud.dsf",
        )

    def build_camera(self, sim: RobotechSimWorld, w: Int, h: Int) -> Camera3D:
        var fwd = sim.forward_vec()
        var right = sim.right_vec()
        var shake_off = Vec3(
            sin(sim.sim_time * Float32(43.0)) * sim.cam_shake * Float32(0.32),
            cos(sim.sim_time * Float32(51.0)) * sim.cam_shake * Float32(0.32),
            0.0,
        )
        if self.cam_mode == CAM_COCKPIT:
            var p_pose = MechaPose.from_angles(sim.pos, sim.yaw, sim.pitch, sim.roll.value * Float32(0.35), Float32(1.0))
            var eye = p_pose.world_pt(0.0, 0.42, 0.65) + shake_off
            return Camera3D(w, h, eye, eye + p_pose.fwd * Float32(45.0), 62.0)
        if self.cam_mode == CAM_ACTION:
            var eye = sim.pos + right * Float32(5.4) - fwd * Float32(10.6) + Vec3(0.0, 1.8, 0.0) + shake_off
            return Camera3D(w, h, eye, sim.pos + fwd * Float32(11.5) + Vec3(0.0, 0.3, 0.0), 54.0)
        return Camera3D(w, h, sim.cam_eye_spring.value + shake_off, sim.cam_target_spring.value, 56.0)

    def lighting_profile_for_theater(self, theater: Int) -> LightingProfile:
        var p = LightingProfile()
        if theater == THEATER_MACROSS_ISLAND:
            # Sunlit South Pacific tropical daylight + cerulean sky scatter + ocean bounce
            p.key_direction = Vec3(-0.42, -0.72, -0.55).normalized()
            p.key_color = Vec3(0.88, 0.85, 0.79)
            p.fill_direction = Vec3(0.32, 0.68, -0.65).normalized()
            p.fill_color = Vec3(0.18, 0.32, 0.48)
            p.hemisphere_sky = Vec3(0.22, 0.36, 0.54)
            p.hemisphere_ground = Vec3(0.08, 0.17, 0.25)
            p.rim_direction = Vec3(0.24, -0.28, 0.93).normalized()
            p.rim_color = Vec3(0.20, 0.36, 0.50)
            p.fog_color = Vec3(82, 154, 222)
            p.fog_start = 240.0
            p.fog_end = 640.0
            p.height_fog = 0.0
            p.exposure = 0.94
            p.saturation = 1.09
            p.filmic = 1.0
            p.cdl_slope = Vec3(1.02, 1.01, 1.03)
            p.cdl_offset = Vec3(0.0, 0.0, 0.002)
            p.cdl_power = Vec3(0.99, 0.99, 0.98)
            return p

        # High-orbit solar key light + cool cobalt Earthshine fill + anime-cinema rim
        p.key_direction = Vec3(-0.44, -0.68, -0.58).normalized()
        p.key_color = Vec3(0.92, 0.89, 0.84)
        p.fill_direction = Vec3(0.35, 0.74, -0.56).normalized()
        p.fill_color = Vec3(0.14, 0.26, 0.44)
        p.hemisphere_sky = Vec3(0.16, 0.22, 0.34)
        p.hemisphere_ground = Vec3(0.06, 0.11, 0.20)
        p.rim_direction = Vec3(0.24, -0.28, 0.93).normalized()
        p.rim_color = Vec3(0.18, 0.34, 0.48)
        p.fog_color = Vec3(5, 9, 18)
        p.fog_start = 190.0
        p.fog_end = 540.0
        p.height_fog = 0.0
        p.exposure = 0.98
        p.saturation = 1.08
        p.filmic = 1.0  # Troy Sobotka AgX highlight roll-off
        p.cdl_slope = Vec3(1.02, 1.00, 1.04)
        p.cdl_offset = Vec3(0.0, 0.0, 0.003)
        p.cdl_power = Vec3(0.98, 0.99, 0.97)
        return p

    def _register_combat_lights_and_shadows(mut self, sim: RobotechSimWorld, boost_held: Bool):
        var fwd = sim.forward_vec()
        var right = sim.right_vec()
        var shield_on = sim.mode == MODE_BATTLOID and boost_held and sim.shield > Float32(5.0)

        # 0. SDF-1 Main Boom Reflex Cannon Convergence Megablast Light!
        if sim.reflex_beam_t > Float32(0.0):
            var fade = min(Float32(1.0), sim.reflex_beam_t / Float32(0.5))
            self.scene.add_point_light(
                sim.reflex_target_pos - Vec3(0.0, 0.0, 24.0),
                Vec3(0.42, 0.94, 1.0),
                64.0,
                0.88 * fade,
                priority=100,
                stable_id=900,
            )

        # 1. Player VF-1S Afterburners, Gunpod / Strike Cannon Muzzle Flash & Pinpoint Barrier Lights
        var ab_int = (Float32(0.78) if sim.fast_packs else Float32(0.68)) if boost_held else Float32(0.48)
        self.scene.add_point_light(
            sim.pos - fwd * Float32(3.6),
            Vec3(0.28, 0.86, 1.0),
            12.0,
            ab_int,
            priority=86,
            stable_id=1,
        )
        if sim.gun_flash_t > Float32(0.0):
            self.scene.add_point_light(
                sim.pos + fwd * Float32(4.2) + right * Float32(0.9),
                Vec3(1.0, 0.88, 0.42),
                11.0,
                0.78,
                priority=96,
                stable_id=2,
            )
        if sim.strike_flash_t > Float32(0.0):
            self.scene.add_point_light(
                sim.pos + fwd * Float32(3.6) + right * Float32(0.55) + Vec3(0.0, 0.9, 0.0),
                Vec3(0.32, 0.96, 1.0),
                12.0,
                0.82,
                priority=95,
                stable_id=5,
            )
        if shield_on:
            self.scene.add_point_light(
                sim.pos - right * Float32(1.8) + fwd * Float32(1.4),
                Vec3(0.25, 0.92, 1.0),
                10.0,
                0.65,
                priority=94,
                stable_id=3,
            )

        # Wingman Skull-2 Afterburner Light
        self.scene.add_point_light(
            sim.wing_pos - fwd * Float32(3.2),
            Vec3(0.30, 0.68, 1.0),
            11.0,
            0.70,
            priority=64,
            stable_id=4,
        )

        # Thuverl-Salan Destroyer Boss Plasma Maw Light
        if sim.boss_active and sim.boss_flash_t > Float32(0.0):
            self.scene.add_point_light(
                sim.boss_pos + Vec3(0.0, 2.0, -38.0),
                Vec3(1.0, 0.28, 0.56),
                42.0,
                2.15,
                priority=97,
                stable_id=901,
            )

        # 2. Active Plasma Explosion Fireballs (Highest Priority Dynamic Lights!)
        for fi in range(MAX_FIREBALLS):
            if sim.fireballs[fi].active and sim.fireballs[fi].duration > Float32(0.0):
                var t = sim.fireballs[fi].life / sim.fireballs[fi].duration
                self.scene.add_point_light(
                    sim.fireballs[fi].pos,
                    Vec3(1.0, 0.56, 0.24),
                    24.0,
                    1.85 * t,
                    priority=98,
                    stable_id=100 + fi,
                )

        # 3. Zentraedi Plasma Muzzle Flashes
        for ei in range(MAX_ENEMIES):
            if sim.enemies[ei].active and sim.enemies[ei].firing_flash > Float32(0.0):
                self.scene.add_point_light(
                    sim.enemies[ei].pos + Vec3(0.0, 1.6, 0.0),
                    Vec3(1.0, 0.24, 0.42),
                    18.0,
                    1.35,
                    priority=82,
                    stable_id=200 + ei,
                )

        # 4. Corkscrewing Itano Circus Micro-Missile Exhaust Lights (Fill remaining budget up to 48)
        var mis_lights = 0
        for m in range(MAX_MISSILES):
            if not sim.mis_active[m]:
                continue
            if mis_lights >= 22:
                break
            var mp = Vec3(sim.mis_x.get(m), sim.mis_y.get(m), sim.mis_z.get(m))
            if (mp - sim.pos).length_squared() < Float32(36.0):
                continue
            var mcol = Vec3(1.0, 0.84, 0.48) if sim.mis_team[m] == TEAM_UN_SPACY else Vec3(1.0, 0.26, 0.46)
            self.scene.add_point_light(mp, mcol, 9.5, 0.52, priority=70, stable_id=400 + m)
            mis_lights += 1

        # 5. Analytic Soft-Penumbra Self-Shadowing Spheres (`MAX_SHADOW_SPHERES = 8`)
        if self.cam_mode != CAM_COCKPIT:
            self.scene.add_shadow_sphere(sim.pos + Vec3(0.0, 0.95 + sim.morph_gb.value * 0.65, 0.25), 1.35)
            if sim.morph_gb.value < Float32(0.75):
                self.scene.add_shadow_sphere(sim.pos + right * Float32(1.65) + Vec3(0.0, 0.25, -0.2), 0.95)
        self.scene.add_shadow_sphere(sim.wing_pos + Vec3(0.0, 0.65, 0.0), 1.30)
        if sim.boss_active:
            self.scene.add_shadow_sphere(sim.boss_pos, 14.5)
        for ei in range(MAX_ENEMIES):
            if not sim.enemies[ei].active:
                continue
            if len(self.scene.sphere_centers) >= 8:
                break
            var rad = Float32(2.1) if sim.enemies[ei].kind >= ENEMY_GLAUG else Float32(1.6)
            self.scene.add_shadow_sphere(sim.enemies[ei].pos + Vec3(0.0, 0.75, 0.0), rad)

    def _render_briefing_frame(mut self, mut canvas: SimdCanvas, sim: RobotechSimWorld):
        """Renders the 3D CVS-101 Prometheus Flight Hangar + Rotating Pilot Valkyrie + GPU Planar Deck Shadow + Tactical Briefing Overlay."""
        var w = canvas.width
        var h = canvas.height
        self._ensure_size(w, h)

        var p = LightingProfile()
        p.key_direction = Vec3(0.24, -0.88, -0.41).normalized()
        p.key_color = Vec3(1.02, 0.98, 0.92)
        p.fill_direction = Vec3(-0.36, 0.62, 0.68).normalized()
        p.fill_color = Vec3(0.26, 0.42, 0.60)
        p.hemisphere_sky = Vec3(0.28, 0.38, 0.52)
        p.hemisphere_ground = Vec3(0.12, 0.16, 0.22)
        p.rim_direction = Vec3(-0.28, -0.22, -0.93).normalized()
        p.rim_color = Vec3(0.28, 0.48, 0.66)
        p.fog_color = Vec3(12, 20, 32)
        p.fog_start = 28.0
        p.fog_end = 72.0
        p.exposure = 1.02
        p.saturation = 1.08
        p.filmic = 1.0

        var eye = Vec3(-0.15, 3.35, 10.4)
        var tgt = Vec3(0.10, 1.35, 0.0)
        self.scene.begin_frame(eye, tgt, 46.0, p, dt=1.0 / 60.0, quality=2, reduced_effects=False)
        self.scene.add_point_light(Vec3(2.55, 5.6, 2.2), Vec3(0.65, 0.94, 1.0), 15.0, 0.95, priority=95, stable_id=10)
        self.scene.add_point_light(Vec3(5.2, 2.4, 3.4), Vec3(1.0, 0.80, 0.45), 12.0, 0.62, priority=80, stable_id=11)
        self.scene.commit_lighting()

        # Hangar floor, turntable & gantry
        var deck_mat = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
        self.scene.draw_model(self.hangar_deck, deck_mat)

        # Subtle hangar atmospheric starlight beyond launch bay opening
        for i in range(28):
            var sx = Float32((i * 37) % 24) - Float32(12.0)
            var sy = Float32(1.2) + Float32((i * 19) % 10) * Float32(0.65)
            self.scene.draw_billboard(Vec3(sx, sy, -18.0), 0.14, FX_SOFT, 0xFFD8F0FF, 185, emissive=1.8)

        # Rotating 3D Squadron VF-1 Valkyrie + GPU Directional Planar Deck Shadow in Left Bay!
        var fg = sim.morph_fg.value
        var turntable_y = Float32(1.52) + fg * Float32(0.95)
        var rot_yaw = Float32(-0.62) + sim.sim_time * Float32(0.48)
        var vf_pose = MechaPose.from_angles(Vec3(2.55, turntable_y, 0.35), rot_yaw, Float32(0.03) * (Float32(1.0) - fg), Float32(0.0), Float32(1.04))
        pose_and_draw_vf1(
            self.scene,
            self.assets,
            sim.pilot_id,
            vf_pose,
            sim.morph_fg.value,
            sim.morph_gb.value,
            afterburner=0.35,
            gun_flash=False,
            shield_active=sim.mode == MODE_BATTLOID,
            fast_packs=sim.fast_packs,
            strike_flash=False,
        )
        draw_vf1_planar_shadow(
            self.scene,
            self.assets,
            sim.pilot_id,
            vf_pose,
            sim.fast_packs,
            plane_point=Vec3(0.0, 0.035, 0.0),
            plane_normal=Vec3(0.0, 1.0, 0.0),
            shadow_color=0xFF060A12,
            max_alpha=185,
        )

        self.scene.end_scene()
        self._draw_briefing_hud(sim)
        _ = self.scene.present_headless()

        if len(canvas.pixels) >= w * h and len(self.scene.fb.pixels) >= w * h:
            _ = external_call["memcpy", Int](
                Int(canvas.pixels.unsafe_ptr()),
                Int(self.scene.fb.pixels.unsafe_ptr()),
                w * h * 4,
            )
        canvas.triangles_drawn = (
            self.scene.fb.gpu.triangle_count if self.scene.using_gpu() else self.scene.fb.drawn_triangles
        )
        canvas.sprites_drawn = 28

    def render_frame(mut self, mut canvas: SimdCanvas, sim: RobotechSimWorld, boost_held: Bool = False):
        if self.show_briefing:
            self._render_briefing_frame(canvas, sim)
            return

        var w = canvas.width
        var h = canvas.height
        self._ensure_size(w, h)

        var cam = self.build_camera(sim, w, h)
        var profile = self.lighting_profile_for_theater(sim.theater)
        var to_sun = Vec3(-profile.key_direction.x, -profile.key_direction.y, -profile.key_direction.z).normalized()

        self.scene.begin_frame(cam.eye, cam.target, cam.fov_rad * Float32(57.2957795), profile, dt=1.0 / 60.0, quality=2, reduced_effects=False)
        self._register_combat_lights_and_shadows(sim, boost_held)
        self.scene.commit_lighting()

        # 1. Theater Environment + SDF-1 Cruiser / Humanoid Storm Attacker + Daedalus Attack + Reflex Cannon
        var daed_tgt = sim.boss_pos if sim.boss_active else (sim.pos + sim.forward_vec() * Float32(145.0))
        draw_macross_environment(
            self.scene,
            self.assets,
            sim.pos,
            to_sun,
            sim.theater,
            sim.sim_time,
            sim.reflex_beam_t,
            sim.reflex_target_pos,
            sim.sdf1_storm_morph.value,
            sim.daedalus_attack_t,
            daed_tgt,
        )

        # 2. Colossal Zentraedi Thuverl-Salan Heavy Destroyer Boss (Mission 3+)
        if sim.boss_active:
            var b_pose = MechaPose.from_angles(sim.boss_pos, sim.boss_yaw, Float32(-0.03), Float32(0.04), Float32(1.0))
            draw_thuverl_salan_boss(self.scene, self.assets, b_pose, sim.boss_flash_t > Float32(0.0))

        # 3. Zentraedi Regult Battlepods, Glaug Officers & Miriya's Queadluun-Rau Aces (+ Low-Alt Ocean/Island Shadows)
        var sea_plane = Vec3(0.0, 0.35, 0.0)
        for i in range(MAX_ENEMIES):
            if not sim.enemies[i].active:
                continue
            ref e = sim.enemies[i]
            var scale = Float32(1.38) if e.kind == ENEMY_QUEADLUUN else (Float32(1.55) if e.kind == ENEMY_GLAUG else Float32(1.25))
            var roll_lean = sin(e.stride_phase * Float32(0.5)) * Float32(0.22) if e.kind == ENEMY_QUEADLUUN else Float32(0.0)
            var e_pose = MechaPose.from_angles(
                e.pos,
                e.yaw,
                e.pitch * Float32(0.35),
                roll_lean,
                scale,
            )
            if e.kind == ENEMY_QUEADLUUN:
                pose_and_draw_queadluun(self.scene, self.assets, e_pose, e.stride_phase, e.firing_flash > Float32(0.0))
                if sim.theater == THEATER_MACROSS_ISLAND and e.pos.y < Float32(58.0):
                    self.scene.draw_planar_shadow_model(self.assets.queadluun, e_pose.to_mat4(), sea_plane)
            elif e.kind == ENEMY_GLAUG:
                pose_and_draw_glaug(self.scene, self.assets, e_pose, e.stride_phase, e.firing_flash > Float32(0.0))
                if sim.theater == THEATER_MACROSS_ISLAND and e.pos.y < Float32(58.0):
                    self.scene.draw_planar_shadow_model(self.assets.glaug, e_pose.to_mat4(), sea_plane)
            else:
                pose_and_draw_regult(self.scene, self.assets, e_pose, e.stride_phase, e.firing_flash > Float32(0.0))
                if sim.theater == THEATER_MACROSS_ISLAND and e.pos.y < Float32(58.0):
                    self.scene.draw_planar_shadow_model(self.assets.regult, e_pose.to_mat4(), sea_plane)

        # 4. Wingman VF-1 ("MAX STERLING - BLUE 2", or "ROY FOCKER - SKULL 1" when player flies as Max)
        var wing_pilot = PILOT_SKULL_LEADER if sim.pilot_id == PILOT_MAX_STERLING else PILOT_MAX_STERLING
        var wing_pose = MechaPose.from_angles(sim.wing_pos, sim.wing_yaw, sim.wing_pitch, Float32(-0.18), Float32(1.0))
        pose_and_draw_vf1(
            self.scene,
            self.assets,
            wing_pilot,
            wing_pose,
            sim.wing_morph_fg.value,
            sim.wing_morph_gb.value,
            afterburner=0.68,
            gun_flash=sim.wing_gun_flash > Float32(0.0),
            shield_active=False,
            fast_packs=sim.fast_packs,
            strike_flash=False,
        )
        if sim.theater == THEATER_MACROSS_ISLAND and sim.wing_pos.y < Float32(68.0):
            draw_vf1_planar_shadow(self.scene, self.assets, wing_pilot, wing_pose, sim.fast_packs, sea_plane)

        # 5. Player VF-1 Valkyrie (Active Squadron Pilot Livery) or 1st-Person Physical 3D Cockpit Canopy
        if self.cam_mode != CAM_COCKPIT:
            var p_pose = MechaPose.from_angles(sim.pos, sim.yaw, sim.pitch, sim.roll.value, Float32(1.05))
            var ab = Float32(1.0) if boost_held else Float32(0.58)
            var shield_on = sim.mode == MODE_BATTLOID and boost_held and sim.shield > Float32(5.0)
            pose_and_draw_vf1(
                self.scene,
                self.assets,
                sim.pilot_id,
                p_pose,
                sim.morph_fg.value,
                sim.morph_gb.value,
                afterburner=ab,
                gun_flash=sim.gun_flash_t > Float32(0.0),
                shield_active=shield_on,
                fast_packs=sim.fast_packs,
                strike_flash=sim.strike_flash_t > Float32(0.0),
            )
            if sim.theater == THEATER_MACROSS_ISLAND and sim.pos.y < Float32(68.0):
                draw_vf1_planar_shadow(self.scene, self.assets, sim.pilot_id, p_pose, sim.fast_packs, sea_plane)
            elif sim.skimming_hull and sim.boss_active:
                draw_vf1_planar_shadow(
                    self.scene,
                    self.assets,
                    sim.pilot_id,
                    p_pose,
                    sim.fast_packs,
                    sim.boss_pos + Vec3(0.0, 9.2, 0.0),
                    max_alpha=148,
                )
        else:
            var c_pose = MechaPose.from_angles(sim.pos, sim.yaw, sim.pitch, sim.roll.value * Float32(0.35), Float32(1.0))
            self.scene.draw_model(self.assets.cockpit, c_pose.to_mat4(), no_fog=True)
            if sim.gun_flash_t > Float32(0.0):
                var m0 = c_pose.world_pt(0.0, -0.14, 3.85)
                var m1 = c_pose.world_pt(0.0, -0.06, 11.50)
                self.scene.draw_beam_3d(m0, m1, 0.14, FX_BOLT, 0xFFFFE478, 220, emissive=2.4)

        # 6. Corkscrewing Itano Circus Volumetric `TEX_RIBBON` Smoke Trails + Warhead Plumes
        var trail_pts = List[Vec3](capacity=MIS_TRAIL_LEN)
        for m in range(MAX_MISSILES):
            if not sim.mis_active[m]:
                continue
            var cur = Vec3(sim.mis_x.get(m), sim.mis_y.get(m), sim.mis_z.get(m))
            var vel = Vec3(sim.mis_vx.get(m), sim.mis_vy.get(m), sim.mis_vz.get(m))
            var vdir = vel.normalized() if vel.length_squared() > 0.01 else Vec3(0, 0, 1)
            var is_un = sim.mis_team[m] == TEAM_UN_SPACY

            trail_pts.clear()
            var tbase = m * MIS_TRAIL_LEN
            var tlen = sim.mis_trail_len[m]
            for k in range(tlen):
                trail_pts.append(sim.mis_trail[tbase + k])
            if len(trail_pts) < 2:
                trail_pts.clear()
                trail_pts.append(cur)
                trail_pts.append(cur - vdir * Float32(5.5))

            self.scene.draw_ribbon_3d(
                trail_pts,
                head_width=0.20,
                tail_width=0.68,
                head_color=UInt32(0xFFF6FCFF) if is_un else UInt32(0xFFFF9EB8),
                tail_color=UInt32(0xFF6B88AD) if is_un else UInt32(0xFF7C2648),
                head_alpha=205,
                tail_alpha=14,
                emissive=Float32(0.55) if is_un else Float32(0.75),
            )
            self.scene.draw_thruster_plume(
                cur,
                vdir * -1.0,
                1.35,
                0.16,
                UInt32(0xFFFFFFE0),
                UInt32(0xFFFFA838) if is_un else UInt32(0xFFFF2E68),
                alpha=240,
                emissive=2.5,
            )
            self.scene.draw_billboard(cur, 0.34, FX_SOFT, UInt32(0xFFFFECA0) if is_un else UInt32(0xFFFF4272), 235, emissive=2.6)

        # 7. GU-11 Gunpod Tracers, RO-X2A Strike Beam Cannon Bolts & Zentraedi Heavy Plasma Bolts
        for b in range(MAX_BOLTS):
            if not sim.bolt_active[b]:
                continue
            var bp = Vec3(sim.bolt_x.get(b), sim.bolt_y.get(b), sim.bolt_z.get(b))
            var bv = Vec3(sim.bolt_vx.get(b), sim.bolt_vy.get(b), sim.bolt_vz.get(b)).normalized()
            var bk = sim.bolt_kind[b]
            var is_un = sim.bolt_team[b] == TEAM_UN_SPACY
            var bcol = UInt32(0xFF38EEFF) if bk == BOLT_UN_STRIKE_BEAM else (UInt32(0xFFFFE468) if is_un else (UInt32(0xFFFF4888) if bk == BOLT_BOSS_HEAVY else UInt32(0xFFFF3262)))
            var b_len = Float32(7.6) if bk == BOLT_UN_STRIKE_BEAM else (Float32(5.4) if is_un else (Float32(12.0) if bk == BOLT_BOSS_HEAVY else Float32(7.2)))
            var b_hw = Float32(0.18) if bk == BOLT_UN_STRIKE_BEAM else (Float32(0.14) if is_un else (Float32(0.52) if bk == BOLT_BOSS_HEAVY else Float32(0.24)))
            self.scene.draw_beam_3d(bp - bv * b_len, bp, b_hw, FX_BOLT, bcol, 238, emissive=2.65)
            self.scene.draw_billboard(bp, b_hw * 2.2, FX_SOFT, UInt32(0xFFFFFFFF) if is_un else UInt32(0xFFFF98B8), 220, emissive=2.85)

        # 8. Expanding Volumetric HDR Plasma Fireballs
        for fi in range(MAX_FIREBALLS):
            if not sim.fireballs[fi].active or sim.fireballs[fi].duration <= Float32(0.0):
                continue
            var fb = sim.fireballs[fi]
            var life_frac = fb.life / fb.duration
            var grow = Float32(1.0) - life_frac * life_frac * Float32(0.65)
            var rad = fb.max_radius * grow
            var smoke_a = UInt8(max(0, min(220, Int(life_frac * Float32(185.0)))))
            var core_a = UInt8(max(0, min(250, Int(life_frac * Float32(250.0)))))
            self.scene.draw_billboard(fb.pos, rad * 1.35, FX_SCORCH, 0xFF3E2630, smoke_a, emissive=0.25, rotation=life_frac * 2.2)
            self.scene.draw_billboard(fb.pos, rad, FX_SOFT, 0xFFFF5E28, core_a, emissive=2.35)
            self.scene.draw_billboard(fb.pos, rad * 0.52, FX_SOFT, 0xFFFFFAC8, core_a, emissive=3.20)

        # 9. Parallel SIMD Combat Sparks + High-Mach Velocity Motion Streaks
        var spark_drawn = 0
        for s in range(MAX_SPARKS):
            var sl = sim.spark_life.get(s)
            if sl <= Float32(0.0):
                continue
            var k = Int(sim.spark_kind[s])
            if k == 0 and (s & 3) != 0:
                continue
            var sp = Vec3(sim.spark_x.get(s), sim.spark_y.get(s), sim.spark_z.get(s))
            var scol = UInt32(0xFF5CEBFF) if k == 0 else (UInt32(0xFFFFB238) if k == 1 else UInt32(0xFFFF3868))
            var srad = Float32(0.24) if k == 0 else Float32(0.42)
            var salpha = UInt8(max(24, min(230, Int(sl * Float32(320.0)))))
            self.scene.draw_billboard(sp, srad, FX_SOFT, scol, salpha, emissive=2.2)
            spark_drawn += 1

        var speed = sim.vel.length()
        if speed > Float32(52.0):
            var vdir = sim.vel * (Float32(1.0) / speed)
            var streak_alpha = UInt8(min(130, Int((speed - Float32(52.0)) * Float32(2.2))))
            for st in range(16):
                var ang = Float32(st) * Float32(0.3927) + sim.sim_time * Float32(0.7)
                var dist_r = Float32(6.5) + Float32(st % 4) * Float32(2.8)
                var fwd_z = Float32(8.0) + Float32((st * 7) % 18)
                var anchor = (
                    self.scene.fb.camera
                    + self.scene.fb.forward * fwd_z
                    + self.scene.fb.right * (cos(ang) * dist_r)
                    + self.scene.fb.up * (sin(ang) * dist_r)
                )
                self.scene.draw_beam_3d(
                    anchor,
                    anchor - vdir * Float32(4.2),
                    0.045,
                    FX_BOLT,
                    0xFFA8E6FF,
                    streak_alpha,
                    emissive=1.4,
                )

        # 10. Finish 3D Scene, Record Crisp SDF TrueType Tactical HUD & Present GPU Frame!
        self.scene.end_scene()
        self._draw_tactical_hud(sim)
        _ = self.scene.present_headless()

        # Copy finished 32-bit pixels + telemetry counters into `SimdCanvas` for Wayland / PNG / FLIP tests
        if len(canvas.pixels) >= w * h and len(self.scene.fb.pixels) >= w * h:
            _ = external_call["memcpy", Int](
                Int(canvas.pixels.unsafe_ptr()),
                Int(self.scene.fb.pixels.unsafe_ptr()),
                w * h * 4,
            )
        canvas.triangles_drawn = (
            self.scene.fb.gpu.triangle_count if self.scene.using_gpu() else self.scene.fb.drawn_triangles
        )
        canvas.sprites_drawn = spark_drawn

    def _project_ui(self, world_pos: Vec3) -> Tuple[Int, Int, Bool]:
        """Projects a 3D world coordinate onto the 1280x720 virtual UI canvas."""
        var rel = world_pos - self.scene.fb.camera
        var depth = dot(rel, self.scene.fb.forward)
        if depth <= Float32(0.2):
            return (0, 0, False)
        var sx = Float32(self.scene.fb.width) * 0.5 + dot(rel, self.scene.fb.right) * self.scene.fb.focal / depth
        var sy = Float32(self.scene.fb.height) * 0.5 - dot(rel, self.scene.fb.up) * self.scene.fb.focal / depth
        var ux = Int((sx - Float32(self.scene.fb.ui_x)) / self.scene.fb.ui_scale + 0.5)
        var uy = Int((sy - Float32(self.scene.fb.ui_y)) / self.scene.fb.ui_scale + 0.5)
        if ux < -40 or ux > 1320 or uy < -40 or uy > 760:
            return (ux, uy, False)
        return (ux, uy, True)

    def _draw_hud_panel(mut self, x: Int, y: Int, w: Int, h: Int, accent: UInt32, alpha: UInt8 = 205):
        self.scene.fb.draw_nine_slice(0, x, y, w, h, alpha=alpha, corner_scale=0.65)
        var r = UInt8((accent >> 16) & 255)
        var g = UInt8((accent >> 8) & 255)
        var b = UInt8(accent & 255)
        self.scene.fb.draw_rectangle(x + 2, y + 2, w - 4, 2, r, g, b, 235)
        self.scene.fb.draw_rectangle(x + 2, y + h - 4, 18, 2, r, g, b, 200)
        self.scene.fb.draw_rectangle(x + w - 20, y + h - 4, 18, 2, r, g, b, 200)

    def _draw_bracket_ui(mut self, cx: Int, cy: Int, half: Int, r: UInt8, g: UInt8, b: UInt8, alpha: UInt8 = 235):
        var arm = max(5, half // 2)
        self.scene.fb.draw_rectangle(cx - half, cy - half, arm, 2, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx - half, cy - half, 2, arm, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx + half - arm, cy - half, arm, 2, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx + half - 2, cy - half, 2, arm, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx - half, cy + half - 2, arm, 2, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx - half, cy + half - arm, 2, arm, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx + half - arm, cy + half - 2, arm, 2, r, g, b, alpha)
        self.scene.fb.draw_rectangle(cx + half - 2, cy + half - arm, 2, arm, r, g, b, alpha)

    def _draw_tactical_hud(mut self, sim: RobotechSimWorld):
        """Renders crisp SDF TrueType UN Spacy / SDF-1 tactical HUD on the 1280x720 virtual canvas."""
        var cx = 640
        var cy = 360

        # A. Holographic Pitch Ladder & Velocity Vector in 1st-Person Cockpit Mode
        if self.cam_mode == CAM_COCKPIT:
            for rung in range(-2, 3):
                if rung == 0:
                    continue
                var ry = cy + rung * 46 + Int(sim.pitch * Float32(92.0))
                if ry > 110 and ry < 560:
                    self.scene.fb.draw_rectangle(cx - 76, ry, 44, 2, 56, 255, 180, 195)
                    self.scene.fb.draw_rectangle(cx + 32, ry, 44, 2, 56, 255, 180, 195)
                    self.scene.fb.draw_text(String(-rung * 10), cx + 84, ry - 6, 12, 92, 255, 196)
            self.scene.fb.draw_text("HDG " + String(Int((sim.yaw * 57.2958 + 360.0)) % 360) + " DEG", cx, 64, 14, 92, 255, 196, align=1)

        # B. 3D Projected Zentraedi Target Reticles, Health Bars & Multi-Lock Counters
        for i in range(MAX_ENEMIES):
            if not sim.enemies[i].active:
                continue
            ref e = sim.enemies[i]
            var sp = self._project_ui(e.pos)
            if not sp[2]:
                continue
            var sx = sp[0]
            var sy = sp[1]
            if sx < 32 or sx > 1248 or sy < 48 or sy > 668:
                continue
            var dist_m = Int((e.pos - sim.pos).length())
            var is_qdl = e.kind == ENEMY_QUEADLUUN
            var is_glg = e.kind == ENEMY_GLAUG
            var br = UInt8(255) if (is_qdl or is_glg) else UInt8(74)
            var bg = UInt8(78) if is_qdl else (UInt8(210) if is_glg else UInt8(224))
            var bb = UInt8(196) if is_qdl else (UInt8(56) if is_glg else UInt8(158))
            var box_r = max(13, min(36, 1650 // max(25, dist_m)))
            self._draw_bracket_ui(sx, sy, box_r, br, bg, bb)

            var bw = box_r * 2
            var fill_w = max(2, Int(Float32(bw) * (e.hp / max(Float32(1.0), e.max_hp))))
            self.scene.fb.draw_rectangle(sx - box_r, sy - box_r - 7, bw, 4, 18, 28, 42, 210)
            self.scene.fb.draw_rectangle(sx - box_r, sy - box_r - 7, fill_w, 4, br, bg, bb, 240)
            var tag = "QUEADLUUN " if is_qdl else ("GLAUG " if is_glg else "REGULT ")
            self.scene.fb.draw_text(tag + String(dist_m) + "M", sx, sy + box_r + 6, 12, br, bg, bb, align=1)

            if e.lock_marks > 0:
                self._draw_bracket_ui(sx, sy, box_r + 6, 255, 56, 88, 250)
                self.scene.fb.draw_text("LOCK X" + String(e.lock_marks), sx, sy - box_r - 22, 13, 255, 224, 102, align=1)

        # Thuverl-Salan Destroyer Boss + 4 Destructible Subsystems + Top Capital-Ship Armor Bar!
        if sim.boss_active:
            var bp_ui = self._project_ui(sim.boss_pos)
            if bp_ui[2] and bp_ui[0] > 40 and bp_ui[0] < 1240 and bp_ui[1] > 56 and bp_ui[1] < 650:
                var bdist = Int((sim.boss_pos - sim.pos).length())
                self._draw_bracket_ui(bp_ui[0], bp_ui[1], 36, 255, 84, 152, 245)
                self.scene.fb.draw_text("THUVERL-SALAN " + String(bdist) + "M", bp_ui[0], bp_ui[1] + 42, 13, 255, 120, 182, align=1)
                if sim.boss_lock_marks > 0:
                    self.scene.fb.draw_text("SALVO LOCK X" + String(sim.boss_lock_marks), bp_ui[0], bp_ui[1] - 52, 13, 255, 228, 96, align=1)

            # 3D Subsystem Weakpoint Reticles (Port/Stbd Turrets, Torpedo Bay, Command Bridge)
            for s in range(len(sim.boss_subsystems)):
                if not sim.boss_subsystems[s].active:
                    continue
                var spos = sim.boss_subsystem_world_pos(s)
                var sp_ui = self._project_ui(spos)
                if not sp_ui[2] or sp_ui[0] < 36 or sp_ui[0] > 1244 or sp_ui[1] < 54 or sp_ui[1] > 654:
                    continue
                var is_br = s == SUBSYS_BRIDGE
                var sr = UInt8(255) if is_br else UInt8(72)
                var sg = UInt8(216) if is_br else UInt8(242)
                var sb = UInt8(68) if is_br else UInt8(255)
                self._draw_bracket_ui(sp_ui[0], sp_ui[1], 14, sr, sg, sb, 245)
                var shp_w = max(2, Int(Float32(28.0) * (sim.boss_subsystems[s].hp / max(Float32(1.0), sim.boss_subsystems[s].max_hp))))
                self.scene.fb.draw_rectangle(sp_ui[0] - 14, sp_ui[1] - 20, 28, 3, 18, 28, 42, 215)
                self.scene.fb.draw_rectangle(sp_ui[0] - 14, sp_ui[1] - 20, shp_w, 3, sr, sg, sb, 245)
                self.scene.fb.draw_text(
                    sim.boss_subsystem_label(s),
                    sp_ui[0],
                    sp_ui[1] + 18,
                    11,
                    sr,
                    sg,
                    sb,
                    align=1,
                )
                if sim.boss_subsystems[s].lock_marks > 0:
                    self._draw_bracket_ui(sp_ui[0], sp_ui[1], 19, 255, 62, 92, 250)

            var bridge_down = sim.boss_bridge_destroyed()
            self._draw_hud_panel(cx - 285, 50, 570, 32, UInt32(0xFFD038) if bridge_down else UInt32(0xFF3888), 215)
            var boss_hdr = (
                "DESTROYER [2X CRIT EXPOSED!]"
                if bridge_down
                else ("DESTROYER [SUBSYS " + String(sim.intact_subsystem_count()) + "/4]")
            )
            self.scene.fb.draw_text(
                boss_hdr,
                cx - 272,
                59,
                12,
                UInt8(255),
                UInt8(226) if bridge_down else UInt8(168),
                UInt8(96) if bridge_down else UInt8(210),
            )
            var b_ratio = max(Float32(0.0), min(Float32(1.0), sim.boss_hp / max(Float32(1.0), sim.boss_max_hp)))
            self.scene.fb.draw_rectangle_rounded(cx + 12, 60, 256, 12, 0.4, 28, 18, 36, 225)
            self.scene.fb.draw_rectangle_rounded(cx + 12, 60, max(4, Int(b_ratio * Float32(256.0))), 12, 0.4, 255, 58, 132, 248)

        # Hull Trench Skimming Tactical Bonus Badge!
        if sim.skimming_hull:
            var skim_y = 86 if sim.boss_active else 52
            self._draw_hud_panel(cx - 225, skim_y, 450, 26, 0x38FFB8, 220)
            self.scene.fb.draw_text(
                "HULL TRENCH SKIMMING — 1.35X WEAPON DMG + FAST REFLEX!",
                cx,
                skim_y + 7,
                12,
                96,
                255,
                204,
                align=1,
                fit=428,
            )

        # Wingman IFF Tag
        var wp = self._project_ui(sim.wing_pos + Vec3(0.0, 2.6, 0.0))
        if wp[2] and wp[0] > 40 and wp[0] < 1220 and wp[1] > 45 and wp[1] < 660:
            var w_tag = "SKULL-1 [FOCKER]" if sim.pilot_id == PILOT_MAX_STERLING else "BLUE-2 [MAX]"
            self.scene.fb.draw_text(w_tag, wp[0], wp[1] - 12, 12, 98, 192, 255, align=1)

        # C. Center Bore-Sight Gunpod Pipper
        self.scene.fb.draw_circle(cx, cy, 3.0, 76, 232, 255, 190)
        self.scene.fb.draw_rectangle(cx - 22, cy - 1, 14, 2, 76, 232, 255, 225)
        self.scene.fb.draw_rectangle(cx + 8, cy - 1, 14, 2, 76, 232, 255, 225)
        self.scene.fb.draw_rectangle(cx - 1, cy - 22, 2, 14, 76, 232, 255, 225)
        self.scene.fb.draw_rectangle(cx - 1, cy + 8, 2, 14, 76, 232, 255, 225)

        # D. Top Tactical Telemetry & Campaign Mission Header
        self._draw_hud_panel(14, 10, 916, 36, 0xFFD038, 200)
        self.scene.fb.draw_text(
            sim.mission_title()
            + "  |  WAVE "
            + String(sim.wave)
            + "  |  "
            + sim.pilot_callsign()
            + "  |  SCORE: "
            + String(sim.score)
            + "  |  "
            + self.backend_label(),
            28,
            19,
            14,
            255,
            232,
            156,
            fit=890,
        )

        # Live `.flrp` Deterministic Flight Recorder Badge (when active)
        if self.replay_label.byte_length() > 0:
            self._draw_hud_panel(938, 10, 162, 36, 0xFF4268, 220)
            self.scene.fb.draw_text(self.replay_label, 1019, 20, 12, 255, 168, 186, align=1, fit=148)
        else:
            self._draw_hud_panel(938, 10, 162, 36, 0x4CE8FF, 190)
            self.scene.fb.draw_text("[B] BRIEF  [P] PILOT", 1019, 20, 11, 168, 236, 255, align=1, fit=148)

        if sim.comms_timer > Float32(0.0):
            self._draw_hud_panel(cx - 305, 674, 610, 34, 0x4CE8FF, 210)
            self.scene.fb.draw_text(sim.comms_banner, cx, 683, 13, 156, 242, 255, align=1, fit=584)

        if sim.incoming_missile_count > 0:
            var warn_y = 116 if (sim.boss_active and sim.skimming_hull) else (88 if sim.boss_active else 54)
            self._draw_hud_panel(cx - 240, warn_y, 480, 30, 0xFF2E4C, 225)
            self.scene.fb.draw_text(
                "WARNING: " + String(sim.incoming_missile_count) + " ZENTRAEDI MISSILES INBOUND — [Q/E] EVADE!",
                cx,
                warn_y + 8,
                13,
                255,
                156,
                172,
                align=1,
                fit=456,
            )

        # E. Top-Right 360-Degree SDF-1 Spherical Radar Scope
        var rx = 1108
        var ry = 10
        var rs = 158
        self._draw_hud_panel(rx, ry, rs, rs, 0x38FFB0, 205)
        self.scene.fb.draw_text("SDF-1 RADAR", rx + rs // 2, ry + 10, 13, 72, 255, 182, align=1)
        var rcx = rx + rs // 2
        var rcy = ry + rs // 2 + 8
        self.scene.fb.draw_circle(rcx, rcy, 54.0, 18, 48, 68, 120)
        self.scene.fb.draw_rectangle(rcx - 56, rcy, 112, 1, 36, 92, 118, 190)
        self.scene.fb.draw_rectangle(rcx, rcy - 52, 1, 104, 36, 92, 118, 190)
        self.scene.fb.draw_circle(rcx, rcy, 3.5, 255, 255, 255, 255)

        var cy_yaw = cos(-sim.yaw)
        var sy_yaw = sin(-sim.yaw)
        for i in range(MAX_ENEMIES):
            if not sim.enemies[i].active:
                continue
            var rel = sim.enemies[i].pos - sim.pos
            var lx = (rel.x * cy_yaw - rel.z * sy_yaw) * Float32(0.24)
            var lz = (rel.x * sy_yaw + rel.z * cy_yaw) * Float32(0.24)
            var px = rcx + max(-54, min(54, Int(lx)))
            var py = rcy - max(-50, min(50, Int(lz)))
            if sim.enemies[i].kind == ENEMY_QUEADLUUN:
                self.scene.fb.draw_circle(px, py, 3.6, 255, 78, 196, 250)
            elif sim.enemies[i].kind == ENEMY_GLAUG:
                self.scene.fb.draw_circle(px, py, 3.5, 255, 212, 56, 250)
            else:
                self.scene.fb.draw_circle(px, py, 2.8, 255, 64, 92, 245)

        if sim.boss_active:
            var brel = sim.boss_pos - sim.pos
            var blx = (brel.x * cy_yaw - brel.z * sy_yaw) * Float32(0.22)
            var blz = (brel.x * sy_yaw + brel.z * cy_yaw) * Float32(0.22)
            var bpx = rcx + max(-52, min(52, Int(blx)))
            var bpy = rcy - max(-48, min(48, Int(blz)))
            self.scene.fb.draw_rectangle(bpx - 4, bpy - 4, 8, 8, 255, 72, 148, 255)

        for m in range(MAX_MISSILES):
            if sim.mis_active[m] and sim.mis_team[m] != TEAM_UN_SPACY:
                var rel = Vec3(sim.mis_x.get(m), sim.mis_y.get(m), sim.mis_z.get(m)) - sim.pos
                var lx = (rel.x * cy_yaw - rel.z * sy_yaw) * Float32(0.24)
                var lz = (rel.x * sy_yaw + rel.z * cy_yaw) * Float32(0.24)
                var px = rcx + max(-54, min(54, Int(lx)))
                var py = rcy - max(-50, min(50, Int(lz)))
                self.scene.fb.draw_rectangle(px - 1, py - 1, 3, 3, 255, 120, 255, 240)

        # F. Bottom-Left VF-1 Transformation, Squadron Pilot & SDF-1 Storm Attacker Panel
        var bl_y = 584
        var mode_col = UInt32(0x4CE8FF) if sim.mode == MODE_FIGHTER else (UInt32(0xFFD038) if sim.mode == MODE_GUARDIAN else UInt32(0x38FFB0))
        self._draw_hud_panel(14, bl_y, 386, 124, mode_col, 210)
        var rig_prefix = ("SUPER " if sim.fast_packs else "") + sim.pilot_airframe() + ": "
        self.scene.fb.draw_text(
            rig_prefix + sim.mode_name(),
            28,
            bl_y + 10,
            15,
            UInt8((mode_col >> 16) & 255),
            UInt8((mode_col >> 8) & 255),
            UInt8(mode_col & 255),
            fit=358,
        )
        var sdf_tag = "STORM [H:DAEDALUS]" if sim.sdf1_storm_mode else "CRUISER [H:STORM]"
        self.scene.fb.draw_text("[1-3] MODE  [G] FAST  " + sdf_tag, 28, bl_y + 31, 11, 162, 212, 236, fit=358)

        self.scene.fb.draw_text("HULL", 28, bl_y + 50, 11, 220, 232, 246)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 51, 168, 8, 0.4, 24, 36, 52, 220)
        self.scene.fb.draw_rectangle_rounded(
            104,
            bl_y + 51,
            max(4, Int(sim.hull * Float32(1.68))),
            8,
            0.4,
            UInt8(74) if sim.hull > Float32(35.0) else UInt8(255),
            UInt8(216) if sim.hull > Float32(35.0) else UInt8(62),
            UInt8(152) if sim.hull > Float32(35.0) else UInt8(88),
            245,
        )

        self.scene.fb.draw_text("BARRIER", 28, bl_y + 67, 11, 220, 232, 246)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 68, 168, 8, 0.4, 24, 36, 52, 220)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 68, max(4, Int(sim.shield * Float32(1.68))), 8, 0.4, 76, 232, 255, 245)

        self.scene.fb.draw_text("THRUST", 28, bl_y + 84, 11, 220, 232, 246)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 85, 168, 8, 0.4, 24, 36, 52, 220)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 85, max(4, Int(sim.boost_fuel * Float32(1.68))), 8, 0.4, 255, 184, 56, 245)

        self.scene.fb.draw_text("SDF-1", 28, bl_y + 101, 11, 162, 242, 220)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 102, 168, 8, 0.4, 24, 36, 52, 220)
        self.scene.fb.draw_rectangle_rounded(104, bl_y + 102, max(4, Int(sim.sdf1_hull * Float32(1.68))), 8, 0.4, 64, 248, 188, 245)
        self.scene.fb.draw_text(String(Int(sim.vel.length())) + " M/S", 384, bl_y + 74, 18, 255, 226, 128, align=2)

        # G. Bottom-Right GU-11 / Strike Cannon, 60-Missile Itano Circus & SDF-1 Reflex Cannon Panel
        var br_x = 880
        self._draw_hud_panel(br_x, bl_y, 386, 124, 0xFF8C38, 210)
        var bgm_tag = "  [N:BGM]" if self.bgm_on else "  [N:MUTE]"
        self.scene.fb.draw_text("ORDNANCE & SDF-1" + bgm_tag, br_x + 14, bl_y + 10, 15, 255, 186, 88, fit=358)
        var gun_txt = "RELOADING..." if sim.gun_reload_t > Float32(0.0) else (String(sim.gun_ammo) + "/200")
        var cannon_tag = " + RO-X2A STRIKE" if sim.fast_packs else ""
        self.scene.fb.draw_text("GU-11" + cannon_tag + ": " + gun_txt, br_x + 14, bl_y + 32, 12, 232, 240, 250, fit=358)

        var mis_cap = sim.max_missiles()
        self.scene.fb.draw_text(
            "MISSILES [M/R]: " + String(sim.missiles_ready) + " / " + String(mis_cap) + "  (LOCKS: " + String(sim.total_locks) + ")",
            br_x + 14,
            bl_y + 50,
            12,
            156,
            242,
            255,
            fit=358,
        )
        self.scene.fb.draw_rectangle_rounded(br_x + 14, bl_y + 66, 358, 9, 0.4, 24, 36, 52, 220)
        self.scene.fb.draw_rectangle_rounded(br_x + 14, bl_y + 66, max(4, (sim.missiles_ready * 358) // max(1, mis_cap)), 9, 0.4, 255, 142, 56, 245)

        var r_ready = sim.reflex_charge >= Float32(100.0)
        var r_label = (
            "DAEDALUS PINPOINT ATTACK!!"
            if sim.daedalus_attack_t > Float32(0.0)
            else (
                "SDF-1 REFLEX CANNON: FIRING!!"
                if sim.reflex_beam_t > Float32(0.0)
                else ("SDF-1 REFLEX [F]: READY!  [F6/F7] REC" if r_ready else ("SDF-1 REFLEX [F]: " + String(Int(sim.reflex_charge)) + "%  [F6/F7] REC"))
            )
        )
        self.scene.fb.draw_text(
            r_label,
            br_x + 14,
            bl_y + 82,
            12,
            UInt8(88) if r_ready else UInt8(210),
            UInt8(255) if r_ready else UInt8(224),
            UInt8(232) if r_ready else UInt8(242),
            fit=358,
        )
        self.scene.fb.draw_rectangle_rounded(br_x + 14, bl_y + 100, 358, 9, 0.4, 24, 36, 52, 220)
        var r_fill = 358 if sim.reflex_beam_t > Float32(0.0) else max(4, Int(sim.reflex_charge * Float32(3.58)))
        self.scene.fb.draw_rectangle_rounded(
            br_x + 14,
            bl_y + 100,
            r_fill,
            9,
            0.4,
            UInt8(72) if r_ready or sim.reflex_beam_t > Float32(0.0) else UInt8(64),
            UInt8(250) if r_ready or sim.reflex_beam_t > Float32(0.0) else UInt8(188),
            UInt8(255),
            UInt8(248),
        )

    def _draw_briefing_hud(mut self, sim: RobotechSimWorld):
        """Renders the Holographic UN Spacy Tactical Mission Briefing & Squadron Pilot Select Cards."""
        # Top Command Banner
        self._draw_hud_panel(20, 14, 1240, 46, 0xFFD038, 215)
        self.scene.fb.draw_text(
            "UN SPACY TACTICAL BRIEFING  —  CVS-101 PROMETHEUS HANGAR BAY",
            38,
            27,
            17,
            255,
            228,
            118,
        )
        self.scene.fb.draw_text(
            "[SPACE / ENTER / B] SCRAMBLE LAUNCH!   |   " + self.backend_label(),
            1240,
            29,
            13,
            108,
            255,
            204,
            align=2,
        )

        # Left Turntable Caption Card (beneath rotating 3D VF-1 + Planar Deck Shadow)
        self._draw_hud_panel(24, 586, 560, 114, 0x4CE8FF, 205)
        var rig_name = ("SUPER " if sim.fast_packs else "") + sim.pilot_airframe() + "  —  " + sim.mode_name() + " CONFIG"
        self.scene.fb.draw_text(rig_name, 42, 600, 16, 116, 238, 255, fit=524)
        self.scene.fb.draw_text(
            "PILOT PERK: " + sim.pilot_perk_summary(),
            42,
            626,
            12,
            255,
            224,
            138,
            fit=524,
        )
        self.scene.fb.draw_text(
            "[1/2/3] INSPECT FIGHTER / GUARDIAN / BATTLOID   [G] TOGGLE FAST PACKS",
            42,
            650,
            11,
            178,
            214,
            242,
            fit=524,
        )
        self.scene.fb.draw_text(
            "[P] CYCLE SQUADRON PILOT   [F1] CYCLE CAMPAIGN MISSION   [N] BGM",
            42,
            672,
            11,
            152,
            242,
            208,
            fit=524,
        )

        # Right-Top Tactical Mission Intel Card
        var rx = 616
        self._draw_hud_panel(rx, 76, 644, 244, 0x38FFB8, 215)
        self.scene.fb.draw_text("ACTIVE OPERATION [F1 TO SELECT]:", rx + 20, 92, 12, 138, 255, 204)
        self.scene.fb.draw_text(sim.mission_title(), rx + 20, 114, 18, 255, 236, 148, fit=604)

        var obj1 = "PRIMARY: INTERCEPT REGULT RECON PODS & GLAUG OFFICERS OVER SOUTH ATARIA."
        var obj2 = "ENVIRONMENT: TROPICAL PACIFIC OCEAN + CUMULUS CLOUD DECK (52M ALTITUDE)."
        var obj3 = "TACTICAL TIP: LOW-ALTITUDE GUARDIAN HOVER [2] EXCELS AT ISLAND SKIMMING!"
        if sim.theater == THEATER_HIGH_ORBIT:
            obj1 = "PRIMARY: DEFEND SDF-1 AFTER PLUTO SPACE FOLD AGAINST QUADRONO ACES."
            obj2 = "ORDNANCE: NP-BP-01 SUPER/STRIKE FAST PACKS ARMED (60 MICRO-MISSILES)."
            obj3 = "TACTICAL TIP: PRESS [H] TO MORPH SDF-1 INTO HUMANOID STORM ATTACKER!"
        elif sim.theater == THEATER_BOSS_FLEET:
            obj1 = "PRIMARY: DESTROY THUVERL-SALAN HEAVY DESTROYER & ESCORT QUADRONO ACES."
            obj2 = "SUBSYSTEMS: TARGET PORT/STBD TURRETS, TORPEDO BAY & BRIDGE [2X CRIT]!"
            obj3 = "TACTICAL TIP: SKIM WITHIN 66M OF HULL FOR 1.35X DMG, OR [H] DAEDALUS RAM!"

        self.scene.fb.draw_text(obj1, rx + 20, 150, 12, 224, 240, 252, fit=604)
        self.scene.fb.draw_text(obj2, rx + 20, 174, 12, 186, 224, 248, fit=604)
        self.scene.fb.draw_text(obj3, rx + 20, 198, 12, 118, 248, 214, fit=604)

        # Mini theater selector pills
        for t_i in range(3):
            var px = rx + 20 + t_i * 204
            var sel = sim.theater == t_i
            self.scene.fb.draw_rectangle_rounded(
                px,
                232,
                192,
                34,
                0.35,
                UInt8(42) if sel else UInt8(18),
                UInt8(108) if sel else UInt8(30),
                UInt8(96) if sel else UInt8(46),
                235,
            )
            var m_short = "M1: MACROSS ISLAND" if t_i == 0 else ("M2: SPACE FOLD ORBIT" if t_i == 1 else "M3: FLEET DESTROYER")
            self.scene.fb.draw_text(
                m_short,
                px + 96,
                243,
                11,
                UInt8(255) if sel else UInt8(152),
                UInt8(242) if sel else UInt8(178),
                UInt8(136) if sel else UInt8(204),
                align=1,
                fit=180,
            )
        self.scene.fb.draw_text(
            "WAVE " + String(sim.wave) + "   |   HOSTILES ON RADAR: " + String(sim.active_hostile_count()),
            rx + 20,
            284,
            12,
            255,
            196,
            118,
        )

        # Right-Bottom Squadron Pilot Roster & Performance Bars Card
        self._draw_hud_panel(rx, 336, 644, 364, 0xFF8C38, 215)
        self.scene.fb.draw_text("VERITECH SQUADRON ROSTER [PRESS P TO SELECT]:", rx + 20, 352, 13, 255, 198, 108)

        for p_i in range(3):
            var card_y = 380 + p_i * 98
            var active = sim.pilot_id == p_i
            var bg_r = UInt8(48) if (active and p_i == 2) else (UInt8(28) if active else UInt8(14))
            var bg_g = UInt8(24) if (active and p_i == 2) else (UInt8(52) if active else UInt8(22))
            var bg_b = UInt8(42) if (active and p_i == 2) else (UInt8(78) if active else UInt8(34))
            self.scene.fb.draw_rectangle_rounded(rx + 18, card_y, 608, 86, 0.22, bg_r, bg_g, bg_b, 232)
            if active:
                self.scene.fb.draw_rectangle(rx + 18, card_y, 5, 86, 255, 214, 72, 255)

            var p_title = (
                "ROY FOCKER  —  SKULL LEADER (VF-1S)"
                if p_i == PILOT_SKULL_LEADER
                else (
                    "MAX STERLING  —  BLUE-2 ACE (VF-1J COBALT)"
                    if p_i == PILOT_MAX_STERLING
                    else "MIRIYA PARINA  —  RED-1 QUADRONO ACE (VF-1J CRIMSON)"
                )
            )
            var p_perk = (
                "+35% SDF-1 Reflex Cannon Charge  |  Balanced Command Spec"
                if p_i == PILOT_SKULL_LEADER
                else (
                    "+18% Turn Agility  |  +12% Top Mach Speed & Afterburner"
                    if p_i == PILOT_MAX_STERLING
                    else "+22% GU-11 & Strike Cannon ROF  |  +35% Multi-Lock Sweep"
                )
            )
            self.scene.fb.draw_text(
                ("> " if active else "  ") + p_title,
                rx + 32,
                card_y + 10,
                14,
                UInt8(255) if active else UInt8(192),
                UInt8(236) if active else UInt8(208),
                UInt8(138) if active else UInt8(226),
                fit=580,
            )
            self.scene.fb.draw_text(p_perk, rx + 36, card_y + 32, 11, 168, 228, 250, fit=576)

            # 3 Crisp Performance Stat Bars (AGILITY, FIREPOWER, TACTICAL)
            var s_agil = 108 if p_i == 0 else (142 if p_i == 1 else 124)
            var s_fire = 114 if p_i == 0 else (106 if p_i == 1 else 144)
            var s_tact = 144 if p_i == 0 else (112 if p_i == 1 else 136)
            self.scene.fb.draw_text("AGIL", rx + 36, card_y + 58, 10, 180, 200, 220)
            self.scene.fb.draw_rectangle_rounded(rx + 74, card_y + 60, 146, 7, 0.4, 22, 34, 48, 220)
            self.scene.fb.draw_rectangle_rounded(rx + 74, card_y + 60, s_agil, 7, 0.4, 76, 228, 255, 245)

            self.scene.fb.draw_text("ROF", rx + 234, card_y + 58, 10, 180, 200, 220)
            self.scene.fb.draw_rectangle_rounded(rx + 266, card_y + 60, 146, 7, 0.4, 22, 34, 48, 220)
            self.scene.fb.draw_rectangle_rounded(rx + 266, card_y + 60, s_fire, 7, 0.4, 255, 112, 152, 245)

            self.scene.fb.draw_text("SYNC", rx + 426, card_y + 58, 10, 180, 200, 220)
            self.scene.fb.draw_rectangle_rounded(rx + 464, card_y + 60, 146, 7, 0.4, 22, 34, 48, 220)
            self.scene.fb.draw_rectangle_rounded(rx + 464, card_y + 60, s_tact, 7, 0.4, 88, 255, 188, 245)

"""Robotech 3D Scene Compositor & SDF-1 Veritech Tactical Cockpit HUD (`flint.render.simd_canvas`)."""
from std.math import sin, cos
from flint.core.math import Vec3
from flint.render.simd_canvas import Camera3D, SimdCanvas
from robotech.mecha_meshes import (
    MechaPose,
    draw_veritech_valkyrie,
    draw_regult_battlepod,
    draw_glaug_officer,
    draw_sdf1_environment,
)
from robotech.sim_world import (
    RobotechSimWorld,
    MODE_FIGHTER,
    MODE_GUARDIAN,
    MODE_BATTLOID,
    ENEMY_GLAUG,
    TEAM_UN_SPACY,
    MAX_ENEMIES,
    MAX_MISSILES,
    MAX_BOLTS,
    MAX_SPARKS,
)

comptime CAM_CHASE: Int = 0
comptime CAM_COCKPIT: Int = 1
comptime CAM_ACTION: Int = 2


struct RobotechRenderer(Movable):
    var sprite_cyan: List[UInt32]
    var sprite_fire: List[UInt32]
    var sprite_plasma: List[UInt32]
    var cam_mode: Int

    def __init__(out self):
        self.sprite_cyan = Self._make_glow_sprite(0xFFFFFF, 0x3CE8FF)
        self.sprite_fire = Self._make_glow_sprite(0xFFF6B0, 0xFF8228)
        self.sprite_plasma = Self._make_glow_sprite(0xFFD0E8, 0xFF2858)
        self.cam_mode = CAM_CHASE

    @staticmethod
    def _make_glow_sprite(core: UInt32, outer: UInt32) -> List[UInt32]:
        var px = List[UInt32](length=64, fill=0)
        for y in range(8):
            for x in range(8):
                var d = abs(Float32(x) - Float32(3.5)) + abs(Float32(y) - Float32(3.5))
                if d < Float32(1.7):
                    px[y * 8 + x] = core | 0xFF000000
                elif d < Float32(3.5):
                    px[y * 8 + x] = outer | 0xFF000000
        return px^

    def build_camera(self, sim: RobotechSimWorld, w: Int, h: Int) -> Camera3D:
        var fwd = sim.forward_vec()
        var right = sim.right_vec()
        var shake_off = Vec3(
            sin(sim.sim_time * Float32(43.0)) * sim.cam_shake * Float32(0.35),
            cos(sim.sim_time * Float32(51.0)) * sim.cam_shake * Float32(0.35),
            0.0,
        )
        if self.cam_mode == CAM_COCKPIT:
            var eye = sim.pos + Vec3(0.0, 0.85, 0.0) + fwd * Float32(1.6) + shake_off
            return Camera3D(w, h, eye, eye + fwd * Float32(40.0), 62.0)
        if self.cam_mode == CAM_ACTION:
            var eye = sim.pos + right * Float32(11.5) - fwd * Float32(9.5) + Vec3(0.0, 4.5, 0.0) + shake_off
            return Camera3D(w, h, eye, sim.pos + fwd * Float32(22.0), 56.0)
        return Camera3D(w, h, sim.cam_eye_spring.value + shake_off, sim.cam_target_spring.value, 56.0)

    def render_frame(mut self, mut canvas: SimdCanvas, sim: RobotechSimWorld, boost_held: Bool = False):
        var w = canvas.width
        var h = canvas.height
        var cam = self.build_camera(sim, w, h)
        var sun = Vec3(0.42, 0.82, 0.38).normalized()

        # Deep Macross space backdrop gradient
        canvas.clear(0x060B16, 0x102038)
        draw_sdf1_environment(canvas, cam, sim.pos, sun)

        # 1. Render Zentraedi Battlepods & Glaug Officers
        for i in range(MAX_ENEMIES):
            if not sim.enemies[i].active:
                continue
            ref e = sim.enemies[i]
            var e_pose = MechaPose.from_angles(
                e.pos, e.yaw, e.pitch * Float32(0.35), Float32(0.0),
                Float32(1.55) if e.kind == ENEMY_GLAUG else Float32(1.25),
            )
            if e.kind == ENEMY_GLAUG:
                draw_glaug_officer(canvas, cam, e_pose, e.stride_phase, e.firing_flash > Float32(0.0), sun)
            else:
                draw_regult_battlepod(canvas, cam, e_pose, e.stride_phase, e.firing_flash > Float32(0.0), sun)

        # 2. Render Wingman ("MAX STERLING - SKULL 2" in iconic Cobalt Blue VF-1J livery)
        var wing_pose = MechaPose.from_angles(sim.wing_pos, sim.wing_yaw, sim.wing_pitch, Float32(-0.15), Float32(1.0))
        draw_veritech_valkyrie(
            canvas, cam, wing_pose,
            sim.wing_morph_fg.value, sim.wing_morph_gb.value,
            0x3898FF, 0xEEF3FA,
            Float32(0.65), sim.wing_gun_flash > Float32(0.0), False, sun,
        )

        # 3. Render Player VF-1S Valkyrie ("SKULL LEADER") unless inside 1st-person Cockpit
        if self.cam_mode != CAM_COCKPIT:
            var p_pose = MechaPose.from_angles(sim.pos, sim.yaw, sim.pitch, sim.roll.value, Float32(1.05))
            var ab = Float32(1.0) if boost_held else Float32(0.55)
            var shield_on = sim.mode == MODE_BATTLOID and boost_held and sim.shield > Float32(5.0)
            draw_veritech_valkyrie(
                canvas, cam, p_pose,
                sim.morph_fg.value, sim.morph_gb.value,
                0xFFD038, 0xE83A3A,
                ab, sim.gun_flash_t > Float32(0.0), shield_on, sun,
            )

        # 4. Render Corkscrewing Itano Circus Micro-Missiles & Contrail Ribbons
        for m in range(MAX_MISSILES):
            if not sim.mis_active[m]:
                continue
            var cur = Vec3(sim.mis_x.get(m), sim.mis_y.get(m), sim.mis_z.get(m))
            var prv = Vec3(sim.mis_px.get(m), sim.mis_py.get(m), sim.mis_pz.get(m))
            var v = Vec3(sim.mis_vx.get(m), sim.mis_vy.get(m), sim.mis_vz.get(m)).normalized()
            var tail = cur - v * Float32(5.8)
            var trail_col = UInt32(0xD8F6FF) if sim.mis_team[m] == TEAM_UN_SPACY else UInt32(0xFF5278)
            var head_col = UInt32(0xFFF078) if sim.mis_team[m] == TEAM_UN_SPACY else UInt32(0xFF2848)
            canvas.draw_line_3d(cam, tail, prv, trail_col, 1)
            canvas.draw_line_3d(cam, prv, cur, head_col, 2)

        # 5. Render GU-11 Gunpod Tracers & Zentraedi Plasma Bolts
        for b in range(MAX_BOLTS):
            if not sim.bolt_active[b]:
                continue
            var bp = Vec3(sim.bolt_x.get(b), sim.bolt_y.get(b), sim.bolt_z.get(b))
            var bv = Vec3(sim.bolt_vx.get(b), sim.bolt_vy.get(b), sim.bolt_vz.get(b)).normalized()
            var col = UInt32(0xFFE468) if sim.bolt_team[b] == TEAM_UN_SPACY else UInt32(0xFF3258)
            canvas.draw_line_3d(cam, bp - bv * Float32(4.2), bp, col, 2)

        # 6. Composite 2,048 Depth-Tested 8-Wide SIMD Sparks
        var cyan_addr = Int(self.sprite_cyan.unsafe_ptr())
        var fire_addr = Int(self.sprite_fire.unsafe_ptr())
        var plas_addr = Int(self.sprite_plasma.unsafe_ptr())
        for s in range(MAX_SPARKS):
            if sim.spark_life.get(s) <= Float32(0.0):
                continue
            var proj = cam.project(Vec3(sim.spark_x.get(s), sim.spark_y.get(s), sim.spark_z.get(s)))
            if proj[3]:
                var k = sim.spark_kind[s]
                var addr = cyan_addr if k == 0 else (fire_addr if k == 1 else plas_addr)
                canvas.blit_sprite_simd(addr, 8, 8, Int(proj[0]) - 4, Int(proj[1]) - 4, proj[2])

        # 7. Overlay 3D Target Brackets, Multi-Lock Diamonds & SDF-1 Cockpit HUD
        self._draw_tactical_hud(canvas, cam, sim)

    def _draw_bracket(self, mut canvas: SimdCanvas, cx: Int, cy: Int, half: Int, col: UInt32):
        var arm = max(4, half // 2)
        canvas.draw_rect(cx - half, cy - half, arm, 2, col)
        canvas.draw_rect(cx - half, cy - half, 2, arm, col)
        canvas.draw_rect(cx + half - arm, cy - half, arm, 2, col)
        canvas.draw_rect(cx + half - 2, cy - half, 2, arm, col)
        canvas.draw_rect(cx - half, cy + half - 2, arm, 2, col)
        canvas.draw_rect(cx - half, cy + half - arm, 2, arm, col)
        canvas.draw_rect(cx + half - arm, cy + half - 2, arm, 2, col)
        canvas.draw_rect(cx + half - 2, cy + half - arm, 2, arm, col)

    def _draw_tactical_hud(self, mut canvas: SimdCanvas, cam: Camera3D, sim: RobotechSimWorld):
        var w = canvas.width
        var h = canvas.height
        var cx = w // 2
        var cy = h // 2

        # A. Cockpit Canopy Rails & Holographic Pitch Ladder in 1st-Person Cockpit Mode
        if self.cam_mode == CAM_COCKPIT:
            # Canopy bow frame
            canvas.draw_rect(0, 0, w, 28, 0x0C121C)
            canvas.draw_rect(0, h - 36, w, 36, 0x0C121C)
            canvas.draw_rect(cx - 210, 28, 4, h - 64, 0x1E3248)
            canvas.draw_rect(cx + 206, 28, 4, h - 64, 0x1E3248)
            for rung in range(-2, 3):
                if rung == 0:
                    continue
                var ry = cy + rung * 46 + Int(sim.pitch * Float32(90.0))
                if ry > 90 and ry < h - 120:
                    canvas.draw_rect(cx - 70, ry, 42, 2, 0x38FFB0)
                    canvas.draw_rect(cx + 28, ry, 42, 2, 0x38FFB0)

        # B. 3D Projected Zentraedi Target Reticles, Range Readouts & Multi-Lock Counters
        for i in range(MAX_ENEMIES):
            if not sim.enemies[i].active:
                continue
            ref e = sim.enemies[i]
            var sp = cam.project(e.pos)
            if not sp[3]:
                continue
            var sx = Int(sp[0])
            var sy = Int(sp[1])
            if sx < 24 or sx >= w - 24 or sy < 24 or sy >= h - 24:
                continue
            var dist_m = Int((e.pos - sim.pos).length())
            var bcol = UInt32(0xFFD038) if e.kind == ENEMY_GLAUG else UInt32(0x4AD295)
            var box_r = max(12, min(36, 1600 // max(25, dist_m)))
            self._draw_bracket(canvas, sx, sy, box_r, bcol)

            # Enemy Hull Bar
            var bw = box_r * 2
            var fill_w = max(2, Int(Float32(bw) * (e.hp / max(Float32(1.0), e.max_hp))))
            canvas.draw_rect(sx - box_r, sy - box_r - 7, bw, 4, 0x182230)
            canvas.draw_rect(sx - box_r, sy - box_r - 7, fill_w, 4, UInt32(0xFF3E58) if e.kind == ENEMY_GLAUG else UInt32(0x4AD295))

            var tag = "GLAUG " if e.kind == ENEMY_GLAUG else "REGULT "
            canvas.draw_text(sx - box_r, sy + box_r + 5, tag + String(dist_m) + "M", bcol, 1)

            if e.lock_marks > 0:
                self._draw_bracket(canvas, sx, sy, box_r + 6, 0xFF3858)
                canvas.draw_text(sx - 24, sy - box_r - 19, "LOCK X" + String(e.lock_marks), 0xFFE066, 1)

        # Wingman HUD tag
        var wp = cam.project(sim.wing_pos + Vec3(0.0, 2.5, 0.0))
        if wp[3]:
            var wx = Int(wp[0])
            var wy = Int(wp[1])
            if wx > 30 and wx < w - 90 and wy > 30 and wy < h - 30:
                canvas.draw_text(wx - 28, wy - 12, "SKULL-2 [MAX]", 0x58B4FF, 1)

        # C. Center Bore-Sight Crosshair
        canvas.draw_rect(cx - 18, cy, 12, 2, 0x4CE8FF)
        canvas.draw_rect(cx + 6, cy, 12, 2, 0x4CE8FF)
        canvas.draw_rect(cx, cy - 18, 2, 12, 0x4CE8FF)
        canvas.draw_rect(cx, cy + 6, 2, 12, 0x4CE8FF)

        # D. Top Tactical Comms & Score Header
        canvas.draw_panel(14, 10, 860, 32, 0x0C1624, 0xFFD038)
        canvas.draw_text(
            24, 18,
            "ROBOTECH SDF-1 SIM  |  WAVE "
            + String(sim.wave)
            + "  |  HOSTILES: "
            + String(sim.active_enemy_count())
            + "  |  KILLS: "
            + String(sim.kills)
            + "  |  SCORE: "
            + String(sim.score),
            0xFFE699, 2,
        )

        if sim.comms_timer > Float32(0.0):
            canvas.draw_panel(cx - 245, h - 42, 490, 28, 0x102034, 0x4CE8FF)
            canvas.draw_text(cx - 233, h - 32, sim.comms_banner, 0x9CF0FF, 1)

        if sim.incoming_missile_count > 0:
            canvas.draw_panel(cx - 210, 82, 420, 26, 0x380E18, 0xFF2E4C)
            canvas.draw_text(
                cx - 196, 90,
                "WARNING: " + String(sim.incoming_missile_count) + " ZENTRAEDI MISSILES INBOUND — [Q/E] EVADE!",
                0xFF98A8, 1,
            )

        # E. Top-Right 360° SDF-1 Spherical Radar Scope
        var rx = w - 166
        var ry = 10
        var rs = 152
        canvas.draw_panel(rx, ry, rs, rs, 0x0A1522, 0x38FFB0)
        canvas.draw_text(rx + 10, ry + 8, "SDF-1 RADAR", 0x38FFB0, 1)
        var rcx = rx + rs // 2
        var rcy = ry + rs // 2 + 6
        canvas.draw_rect(rcx - 58, rcy, 116, 1, 0x1C384E)
        canvas.draw_rect(rcx, rcy - 54, 1, 108, 0x1C384E)
        canvas.draw_rect(rcx - 2, rcy - 2, 5, 5, 0xFFFFFF)  # Skull Leader center

        var cy_yaw = cos(-sim.yaw)
        var sy_yaw = sin(-sim.yaw)
        for i in range(MAX_ENEMIES):
            if not sim.enemies[i].active:
                continue
            var rel = sim.enemies[i].pos - sim.pos
            var lx = (rel.x * cy_yaw - rel.z * sy_yaw) * Float32(0.24)
            var lz = (rel.x * sy_yaw + rel.z * cy_yaw) * Float32(0.24)
            var px = rcx + max(-58, min(58, Int(lx)))
            var py = rcy - max(-52, min(52, Int(lz)))
            var dcol = UInt32(0xFFD038) if sim.enemies[i].kind == ENEMY_GLAUG else UInt32(0xFF3E58)
            canvas.draw_rect(px - 2, py - 2, 4, 4, dcol)

        for m in range(MAX_MISSILES):
            if sim.mis_active[m] and sim.mis_team[m] != TEAM_UN_SPACY:
                var rel = Vec3(sim.mis_x.get(m), sim.mis_y.get(m), sim.mis_z.get(m)) - sim.pos
                var lx = (rel.x * cy_yaw - rel.z * sy_yaw) * Float32(0.24)
                var lz = (rel.x * sy_yaw + rel.z * cy_yaw) * Float32(0.24)
                var px = rcx + max(-58, min(58, Int(lx)))
                var py = rcy - max(-52, min(52, Int(lz)))
                canvas.draw_rect(px - 1, py - 1, 3, 3, 0xFF78FF)

        # F. Bottom-Left VF-1 Transformation & Systems Panel
        var bl_y = h - 118
        var mode_col = UInt32(0x4CE8FF) if sim.mode == MODE_FIGHTER else (UInt32(0xFFD038) if sim.mode == MODE_GUARDIAN else UInt32(0x38FFB0))
        canvas.draw_panel(14, bl_y, 376, 106, 0x0C1624, mode_col)
        canvas.draw_text(26, bl_y + 10, "VF-1S MODE: " + sim.mode_name(), mode_col, 2)
        canvas.draw_text(26, bl_y + 32, "[1] FIGHTER   [2] GUARDIAN   [3] BATTLOID   [TAB] CYCLE", 0x9AB4CE, 1)

        # Hull, Pinpoint Shield & Boost Bars
        canvas.draw_text(26, bl_y + 50, "HULL", 0xDCE6F2, 1)
        canvas.draw_rect(96, bl_y + 50, 160, 8, 0x1C2838)
        canvas.draw_rect(96, bl_y + 50, max(2, Int(sim.hull * Float32(1.6))), 8, UInt32(0x4AD295) if sim.hull > Float32(35.0) else UInt32(0xFF3E58))

        canvas.draw_text(26, bl_y + 66, "BARRIER", 0xDCE6F2, 1)
        canvas.draw_rect(96, bl_y + 66, 160, 8, 0x1C2838)
        canvas.draw_rect(96, bl_y + 66, max(2, Int(sim.shield * Float32(1.6))), 8, 0x4CE8FF)

        canvas.draw_text(26, bl_y + 82, "THRUST", 0xDCE6F2, 1)
        canvas.draw_rect(96, bl_y + 82, 160, 8, 0x1C2838)
        canvas.draw_rect(96, bl_y + 82, max(2, Int(sim.boost_fuel * Float32(1.6))), 8, 0xFFB838)
        canvas.draw_text(270, bl_y + 66, String(Int(sim.vel.length())) + " M/S", 0xFFE082, 2)

        # G. Bottom-Right Ordnance & Itano Circus Salvo Panel
        var br_x = w - 384
        canvas.draw_panel(br_x, bl_y, 370, 106, 0x0C1624, 0xFF8C38)
        canvas.draw_text(br_x + 12, bl_y + 10, "WEAPONS & ITANO CIRCUS", 0xFFB858, 2)
        var gun_txt = "RELOADING DRUM..." if sim.gun_reload_t > Float32(0.0) else (String(sim.gun_ammo) + " / 200 RDS")
        canvas.draw_text(br_x + 12, bl_y + 34, "GU-11 GUNPOD [J / LMB]: " + gun_txt, 0xE8F0F8, 1)
        canvas.draw_text(
            br_x + 12, bl_y + 52,
            "MICRO-MISSILES [M HOLD / R SALVO]: " + String(sim.missiles_ready) + " / 36",
            0x9CF0FF, 1,
        )
        canvas.draw_rect(br_x + 12, bl_y + 68, 344, 10, 0x1C2838)
        canvas.draw_rect(br_x + 12, bl_y + 68, (sim.missiles_ready * 344) // 36, 10, 0xFF8C38)
        canvas.draw_text(
            br_x + 12, bl_y + 86,
            "ACTIVE LOCKS: " + String(sim.total_locks) + "   |   [V] CAM   [Q/E] ROLL",
            0xFFD038, 1,
        )

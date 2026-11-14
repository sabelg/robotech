"""Robotech: SDF-1 Veritech 3D Combat & Transformation Simulation (`flint.*`).

Powered by:
  - `flint.core.ecs`: `ArchetypeRegistry`, `ArchetypeSlots`, 8-wide `SimdColumn`, `run_parallel_ctx`
  - `flint.core.jobs`: Multi-threaded `JobPool` stepping 2,048 contrail/explosion/thruster sparks
  - `flint.core.springs`: Critically damped `SpringFloat` & `SpringVec3` VF-1 transformation & chase cam
  - `flint.physics.broadphase`: Spatial-hash `BroadphaseInput` + `find_pairs` collision detection
"""
from std.math import sin, cos, sqrt, atan2
from flint.core.math import Vec3
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import ArchetypeRegistry, ArchetypeSlots, SimdColumn, run_parallel_ctx
from flint.core.springs import SpringFloat, SpringVec3
from flint.physics.filter import CollisionFilter
from flint.physics.broadphase import BroadphaseInput, find_pairs

comptime MODE_FIGHTER: Int = 0
comptime MODE_GUARDIAN: Int = 1
comptime MODE_BATTLOID: Int = 2

comptime ENEMY_REGULT: Int = 0
comptime ENEMY_GLAUG: Int = 1

comptime TEAM_UN_SPACY: Int = 0
comptime TEAM_ZENTRAEDI: Int = 1

comptime MAX_ENEMIES: Int = 24
comptime MAX_MISSILES: Int = 256
comptime MIS_TRAIL_LEN: Int = 10
comptime MAX_BOLTS: Int = 256
comptime MAX_SPARKS: Int = 2048
comptime SPARK_CHUNK: Int = 256
comptime MAX_FIREBALLS: Int = 24

# Sound event flags bitmask emitted per tick so headless tests & live audio decouple cleanly
comptime EVT_TRANSFORM: Int = 1 << 0
comptime EVT_GUNPOD: Int = 1 << 1
comptime EVT_LOCK: Int = 1 << 2
comptime EVT_SALVO: Int = 1 << 3
comptime EVT_EXPLODE: Int = 1 << 4


@fieldwise_init
struct PilotControls(Copyable, ImplicitlyCopyable, Movable):
    var pitch_in: Float32       # -1..+1
    var yaw_in: Float32         # -1..+1
    var strafe_x: Float32       # -1..+1
    var lift_y: Float32         # -1..+1
    var throttle_in: Float32    # -1..+1
    var boost_held: Bool
    var fire_gunpod: Bool
    var lock_held: Bool
    var launch_salvo: Bool
    var quick_salvo: Bool
    var barrel_roll_dir: Int    # -1, 0, +1
    var request_mode: Int       # -1 = none, 0..2 = mode

    def __init__(out self):
        self.pitch_in = 0.0
        self.yaw_in = 0.0
        self.strafe_x = 0.0
        self.lift_y = 0.0
        self.throttle_in = 0.0
        self.boost_held = False
        self.fire_gunpod = False
        self.lock_held = False
        self.launch_salvo = False
        self.quick_salvo = False
        self.barrel_roll_dir = 0
        self.request_mode = -1


@fieldwise_init
struct ZentraediUnit(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var kind: Int
    var pos: Vec3
    var vel: Vec3
    var yaw: Float32
    var pitch: Float32
    var hp: Float32
    var max_hp: Float32
    var fire_cd: Float32
    var missile_cd: Float32
    var stride_phase: Float32
    var lock_marks: Int
    var firing_flash: Float32

    def __init__(out self):
        self.active = False
        self.kind = ENEMY_REGULT
        self.pos = Vec3()
        self.vel = Vec3()
        self.yaw = 0.0
        self.pitch = 0.0
        self.hp = 100.0
        self.max_hp = 100.0
        self.fire_cd = 1.0
        self.missile_cd = 4.5
        self.stride_phase = 0.0
        self.lock_marks = 0
        self.firing_flash = 0.0


@fieldwise_init
struct CombatFireball(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var pos: Vec3
    var max_radius: Float32
    var life: Float32
    var duration: Float32

    def __init__(out self):
        self.active = False
        self.pos = Vec3()
        self.max_radius = 4.5
        self.life = 0.0
        self.duration = 0.65


@fieldwise_init
struct SparkJobCtx(Copyable, ImplicitlyCopyable, Movable):
    var px: Int
    var py: Int
    var pz: Int
    var vx: Int
    var vy: Int
    var vz: Int
    var life: Int
    var dt: Float32


def step_spark_pool_chunk(ctx_addr: Int, chunk_idx: Int):
    var ctx = Pointer[SparkJobCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var start = chunk_idx * SPARK_CHUNK
    var end = min(MAX_SPARKS, start + SPARK_CHUNK)
    var px = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.px)
    var py = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.py)
    var pz = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.pz)
    var vx = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.vx)
    var vy = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.vy)
    var vz = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.vz)
    var lf = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.life)
    var drag = max(Float32(0.0), Float32(1.0) - ctx.dt * Float32(1.6))

    for i in range(start, end):
        var rem = lf[unsafe_offset=i]
        if rem > Float32(0.0):
            rem -= ctx.dt
            lf[unsafe_offset=i] = max(Float32(0.0), rem)
            px[unsafe_offset=i] = px[unsafe_offset=i] + vx[unsafe_offset=i] * ctx.dt
            py[unsafe_offset=i] = py[unsafe_offset=i] + vy[unsafe_offset=i] * ctx.dt
            pz[unsafe_offset=i] = pz[unsafe_offset=i] + vz[unsafe_offset=i] * ctx.dt
            vx[unsafe_offset=i] = vx[unsafe_offset=i] * drag
            vy[unsafe_offset=i] = vy[unsafe_offset=i] * drag
            vz[unsafe_offset=i] = vz[unsafe_offset=i] * drag


struct RobotechSimWorld(Movable):
    # ECS Registry & Parallel Pool
    var registry: ArchetypeRegistry
    var spark_slots: ArchetypeSlots
    var pool: JobPool

    # Player VF-1S Valkyrie ("SKULL LEADER")
    var mode: Int
    var morph_fg: SpringFloat
    var morph_gb: SpringFloat
    var pos: Vec3
    var vel: Vec3
    var yaw: Float32
    var pitch: Float32
    var roll: SpringFloat
    var barrel_roll_t: Float32
    var barrel_roll_dir: Float32
    var hull: Float32
    var max_hull: Float32
    var shield: Float32
    var boost_fuel: Float32
    var gun_cd: Float32
    var gun_flash_t: Float32
    var gun_ammo: Int
    var gun_reload_t: Float32
    var missiles_ready: Int
    var missile_regen_t: Float32
    var lock_sweep_t: Float32
    var total_locks: Int

    # Wingman VF-1J ("MAX STERLING - SKULL 2")
    var wing_pos: Vec3
    var wing_yaw: Float32
    var wing_pitch: Float32
    var wing_morph_fg: SpringFloat
    var wing_morph_gb: SpringFloat
    var wing_fire_cd: Float32
    var wing_gun_flash: Float32

    # Camera spring state
    var cam_eye_spring: SpringVec3
    var cam_target_spring: SpringVec3
    var cam_shake: Float32

    # Zentraedi Squadron
    var enemies: List[ZentraediUnit]
    var wave: Int
    var score: Int
    var kills: Int
    var incoming_missile_count: Int
    var comms_banner: String
    var comms_timer: Float32

    # Itano Circus Micro-Missiles (SoA)
    var mis_active: List[Bool]
    var mis_team: List[Int]
    var mis_target: List[Int]
    var mis_x: SimdColumn[DType.float32]
    var mis_y: SimdColumn[DType.float32]
    var mis_z: SimdColumn[DType.float32]
    var mis_px: SimdColumn[DType.float32]
    var mis_py: SimdColumn[DType.float32]
    var mis_pz: SimdColumn[DType.float32]
    var mis_vx: SimdColumn[DType.float32]
    var mis_vy: SimdColumn[DType.float32]
    var mis_vz: SimdColumn[DType.float32]
    var mis_phase: SimdColumn[DType.float32]
    var mis_life: SimdColumn[DType.float32]
    var mis_trail: List[Vec3]
    var mis_trail_len: List[Int]
    var fireballs: List[CombatFireball]
    var fireball_cursor: Int

    # GU-11 Gunpod & Zentraedi Plasma Bolts (SoA)
    var bolt_active: List[Bool]
    var bolt_team: List[Int]
    var bolt_x: SimdColumn[DType.float32]
    var bolt_y: SimdColumn[DType.float32]
    var bolt_z: SimdColumn[DType.float32]
    var bolt_vx: SimdColumn[DType.float32]
    var bolt_vy: SimdColumn[DType.float32]
    var bolt_vz: SimdColumn[DType.float32]
    var bolt_life: SimdColumn[DType.float32]

    # 2,048 Parallel SIMD Sparks (Afterburners, Missile Contrails, Plasma Explosions)
    var spark_x: SimdColumn[DType.float32]
    var spark_y: SimdColumn[DType.float32]
    var spark_z: SimdColumn[DType.float32]
    var spark_vx: SimdColumn[DType.float32]
    var spark_vy: SimdColumn[DType.float32]
    var spark_vz: SimdColumn[DType.float32]
    var spark_life: SimdColumn[DType.float32]
    var spark_kind: List[UInt8]   # 0=cyan thruster/trail, 1=amber fire, 2=crimson plasma
    var spark_cursor: Int

    var sim_time: Float32
    var tick_count: Int
    var rng_state: UInt32
    var sound_events: Int

    def __init__(out self, workers: Int = -1):
        self.registry = ArchetypeRegistry()
        _ = self.registry.register("vf1_valkyrie")
        _ = self.registry.register("zentraedi_pod")
        _ = self.registry.register("micro_missile")
        var sp_arch = self.registry.register("combat_spark")
        self.spark_slots = ArchetypeSlots(sp_arch, MAX_SPARKS)
        self.pool = JobPool()
        var w_count = max(1, default_worker_count()) if workers < 0 else workers
        _ = self.pool.start(w_count)

        self.mode = MODE_FIGHTER
        self.morph_fg = SpringFloat(0.0)
        self.morph_gb = SpringFloat(0.0)
        self.pos = Vec3(0.0, 26.0, 0.0)
        self.vel = Vec3(0.0, 0.0, 58.0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = SpringFloat(0.0)
        self.barrel_roll_t = 0.0
        self.barrel_roll_dir = 1.0
        self.hull = 100.0
        self.max_hull = 100.0
        self.shield = 100.0
        self.boost_fuel = 100.0
        self.gun_cd = 0.0
        self.gun_flash_t = 0.0
        self.gun_ammo = 200
        self.gun_reload_t = 0.0
        self.missiles_ready = 36
        self.missile_regen_t = 0.0
        self.lock_sweep_t = 0.0
        self.total_locks = 0

        self.wing_pos = Vec3(-14.0, 28.0, -8.0)
        self.wing_yaw = 0.0
        self.wing_pitch = 0.0
        self.wing_morph_fg = SpringFloat(0.0)
        self.wing_morph_gb = SpringFloat(0.0)
        self.wing_fire_cd = 1.4
        self.wing_gun_flash = 0.0

        self.cam_eye_spring = SpringVec3(Vec3(0.0, 31.0, -16.0))
        self.cam_target_spring = SpringVec3(Vec3(0.0, 26.0, 20.0))
        self.cam_shake = 0.0

        self.enemies = List[ZentraediUnit]()
        for _ in range(MAX_ENEMIES):
            self.enemies.append(ZentraediUnit())
        self.wave = 1
        self.score = 0
        self.kills = 0
        self.incoming_missile_count = 0
        self.comms_banner = "SDF-1 COMBAT SIMULATOR ONLINE — SKULL LEADER CLEAR TO ENGAGE!"
        self.comms_timer = 5.0

        self.mis_active = List[Bool](length=MAX_MISSILES, fill=False)
        self.mis_team = List[Int](length=MAX_MISSILES, fill=TEAM_UN_SPACY)
        self.mis_target = List[Int](length=MAX_MISSILES, fill=-1)
        self.mis_x = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_y = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_z = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_px = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_py = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_pz = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_vx = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_vy = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_vz = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_phase = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_life = SimdColumn[DType.float32](MAX_MISSILES)
        self.mis_trail = List[Vec3](length=MAX_MISSILES * MIS_TRAIL_LEN, fill=Vec3())
        self.mis_trail_len = List[Int](length=MAX_MISSILES, fill=0)
        self.fireballs = List[CombatFireball](length=MAX_FIREBALLS, fill=CombatFireball())
        self.fireball_cursor = 0

        self.bolt_active = List[Bool](length=MAX_BOLTS, fill=False)
        self.bolt_team = List[Int](length=MAX_BOLTS, fill=TEAM_UN_SPACY)
        self.bolt_x = SimdColumn[DType.float32](MAX_BOLTS)
        self.bolt_y = SimdColumn[DType.float32](MAX_BOLTS)
        self.bolt_z = SimdColumn[DType.float32](MAX_BOLTS)
        self.bolt_vx = SimdColumn[DType.float32](MAX_BOLTS)
        self.bolt_vy = SimdColumn[DType.float32](MAX_BOLTS)
        self.bolt_vz = SimdColumn[DType.float32](MAX_BOLTS)
        self.bolt_life = SimdColumn[DType.float32](MAX_BOLTS)

        self.spark_x = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_y = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_z = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_vx = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_vy = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_vz = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_life = SimdColumn[DType.float32](MAX_SPARKS)
        self.spark_kind = List[UInt8](length=MAX_SPARKS, fill=0)
        self.spark_cursor = 0

        self.sim_time = 0.0
        self.tick_count = 0
        self.rng_state = 0x56463153  # "VF1S"
        self.sound_events = 0
        self.spawn_wave(1)

    def shutdown(mut self):
        self.pool.stop()

    @always_inline
    def rand_f32(mut self) -> Float32:
        var x = self.rng_state
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        self.rng_state = x
        return Float32(x & 0xFFFF) / Float32(65535.0)

    @always_inline
    def forward_vec(self) -> Vec3:
        return Vec3(sin(self.yaw) * cos(self.pitch), sin(self.pitch), cos(self.yaw) * cos(self.pitch)).normalized()

    @always_inline
    def right_vec(self) -> Vec3:
        return Vec3(-cos(self.yaw), 0.0, sin(self.yaw)).normalized()

    def mode_name(self) -> String:
        if self.mode == MODE_FIGHTER:
            return "FIGHTER"
        if self.mode == MODE_GUARDIAN:
            return "GUARDIAN"
        return "BATTLOID"

    def set_mode(mut self, target_mode: Int):
        var clamped = max(0, min(2, target_mode))
        if clamped == self.mode:
            return
        self.mode = clamped
        self.sound_events |= EVT_TRANSFORM
        if clamped == MODE_FIGHTER:
            self.comms_banner = "SKULL LEADER: VF-1 TRANSFORM -> FIGHTER INTERCEPT MODE!"
        elif clamped == MODE_GUARDIAN:
            self.comms_banner = "SKULL LEADER: VF-1 TRANSFORM -> GUARDIAN VTOL HOVER MODE!"
        else:
            self.comms_banner = "SKULL LEADER: VF-1 TRANSFORM -> BATTLOID COMBAT MODE!"
        self.comms_timer = 3.2

    def emit_spark(mut self, pos: Vec3, vel: Vec3, life: Float32, kind: UInt8):
        var idx = self.spark_cursor
        self.spark_cursor = (self.spark_cursor + 1) & (MAX_SPARKS - 1)
        self.spark_x.set(idx, pos.x)
        self.spark_y.set(idx, pos.y)
        self.spark_z.set(idx, pos.z)
        self.spark_vx.set(idx, vel.x)
        self.spark_vy.set(idx, vel.y)
        self.spark_vz.set(idx, vel.z)
        self.spark_life.set(idx, life)
        self.spark_kind[idx] = kind

    def emit_explosion(mut self, center: Vec3, count: Int, speed: Float32):
        for _ in range(count):
            var dx = self.rand_f32() * Float32(2.0) - Float32(1.0)
            var dy = self.rand_f32() * Float32(2.0) - Float32(1.0)
            var dz = self.rand_f32() * Float32(2.0) - Float32(1.0)
            var dir = Vec3(dx, dy, dz).normalized()
            var sp = speed * (Float32(0.35) + self.rand_f32() * Float32(0.85))
            var kind = UInt8(1) if self.rand_f32() < Float32(0.65) else UInt8(2)
            self.emit_spark(center, dir * sp, Float32(0.45) + self.rand_f32() * Float32(0.65), kind)
        self.sound_events |= EVT_EXPLODE
        var fb = CombatFireball()
        fb.active = True
        fb.pos = center
        fb.max_radius = Float32(4.4) if count >= 30 else Float32(2.6)
        fb.duration = Float32(0.72) if count >= 30 else Float32(0.42)
        fb.life = fb.duration
        self.fireballs[self.fireball_cursor] = fb
        self.fireball_cursor = (self.fireball_cursor + 1) % MAX_FIREBALLS
        var dist = (center - self.pos).length()
        if dist < Float32(120.0):
            self.cam_shake = min(Float32(2.5), self.cam_shake + (Float32(120.0) - dist) / Float32(75.0))

    def spawn_wave(mut self, wave_num: Int):
        self.wave = wave_num
        var pod_count = min(MAX_ENEMIES - 2, 5 + wave_num * 3)
        var glaug_count = min(4, (wave_num + 1) // 2)
        var fwd = self.forward_vec()
        var right = self.right_vec()

        for i in range(MAX_ENEMIES):
            self.enemies[i] = ZentraediUnit()

        for i in range(pod_count):
            var is_glaug = i < glaug_count
            var angle = (Float32(i) / Float32(max(1, pod_count)) - Float32(0.5)) * Float32(1.85)
            var dist = Float32(95.0) + Float32((i * 19) % 65)
            var alt = Float32(12.0) + Float32((i * 13) % 34)
            var spawn_p = self.pos + fwd * (cos(angle) * dist) + right * (sin(angle) * dist)
            spawn_p.y = alt
            var u = ZentraediUnit()
            u.active = True
            u.kind = ENEMY_GLAUG if is_glaug else ENEMY_REGULT
            u.pos = spawn_p
            u.yaw = self.yaw + Float32(3.14159)
            u.max_hp = Float32(190.0) if is_glaug else Float32(75.0)
            u.hp = u.max_hp
            u.fire_cd = Float32(0.6) + Float32(i % 5) * Float32(0.35)
            u.missile_cd = Float32(2.8) + Float32(i % 4) * Float32(0.9)
            u.stride_phase = Float32(i) * Float32(0.9)
            self.enemies[i] = u

        self.comms_banner = (
            "SDF-1 RADAR: ZENTRAEDI WAVE "
            + String(wave_num)
            + " INBOUND ("
            + String(pod_count)
            + " BATTLEPODS)!"
        )
        self.comms_timer = 4.5

    def active_enemy_count(self) -> Int:
        var n = 0
        for i in range(MAX_ENEMIES):
            if self.enemies[i].active:
                n += 1
        return n

    def spawn_bolt(mut self, origin: Vec3, vel: Vec3, team: Int, life: Float32 = 1.6):
        for i in range(MAX_BOLTS):
            if not self.bolt_active[i]:
                self.bolt_active[i] = True
                self.bolt_team[i] = team
                self.bolt_x.set(i, origin.x)
                self.bolt_y.set(i, origin.y)
                self.bolt_z.set(i, origin.z)
                self.bolt_vx.set(i, vel.x)
                self.bolt_vy.set(i, vel.y)
                self.bolt_vz.set(i, vel.z)
                self.bolt_life.set(i, life)
                return

    def spawn_micro_missile(mut self, origin: Vec3, init_vel: Vec3, team: Int, target_idx: Int):
        for i in range(MAX_MISSILES):
            if not self.mis_active[i]:
                self.mis_active[i] = True
                self.mis_team[i] = team
                self.mis_target[i] = target_idx
                self.mis_x.set(i, origin.x)
                self.mis_y.set(i, origin.y)
                self.mis_z.set(i, origin.z)
                self.mis_px.set(i, origin.x)
                self.mis_py.set(i, origin.y)
                self.mis_pz.set(i, origin.z)
                self.mis_vx.set(i, init_vel.x)
                self.mis_vy.set(i, init_vel.y)
                self.mis_vz.set(i, init_vel.z)
                self.mis_phase.set(i, self.rand_f32() * Float32(6.2831853))
                self.mis_life.set(i, Float32(5.2))
                var base = i * MIS_TRAIL_LEN
                for k in range(MIS_TRAIL_LEN):
                    self.mis_trail[base + k] = origin
                self.mis_trail_len[i] = 1
                return

    def launch_itano_salvo(mut self, max_count: Int = 24):
        """Fires an Ichiro Itano corkscrew micro-missile swarm distributed across locked or front targets."""
        if self.missiles_ready <= 0:
            return
        var targets = List[Int]()
        for i in range(MAX_ENEMIES):
            if self.enemies[i].active and self.enemies[i].lock_marks > 0:
                for _ in range(self.enemies[i].lock_marks):
                    targets.append(i)
                self.enemies[i].lock_marks = 0
        self.total_locks = 0

        # If quick-fired without pre-locking, distribute across all enemies in forward hemisphere
        if len(targets) == 0:
            var fwd = self.forward_vec()
            for i in range(MAX_ENEMIES):
                if self.enemies[i].active:
                    var to_e = (self.enemies[i].pos - self.pos).normalized()
                    if to_e.dot(fwd) > Float32(0.15):
                        targets.append(i)
            if len(targets) == 0:
                for i in range(MAX_ENEMIES):
                    if self.enemies[i].active:
                        targets.append(i)

        if len(targets) == 0:
            return

        var count = min(self.missiles_ready, max(8, min(max_count, len(targets) * 4)))
        var fwd = self.forward_vec()
        var right = self.right_vec()
        var up = right.cross(fwd).normalized()

        for m in range(count):
            var tid = targets[m % len(targets)]
            var sgn = Float32(-1.0) if (m & 1) == 0 else Float32(1.0)
            var ring_ang = Float32(m) * Float32(0.785398)
            var pod_offset = right * (sgn * Float32(1.65)) + up * Float32(0.55)
            # Wild outward radial blossom velocity (signature Macross missile pod burst!)
            var radial = right * (cos(ring_ang) * Float32(34.0)) + up * (sin(ring_ang) * Float32(28.0) + Float32(10.0))
            var launch_vel = self.vel + fwd * Float32(52.0) + radial
            self.spawn_micro_missile(self.pos + pod_offset, launch_vel, TEAM_UN_SPACY, tid)

        self.missiles_ready -= count
        self.sound_events |= EVT_SALVO
        self.comms_banner = "FOX THREE! ITANO CIRCUS SALVO (" + String(count) + " MICRO-MISSILES) AWAY!"
        self.comms_timer = 2.8

    def nearest_reticle_enemy(self, min_dot: Float32 = 0.88) -> Int:
        var fwd = self.forward_vec()
        var best = -1
        var best_score = min_dot
        for i in range(MAX_ENEMIES):
            if self.enemies[i].active:
                var d = (self.enemies[i].pos - self.pos).normalized().dot(fwd)
                if d > best_score:
                    best_score = d
                    best = i
        return best

    def step(mut self, ctrl: PilotControls, dt: Float32):
        self.sound_events = 0
        self.sim_time += dt
        self.tick_count += 1
        if self.comms_timer > Float32(0.0):
            self.comms_timer = max(Float32(0.0), self.comms_timer - dt)
        if self.cam_shake > Float32(0.0):
            self.cam_shake = max(Float32(0.0), self.cam_shake - dt * Float32(3.8))
        for fi in range(MAX_FIREBALLS):
            if self.fireballs[fi].active:
                self.fireballs[fi].life -= dt
                if self.fireballs[fi].life <= Float32(0.0):
                    self.fireballs[fi].active = False

        # 1. Mode Switch & Smooth VF-1 Transformation Springs
        if ctrl.request_mode >= 0:
            self.set_mode(ctrl.request_mode)
        var goal_fg = Float32(0.0) if self.mode == MODE_FIGHTER else Float32(1.0)
        var goal_gb = Float32(1.0) if self.mode == MODE_BATTLOID else Float32(0.0)
        self.morph_fg.update(goal_fg, Float32(0.14), dt)
        self.morph_gb.update(goal_gb, Float32(0.14), dt)

        # 2. Mode-Specific Veritech Flight / VTOL / Battloid Thruster Physics
        var turn_rate = Float32(1.55) if self.mode == MODE_FIGHTER else (Float32(2.15) if self.mode == MODE_GUARDIAN else Float32(2.65))
        self.yaw += ctrl.yaw_in * turn_rate * dt
        self.pitch = max(Float32(-1.05), min(Float32(1.05), self.pitch + ctrl.pitch_in * turn_rate * Float32(0.85) * dt))

        if ctrl.barrel_roll_dir != 0 and self.barrel_roll_t <= Float32(0.0) and self.mode != MODE_BATTLOID:
            self.barrel_roll_t = Float32(0.52)
            self.barrel_roll_dir = Float32(ctrl.barrel_roll_dir)
            # Evasive barrel roll breaks hostile missile locks!
            for m in range(MAX_MISSILES):
                if self.mis_active[m] and self.mis_team[m] == TEAM_ZENTRAEDI:
                    self.mis_target[m] = -1

        var roll_goal = -ctrl.yaw_in * (Float32(0.82) if self.mode == MODE_FIGHTER else Float32(0.28))
        if self.barrel_roll_t > Float32(0.0):
            self.barrel_roll_t = max(Float32(0.0), self.barrel_roll_t - dt)
            roll_goal += self.barrel_roll_dir * (Float32(1.0) - self.barrel_roll_t / Float32(0.52)) * Float32(6.2831853)
        self.roll.update(roll_goal, Float32(0.09), dt)

        var fwd = self.forward_vec()
        var right = self.right_vec()
        var boosting = ctrl.boost_held and self.boost_fuel > Float32(2.0)
        if boosting:
            self.boost_fuel = max(Float32(0.0), self.boost_fuel - dt * Float32(24.0))
        else:
            self.boost_fuel = min(Float32(100.0), self.boost_fuel + dt * Float32(14.0))

        if self.mode == MODE_FIGHTER:
            var base_spd = Float32(62.0) + ctrl.throttle_in * Float32(24.0)
            if boosting:
                base_spd = Float32(104.0)
            var target_vel = fwd * max(Float32(34.0), base_spd)
            if self.barrel_roll_t > Float32(0.0):
                target_vel = target_vel + right * (self.barrel_roll_dir * Float32(38.0))
            self.vel = self.vel + (target_vel - self.vel) * min(Float32(1.0), dt * Float32(5.5))
        elif self.mode == MODE_GUARDIAN:
            var spd = Float32(48.0) if not boosting else Float32(72.0)
            var hover_vel = (
                fwd * (ctrl.throttle_in * spd)
                + right * (ctrl.strafe_x * spd * Float32(0.95))
                + Vec3(0.0, ctrl.lift_y * Float32(34.0), 0.0)
            )
            self.vel = self.vel + (hover_vel - self.vel) * min(Float32(1.0), dt * Float32(6.5))
        else:
            # Battloid 360-degree omni-thruster combat + Pinpoint Barrier Shield on SHIFT
            var spd = Float32(36.0)
            var omni_vel = (
                fwd * (ctrl.throttle_in * spd)
                + right * (ctrl.strafe_x * spd)
                + Vec3(0.0, ctrl.lift_y * Float32(32.0), 0.0)
            )
            self.vel = self.vel + (omni_vel - self.vel) * min(Float32(1.0), dt * Float32(8.0))
            if ctrl.boost_held and self.shield > Float32(1.0):
                self.shield = max(Float32(0.0), self.shield - dt * Float32(18.0))
            else:
                self.shield = min(Float32(100.0), self.shield + dt * Float32(10.0))

        self.pos = self.pos + self.vel * dt
        var floor_h = Float32(3.2) if self.mode != MODE_FIGHTER else Float32(4.5)
        if self.pos.y < floor_h:
            self.pos.y = floor_h
            if self.vel.y < Float32(0.0):
                self.vel.y = 0.0
            if self.pitch < Float32(0.0):
                self.pitch = 0.0

        # Emit cyan VF-1 thermonuclear afterburner sparks
        self.emit_spark(
            self.pos - fwd * Float32(3.2) + right * Float32(-0.55),
            -fwd * Float32(24.0), Float32(0.18), 0,
        )
        self.emit_spark(
            self.pos - fwd * Float32(3.2) + right * Float32(0.55),
            -fwd * Float32(24.0), Float32(0.18), 0,
        )

        # 3. Weapons: GU-11 55mm Gunpod & Itano Circus Multi-Lock Radar
        self.gun_cd = max(Float32(0.0), self.gun_cd - dt)
        self.gun_flash_t = max(Float32(0.0), self.gun_flash_t - dt)
        if self.gun_reload_t > Float32(0.0):
            self.gun_reload_t = max(Float32(0.0), self.gun_reload_t - dt)
            if self.gun_reload_t <= Float32(0.0):
                self.gun_ammo = 200

        if ctrl.fire_gunpod and self.gun_cd <= Float32(0.0) and self.gun_ammo > 0:
            var rof = Float32(0.055) if self.mode == MODE_BATTLOID else Float32(0.078)
            self.gun_cd = rof
            self.gun_flash_t = Float32(0.045)
            self.gun_ammo -= 1
            if self.gun_ammo == 0:
                self.gun_reload_t = Float32(1.1)
            var aim_dir = fwd
            var assist_id = self.nearest_reticle_enemy(0.955)
            if assist_id >= 0:
                var lead_pos = self.enemies[assist_id].pos + self.enemies[assist_id].vel * (
                    (self.enemies[assist_id].pos - self.pos).length() / Float32(260.0)
                )
                aim_dir = (lead_pos - self.pos).normalized()
            self.spawn_bolt(self.pos + fwd * Float32(2.8), self.vel + aim_dir * Float32(260.0), TEAM_UN_SPACY, Float32(1.35))
            self.sound_events |= EVT_GUNPOD

        # Missile regeneration & multi-lock painting
        if self.missiles_ready < 36:
            self.missile_regen_t += dt
            if self.missile_regen_t >= Float32(0.18):
                self.missile_regen_t = 0.0
                self.missiles_ready += 1

        if ctrl.lock_held and self.missiles_ready > 0:
            self.lock_sweep_t += dt
            if self.lock_sweep_t >= Float32(0.075) and self.total_locks < min(24, self.missiles_ready):
                self.lock_sweep_t = 0.0
                for i in range(MAX_ENEMIES):
                    var idx = (self.tick_count + i) % MAX_ENEMIES
                    if self.enemies[idx].active and self.enemies[idx].lock_marks < 4:
                        var d = (self.enemies[idx].pos - self.pos).normalized().dot(fwd)
                        if d > Float32(0.68):
                            self.enemies[idx].lock_marks += 1
                            self.total_locks += 1
                            self.sound_events |= EVT_LOCK
                            break

        if ctrl.launch_salvo and self.total_locks > 0:
            self.launch_itano_salvo(min(30, self.total_locks * 2))
        elif ctrl.quick_salvo:
            self.launch_itano_salvo(24)

        # 4. Update Wingman AI ("MAX STERLING - SKULL 2")
        var wing_goal = self.pos - fwd * Float32(12.0) - right * Float32(15.0) + Vec3(0.0, 3.5, 0.0)
        self.wing_pos = self.wing_pos + (wing_goal - self.wing_pos) * min(Float32(1.0), dt * Float32(3.4))
        self.wing_yaw = self.yaw
        self.wing_pitch = self.pitch * Float32(0.7)
        self.wing_morph_fg.update(goal_fg, Float32(0.18), dt)
        self.wing_morph_gb.update(goal_gb, Float32(0.18), dt)
        self.wing_gun_flash = max(Float32(0.0), self.wing_gun_flash - dt)
        self.wing_fire_cd -= dt
        if self.wing_fire_cd <= Float32(0.0):
            self.wing_fire_cd = Float32(1.65)
            for i in range(MAX_ENEMIES):
                if self.enemies[i].active and (self.enemies[i].pos - self.wing_pos).length() < Float32(185.0):
                    var wdir = (self.enemies[i].pos - self.wing_pos).normalized()
                    self.spawn_micro_missile(
                        self.wing_pos + Vec3(0.0, 1.5, 0.0),
                        wdir * Float32(45.0) + Vec3(0.0, 18.0, 0.0),
                        TEAM_UN_SPACY, i,
                    )
                    self.wing_gun_flash = Float32(0.12)
                    break

        # 5. Update Zentraedi Tactical AI (Regult Battlepods & Glaug Officers)
        for i in range(MAX_ENEMIES):
            if not self.enemies[i].active:
                continue
            var e = self.enemies[i]
            var to_player = self.pos - e.pos
            var dist = max(Float32(1.0), to_player.length())
            var dir_p = to_player * (Float32(1.0) / dist)
            e.yaw = atan2(dir_p.x, dir_p.z)
            e.pitch = max(Float32(-0.5), min(Float32(0.5), dir_p.y))
            e.stride_phase += dt * Float32(6.5)
            e.firing_flash = max(Float32(0.0), e.firing_flash - dt)

            var tang = Vec3(dir_p.z, Float32(0.0), -dir_p.x) * (Float32(1.0) if (i & 1) == 0 else Float32(-1.0))
            var desired_spd = Float32(34.0) if e.kind == ENEMY_GLAUG else Float32(28.0)
            var approach = dir_p * (desired_spd if dist > Float32(62.0) else (Float32(-18.0) if dist < Float32(32.0) else Float32(0.0)))
            var weave_y = sin(self.sim_time * Float32(1.8) + Float32(i)) * Float32(9.0)
            var target_v = approach + tang * Float32(22.0) + Vec3(0.0, weave_y, 0.0)
            e.vel = e.vel + (target_v - e.vel) * min(Float32(1.0), dt * Float32(2.8))
            e.pos = e.pos + e.vel * dt
            if e.pos.y < Float32(3.5):
                e.pos.y = Float32(3.5)

            var fire_bolt_now = False
            e.fire_cd -= dt
            if e.fire_cd <= Float32(0.0) and dist < Float32(190.0):
                e.fire_cd = Float32(1.25) if e.kind == ENEMY_GLAUG else Float32(1.85)
                e.firing_flash = Float32(0.12)
                fire_bolt_now = True

            var fire_missile_now = False
            if e.kind == ENEMY_GLAUG:
                e.missile_cd -= dt
                if e.missile_cd <= Float32(0.0) and dist < Float32(220.0):
                    e.missile_cd = Float32(5.2)
                    fire_missile_now = True

            self.enemies[i] = e
            if fire_bolt_now:
                self.spawn_bolt(e.pos + dir_p * Float32(3.0), dir_p * Float32(145.0), TEAM_ZENTRAEDI, Float32(1.8))
            if fire_missile_now:
                self.spawn_micro_missile(
                    e.pos + Vec3(0.0, 2.4, 0.0),
                    dir_p * Float32(42.0) + Vec3(0.0, 18.0, 0.0),
                    TEAM_ZENTRAEDI, 0,
                )

        # 6. Step Itano Circus Corkscrew Micro-Missiles
        self.incoming_missile_count = 0
        for m in range(MAX_MISSILES):
            if not self.mis_active[m]:
                continue
            var life = self.mis_life.get(m) - dt
            if life <= Float32(0.0):
                self.mis_active[m] = False
                continue
            self.mis_life.set(m, life)

            var mx = self.mis_x.get(m)
            var my = self.mis_y.get(m)
            var mz = self.mis_z.get(m)
            self.mis_px.set(m, mx)
            self.mis_py.set(m, my)
            self.mis_pz.set(m, mz)

            var mpos = Vec3(mx, my, mz)
            var mvel = Vec3(self.mis_vx.get(m), self.mis_vy.get(m), self.mis_vz.get(m))
            var ph = self.mis_phase.get(m) + dt * Float32(13.5)
            self.mis_phase.set(m, ph)

            var tid = self.mis_target[m]
            var has_goal = False
            var goal_pos = Vec3()
            if self.mis_team[m] == TEAM_UN_SPACY:
                if tid >= 0 and tid < MAX_ENEMIES and self.enemies[tid].active:
                    goal_pos = self.enemies[tid].pos + self.enemies[tid].vel * Float32(0.18)
                    has_goal = True
            else:
                self.incoming_missile_count += 1
                if tid >= 0:
                    goal_pos = self.pos + self.vel * Float32(0.12)
                    has_goal = True

            if has_goal:
                var to_g = goal_pos - mpos
                var dist_g = max(Float32(0.1), to_g.length())
                var desired_dir = to_g * (Float32(1.0) / dist_g)
                var basis_r = desired_dir.cross(Vec3(0.0, 1.0, 0.0))
                if basis_r.length_squared() < Float32(0.01):
                    basis_r = Vec3(1.0, 0.0, 0.0)
                basis_r = basis_r.normalized()
                var basis_u = basis_r.cross(desired_dir).normalized()
                # Signature Itano Circus helical corkscrew: wide at launch, tightens near impact!
                var swirl_amp = min(Float32(0.58), dist_g / Float32(85.0)) * Float32(0.55)
                var corkscrew_dir = (
                    desired_dir + basis_r * (cos(ph) * swirl_amp) + basis_u * (sin(ph) * swirl_amp)
                ).normalized()
                var spd = Float32(118.0) if self.mis_team[m] == TEAM_UN_SPACY else Float32(78.0)
                mvel = mvel + (corkscrew_dir * spd - mvel) * min(Float32(1.0), dt * Float32(6.2))

            mpos = mpos + mvel * dt
            self.mis_x.set(m, mpos.x)
            self.mis_y.set(m, mpos.y)
            self.mis_z.set(m, mpos.z)
            self.mis_vx.set(m, mvel.x)
            self.mis_vy.set(m, mvel.y)
            self.mis_vz.set(m, mvel.z)
            var tbase = m * MIS_TRAIL_LEN
            var tlen = min(MIS_TRAIL_LEN, self.mis_trail_len[m] + 1)
            for k in range(tlen - 1, 0, -1):
                self.mis_trail[tbase + k] = self.mis_trail[tbase + k - 1]
            self.mis_trail[tbase] = mpos
            self.mis_trail_len[m] = tlen

            # Contrail smoke spark
            self.emit_spark(
                mpos,
                mvel * Float32(-0.08),
                Float32(0.26),
                UInt8(0) if self.mis_team[m] == TEAM_UN_SPACY else UInt8(2),
            )

        # 7. Step Gunpod & Plasma Bolts
        for b in range(MAX_BOLTS):
            if not self.bolt_active[b]:
                continue
            var bl = self.bolt_life.get(b) - dt
            if bl <= Float32(0.0):
                self.bolt_active[b] = False
                continue
            self.bolt_life.set(b, bl)
            self.bolt_x.set(b, self.bolt_x.get(b) + self.bolt_vx.get(b) * dt)
            self.bolt_y.set(b, self.bolt_y.get(b) + self.bolt_vy.get(b) * dt)
            self.bolt_z.set(b, self.bolt_z.get(b) + self.bolt_vz.get(b) * dt)

        # 8. Broadphase Collision Checks (`flint.physics.broadphase`)
        self._resolve_combat_collisions()

        # 9. Parallel SIMD Spark Integration across `JobPool` (`run_parallel_ctx`)
        var sp_ctx = SparkJobCtx(
            self.spark_x.base_addr(), self.spark_y.base_addr(), self.spark_z.base_addr(),
            self.spark_vx.base_addr(), self.spark_vy.base_addr(), self.spark_vz.base_addr(),
            self.spark_life.base_addr(), dt,
        )
        run_parallel_ctx(self.pool, MAX_SPARKS // SPARK_CHUNK, sp_ctx, step_spark_pool_chunk)
        self.spark_x.keep_alive()
        self.spark_y.keep_alive()
        self.spark_z.keep_alive()
        self.spark_vx.keep_alive()
        self.spark_vy.keep_alive()
        self.spark_vz.keep_alive()
        self.spark_life.keep_alive()

        # 10. Dynamic Spring Chase Camera Update & Wave Progression
        var chase_dist = Float32(14.5) if self.mode == MODE_FIGHTER else Float32(12.5)
        var chase_up = Float32(4.2) if self.mode == MODE_FIGHTER else Float32(4.8)
        var desired_eye = self.pos - fwd * chase_dist + Vec3(0.0, chase_up, 0.0)
        var desired_tgt = self.pos + fwd * Float32(28.0)
        self.cam_eye_spring.update(desired_eye, Float32(0.08), dt)
        self.cam_target_spring.update(desired_tgt, Float32(0.06), dt)

        if self.active_enemy_count() == 0:
            self.hull = min(self.max_hull, self.hull + Float32(30.0))
            self.shield = Float32(100.0)
            self.missiles_ready = 36
            self.spawn_wave(self.wave + 1)

    def _resolve_combat_collisions(mut self):
        var bp = BroadphaseInput()
        var filt = CollisionFilter()
        # Row 0: Player VF-1
        bp.add(self.pos - Vec3(3.2, 3.2, 3.2), self.pos + Vec3(3.2, 3.2, 3.2), True, True, filt)
        # Rows 1..MAX_ENEMIES: Zentraedi Mecha
        for i in range(MAX_ENEMIES):
            var r = Float32(4.6) if self.enemies[i].kind == ENEMY_GLAUG else Float32(3.6)
            var ep = self.enemies[i].pos
            bp.add(ep - Vec3(r, r, r), ep + Vec3(r, r, r), self.enemies[i].active, True, filt)

        var mis_base = 1 + MAX_ENEMIES
        for m in range(MAX_MISSILES):
            var mp = Vec3(self.mis_x.get(m), self.mis_y.get(m), self.mis_z.get(m))
            bp.add(mp - Vec3(2.2, 2.2, 2.2), mp + Vec3(2.2, 2.2, 2.2), self.mis_active[m], True, filt)

        var bolt_base = mis_base + MAX_MISSILES
        for b in range(MAX_BOLTS):
            var bp_pos = Vec3(self.bolt_x.get(b), self.bolt_y.get(b), self.bolt_z.get(b))
            bp.add(bp_pos - Vec3(1.8, 1.8, 1.8), bp_pos + Vec3(1.8, 1.8, 1.8), self.bolt_active[b], True, filt)

        var total_rows = bp.rows()
        var pairs = find_pairs(bp, Int(Pointer(to=self.pool)))
        for p in pairs.pairs:
            var a = p // total_rows
            var b = p % total_rows

            # Player (0) vs Zentraedi Missile / Bolt
            if a == 0 and b >= mis_base and b < bolt_base:
                var mid = b - mis_base
                if self.mis_active[mid] and self.mis_team[mid] == TEAM_ZENTRAEDI:
                    self.mis_active[mid] = False
                    self._damage_player(Float32(14.0), Vec3(self.mis_x.get(mid), self.mis_y.get(mid), self.mis_z.get(mid)))
            elif a == 0 and b >= bolt_base:
                var bid = b - bolt_base
                if self.bolt_active[bid] and self.bolt_team[bid] == TEAM_ZENTRAEDI:
                    self.bolt_active[bid] = False
                    self._damage_player(Float32(6.0), Vec3(self.bolt_x.get(bid), self.bolt_y.get(bid), self.bolt_z.get(bid)))

            # Zentraedi Unit (1..MAX_ENEMIES) vs UN Spacy Missile / Gunpod Bolt
            elif a >= 1 and a <= MAX_ENEMIES:
                var eid = a - 1
                if not self.enemies[eid].active:
                    continue
                if b >= mis_base and b < bolt_base:
                    var mid = b - mis_base
                    if self.mis_active[mid] and self.mis_team[mid] == TEAM_UN_SPACY:
                        self.mis_active[mid] = False
                        self._damage_enemy(eid, Float32(34.0), Vec3(self.mis_x.get(mid), self.mis_y.get(mid), self.mis_z.get(mid)))
                elif b >= bolt_base:
                    var bid = b - bolt_base
                    if self.bolt_active[bid] and self.bolt_team[bid] == TEAM_UN_SPACY:
                        self.bolt_active[bid] = False
                        self._damage_enemy(eid, Float32(19.0), Vec3(self.bolt_x.get(bid), self.bolt_y.get(bid), self.bolt_z.get(bid)))

            # Gunpod shooting down incoming Zentraedi missiles in mid-air!
            elif a >= mis_base and a < bolt_base and b >= bolt_base:
                var mid = a - mis_base
                var bid = b - bolt_base
                if (
                    self.mis_active[mid]
                    and self.bolt_active[bid]
                    and self.mis_team[mid] == TEAM_ZENTRAEDI
                    and self.bolt_team[bid] == TEAM_UN_SPACY
                ):
                    self.mis_active[mid] = False
                    self.bolt_active[bid] = False
                    self.score += 50
                    self.emit_explosion(Vec3(self.mis_x.get(mid), self.mis_y.get(mid), self.mis_z.get(mid)), 18, 22.0)

    def _damage_player(mut self, amount: Float32, hit_pos: Vec3):
        var mitigated = amount * Float32(0.25) if (self.mode == MODE_BATTLOID and self.shield > Float32(5.0)) else amount
        self.hull = max(Float32(10.0), self.hull - mitigated)  # Training simulator safety floor keeps session continuous
        self.emit_explosion(hit_pos, 16, 20.0)
        if self.hull <= Float32(25.0):
            self.comms_banner = "WARNING: VERITECH HULL CRITICAL — DEPLOY PINPOINT BARRIER [3 + SHIFT]!"
            self.comms_timer = 3.0

    def _damage_enemy(mut self, eid: Int, amount: Float32, hit_pos: Vec3):
        self.enemies[eid].hp -= amount
        self.emit_explosion(hit_pos, 14, 18.0)
        if self.enemies[eid].hp <= Float32(0.0):
            var pos = self.enemies[eid].pos
            var was_glaug = self.enemies[eid].kind == ENEMY_GLAUG
            self.enemies[eid].active = False
            self.kills += 1
            self.score += 500 if was_glaug else 200
            self.emit_explosion(pos, 56 if was_glaug else 36, 38.0)
            if was_glaug:
                self.comms_banner = "SPLASH ONE ZENTRAEDI GLAUG OFFICER POD! (+500 PTS)"
                self.comms_timer = 3.0

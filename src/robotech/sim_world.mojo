"""Robotech: SDF-1 Veritech 3D Combat & Macross Saga Campaign Simulation (`flint.*`).

Powered by:
  - `flint.core.ecs`: `ArchetypeRegistry`, `ArchetypeSlots`, 8-wide `SimdColumn`, `run_parallel_ctx`
  - `flint.core.jobs`: Multi-threaded `JobPool` stepping 2,048 contrail/explosion/thruster sparks
  - `flint.core.springs`: Critically damped `SpringFloat` & `SpringVec3` VF-1 transformation & chase cam
  - `flint.physics.broadphase`: Spatial-hash `BroadphaseInput` + `find_pairs` collision detection
  - `flint.assets.checkpoint`: LZ4 + FNV-64 bit-exact versioned binary state checkpoints
"""
from std.math import sin, cos, sqrt, atan2
from flint.core.math import Vec3
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import ArchetypeRegistry, ArchetypeSlots, SimdColumn, run_parallel_ctx
from flint.core.springs import SpringFloat, SpringVec3
from flint.core.serial import BlobWriter, BlobReader, fourcc, fnv64
from flint.assets.checkpoint import save_checkpoint_lz4, load_checkpoint_lz4
from flint.assets.json_write import json_quote
from flint.physics.filter import CollisionFilter
from flint.physics.broadphase import BroadphaseInput, find_pairs

comptime MODE_FIGHTER: Int = 0
comptime MODE_GUARDIAN: Int = 1
comptime MODE_BATTLOID: Int = 2

# Playable UN Spacy / Quadrono Squadron Pilots (`[P]`)
comptime PILOT_SKULL_LEADER: Int = 0
comptime PILOT_MAX_STERLING: Int = 1
comptime PILOT_MIRIYA_ACE: Int = 2

# Macross Saga 3-Mission Campaign Theaters
comptime THEATER_MACROSS_ISLAND: Int = 0
comptime THEATER_HIGH_ORBIT: Int = 1
comptime THEATER_BOSS_FLEET: Int = 2

comptime ENEMY_REGULT: Int = 0
comptime ENEMY_GLAUG: Int = 1
comptime ENEMY_QUEADLUUN: Int = 2

# Destructible Thuverl-Salan Heavy Destroyer Subsystems
comptime SUBSYS_PORT_TURRET: Int = 0
comptime SUBSYS_STBD_TURRET: Int = 1
comptime SUBSYS_TORPEDO_BAY: Int = 2
comptime SUBSYS_BRIDGE: Int = 3
comptime BOSS_SUBSYS_COUNT: Int = 4
comptime SUBSYS_TARGET_BASE: Int = 900
comptime BOSS_TARGET_ID: Int = 999

comptime TEAM_UN_SPACY: Int = 0
comptime TEAM_ZENTRAEDI: Int = 1

# Bolt visual/damage archetypes
comptime BOLT_UN_GUNPOD: UInt8 = 0
comptime BOLT_ZENT_PLASMA: UInt8 = 1
comptime BOLT_UN_STRIKE_BEAM: UInt8 = 2
comptime BOLT_BOSS_HEAVY: UInt8 = 3

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
comptime EVT_REFLEX: Int = 1 << 5
comptime EVT_RADIO: Int = 1 << 6


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
    var toggle_fast_packs: Bool
    var fire_reflex_cannon: Bool
    var toggle_storm_attacker: Bool
    var cycle_pilot: Bool
    var cycle_mission: Bool

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
        self.toggle_fast_packs = False
        self.fire_reflex_cannon = False
        self.toggle_storm_attacker = False
        self.cycle_pilot = False
        self.cycle_mission = False


def encode_pilot_controls(ctrl: PilotControls) -> List[UInt8]:
    """Bit-exact binary encoding of per-tick `PilotControls` for `.flrp` deterministic flight replays."""
    var w = BlobWriter()
    w.f32(ctrl.pitch_in)
    w.f32(ctrl.yaw_in)
    w.f32(ctrl.strafe_x)
    w.f32(ctrl.lift_y)
    w.f32(ctrl.throttle_in)
    w.boolean(ctrl.boost_held)
    w.boolean(ctrl.fire_gunpod)
    w.boolean(ctrl.lock_held)
    w.boolean(ctrl.launch_salvo)
    w.boolean(ctrl.quick_salvo)
    w.integer(ctrl.barrel_roll_dir)
    w.integer(ctrl.request_mode)
    w.boolean(ctrl.toggle_fast_packs)
    w.boolean(ctrl.fire_reflex_cannon)
    w.boolean(ctrl.toggle_storm_attacker)
    w.boolean(ctrl.cycle_pilot)
    w.boolean(ctrl.cycle_mission)
    return w.data.copy()


def decode_pilot_controls(bytes: List[UInt8]) -> PilotControls:
    var c = PilotControls()
    if len(bytes) < 20:
        return c
    var r = BlobReader(bytes.copy())
    c.pitch_in = r.f32()
    c.yaw_in = r.f32()
    c.strafe_x = r.f32()
    c.lift_y = r.f32()
    c.throttle_in = r.f32()
    c.boost_held = r.boolean()
    c.fire_gunpod = r.boolean()
    c.lock_held = r.boolean()
    c.launch_salvo = r.boolean()
    c.quick_salvo = r.boolean()
    c.barrel_roll_dir = r.integer()
    c.request_mode = r.integer()
    c.toggle_fast_packs = r.boolean()
    c.fire_reflex_cannon = r.boolean()
    c.toggle_storm_attacker = r.boolean()
    c.cycle_pilot = r.boolean()
    c.cycle_mission = r.boolean()
    return c


@fieldwise_init
struct DestroyerSubsystem(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var hp: Float32
    var max_hp: Float32
    var local_offset: Vec3
    var lock_marks: Int

    def __init__(out self, local_offset: Vec3 = Vec3(), max_hp: Float32 = 280.0):
        self.active = True
        self.hp = max_hp
        self.max_hp = max_hp
        self.local_offset = local_offset
        self.lock_marks = 0


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

    # Campaign & SDF-1 Tactical Strike / Storm Attacker State
    var theater: Int
    var sdf1_hull: Float32
    var sdf1_storm_mode: Bool
    var sdf1_storm_morph: SpringFloat
    var daedalus_attack_t: Float32
    var daedalus_cd: Float32
    var reflex_charge: Float32
    var reflex_beam_t: Float32
    var reflex_origin_l: Vec3
    var reflex_origin_r: Vec3
    var reflex_target_pos: Vec3

    # Zentraedi Thuverl-Salan Heavy Destroyer Boss & 4 Destructible Subsystems
    var boss_active: Bool
    var boss_pos: Vec3
    var boss_vel: Vec3
    var boss_yaw: Float32
    var boss_hp: Float32
    var boss_max_hp: Float32
    var boss_fire_cd: Float32
    var boss_salvo_cd: Float32
    var boss_flash_t: Float32
    var boss_lock_marks: Int
    var boss_subsystems: List[DestroyerSubsystem]
    var skimming_hull: Bool

    # Player VF-1 Valkyrie & Squadron Pilot
    var pilot_id: Int
    var mode: Int
    var fast_packs: Bool
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
    var strike_cd: Float32
    var strike_flash_t: Float32
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

    # GU-11 Gunpod, Strike Beam Cannon & Zentraedi Plasma Bolts (SoA)
    var bolt_active: List[Bool]
    var bolt_team: List[Int]
    var bolt_kind: List[UInt8]
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
        _ = self.registry.register("queadluun_rau")
        _ = self.registry.register("thuverl_salan")
        _ = self.registry.register("micro_missile")
        var sp_arch = self.registry.register("combat_spark")
        self.spark_slots = ArchetypeSlots(sp_arch, MAX_SPARKS)
        self.pool = JobPool()
        var w_count = max(1, default_worker_count()) if workers < 0 else workers
        _ = self.pool.start(w_count)

        self.theater = THEATER_MACROSS_ISLAND
        self.sdf1_hull = 100.0
        self.sdf1_storm_mode = False
        self.sdf1_storm_morph = SpringFloat(0.0)
        self.daedalus_attack_t = 0.0
        self.daedalus_cd = 0.0
        self.reflex_charge = 68.0
        self.reflex_beam_t = 0.0
        self.reflex_origin_l = Vec3(-62.0, 70.0, 210.0)
        self.reflex_origin_r = Vec3(-42.0, 70.0, 210.0)
        self.reflex_target_pos = Vec3(0.0, 32.0, 150.0)

        self.boss_active = False
        self.boss_pos = Vec3(0.0, 48.0, 210.0)
        self.boss_vel = Vec3()
        self.boss_yaw = 3.14159
        self.boss_hp = 0.0
        self.boss_max_hp = 1800.0
        self.boss_fire_cd = 2.0
        self.boss_salvo_cd = 4.5
        self.boss_flash_t = 0.0
        self.boss_lock_marks = 0
        self.boss_subsystems = List[DestroyerSubsystem]()
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(-13.8, 2.5, 14.0), 280.0))
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(13.8, 2.5, 14.0), 280.0))
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(0.0, -3.5, 28.0), 260.0))
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(0.0, 10.5, -6.0), 320.0))
        self.skimming_hull = False

        self.pilot_id = PILOT_SKULL_LEADER
        self.mode = MODE_FIGHTER
        self.fast_packs = False
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
        self.strike_cd = 0.0
        self.strike_flash_t = 0.0
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
        self.comms_banner = "MISSION 1: MACROSS ISLAND ATMOSPHERIC INTERCEPT — SKULL LEADER SCRAMBLE!"
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
        self.bolt_kind = List[UInt8](length=MAX_BOLTS, fill=BOLT_UN_GUNPOD)
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

    @always_inline
    def max_missiles(self) -> Int:
        return 60 if self.fast_packs else 36

    def mission_title(self) -> String:
        if self.theater == THEATER_MACROSS_ISLAND:
            return "M1: MACROSS ISLAND"
        if self.theater == THEATER_HIGH_ORBIT:
            return "M2: HIGH ORBIT SDF-1"
        return "M3: ZENTRAEDI FLEET"

    def _set_comms(mut self, banner: String, duration: Float32 = 3.4):
        if banner != self.comms_banner:
            self.sound_events |= EVT_RADIO
        self.comms_banner = banner
        self.comms_timer = duration

    def pilot_callsign(self) -> String:
        if self.pilot_id == PILOT_SKULL_LEADER:
            return "SKULL-1 [ROY/RICK]"
        if self.pilot_id == PILOT_MAX_STERLING:
            return "SKULL-2 [MAX ACE]"
        return "QUADRONO [MIRIYA]"

    def pilot_full_name(self) -> String:
        if self.pilot_id == PILOT_SKULL_LEADER:
            return "CDR. ROY FOCKER / LT. RICK HUNTER (VF-1S SKULL-1)"
        if self.pilot_id == PILOT_MAX_STERLING:
            return "LT. MAXIMILIAN STERLING (VF-1J COBALT ACE)"
        return "ACE MIRIYA PARINA STERLING (VF-1J CRIMSON ACE)"

    def pilot_airframe(self) -> String:
        if self.pilot_id == PILOT_SKULL_LEADER:
            return "VF-1S"
        if self.pilot_id == PILOT_MAX_STERLING:
            return "VF-1J [MAX]"
        return "VF-1J [MIRIYA]"

    def pilot_perk_summary(self) -> String:
        if self.pilot_id == PILOT_SKULL_LEADER:
            return "+30% SDF-1 REFLEX CHARGE  |  BALANCED ALL-THEATER SKULL LEADER"
        if self.pilot_id == PILOT_MAX_STERLING:
            return "+18% TURN & ROLL AGILITY  |  +12% MACH SPEED  |  WIDE LEAD ASSIST"
        return "+22% GU-11 & STRIKE ROF   |  +35% RADAR MULTI-LOCK SWEEP SPEED"

    def set_pilot(mut self, pid: Int):
        self.pilot_id = max(0, min(2, pid))
        self.sound_events |= EVT_LOCK
        self._set_comms("SQUADRON PILOT ACTIVE: " + self.pilot_full_name(), 3.4)

    def cycle_pilot(mut self):
        self.set_pilot((self.pilot_id + 1) % 3)

    def _reset_boss_subsystems(mut self, scale_hp: Float32 = 1.0):
        self.boss_subsystems.clear()
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(-13.8, 2.5, 14.0), Float32(280.0) * scale_hp))
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(13.8, 2.5, 14.0), Float32(280.0) * scale_hp))
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(0.0, -3.5, 28.0), Float32(260.0) * scale_hp))
        self.boss_subsystems.append(DestroyerSubsystem(Vec3(0.0, 10.5, -6.0), Float32(320.0) * scale_hp))

    def boss_subsystem_world_pos(self, idx: Int) -> Vec3:
        if idx < 0 or idx >= len(self.boss_subsystems):
            return self.boss_pos
        var off = self.boss_subsystems[idx].local_offset
        var cy = cos(self.boss_yaw)
        var sy = sin(self.boss_yaw)
        var fwd = Vec3(sy, 0.0, cy)
        var right = Vec3(-cy, 0.0, sy)
        return self.boss_pos + right * off.x + Vec3(0.0, off.y, 0.0) + fwd * off.z

    def boss_subsystem_label(self, idx: Int) -> String:
        if idx == SUBSYS_PORT_TURRET:
            return "PORT TURRET"
        if idx == SUBSYS_STBD_TURRET:
            return "STBD TURRET"
        if idx == SUBSYS_TORPEDO_BAY:
            return "TORPEDO BAY"
        return "BRIDGE [CRIT]"

    def intact_subsystem_count(self) -> Int:
        if not self.boss_active:
            return 0
        var c = 0
        for i in range(len(self.boss_subsystems)):
            if self.boss_subsystems[i].active:
                c += 1
        return c

    def boss_bridge_destroyed(self) -> Bool:
        if len(self.boss_subsystems) <= SUBSYS_BRIDGE:
            return False
        return not self.boss_subsystems[SUBSYS_BRIDGE].active

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
            self._set_comms(self.pilot_callsign() + ": VF-1 TRANSFORM -> FIGHTER INTERCEPT MODE!", 3.2)
        elif clamped == MODE_GUARDIAN:
            self._set_comms(self.pilot_callsign() + ": VF-1 TRANSFORM -> GUARDIAN VTOL HOVER MODE!", 3.2)
        else:
            self._set_comms(self.pilot_callsign() + ": VF-1 TRANSFORM -> BATTLOID COMBAT MODE!", 3.2)

    def toggle_fast_pack_system(mut self):
        self.fast_packs = not self.fast_packs
        self.sound_events |= EVT_TRANSFORM
        if self.fast_packs:
            self.missiles_ready = 60
            self._set_comms("NP-BP-01 SUPER/STRIKE FAST PACKS ARMED (60 MISSILES + RO-X2A CANNON)!", 3.6)
        else:
            self.missiles_ready = min(self.missiles_ready, 36)
            self._set_comms("SUPER FAST PACKS JETTISONED — CLEAN ATMOSPHERIC AIRFRAME!", 3.6)

    def set_mission(mut self, theater_idx: Int):
        var t = max(0, min(2, theater_idx))
        for m in range(MAX_MISSILES):
            self.mis_active[m] = False
        for b in range(MAX_BOLTS):
            self.bolt_active[b] = False
        self.reflex_beam_t = 0.0
        self.daedalus_attack_t = 0.0
        self.daedalus_cd = 0.0
        self.sdf1_storm_mode = False
        self.sdf1_storm_morph = SpringFloat(0.0)
        if t == THEATER_MACROSS_ISLAND:
            self.fast_packs = False
            self.missiles_ready = 36
            self.spawn_wave(1)
        elif t == THEATER_HIGH_ORBIT:
            self.fast_packs = True
            self.missiles_ready = 60
            self.spawn_wave(3)
        else:
            self.fast_packs = True
            self.missiles_ready = 60
            self.reflex_charge = max(self.reflex_charge, Float32(100.0))
            self.spawn_wave(5)

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
        fb.max_radius = Float32(8.5) if count >= 60 else (Float32(4.4) if count >= 30 else Float32(2.6))
        fb.duration = Float32(1.05) if count >= 60 else (Float32(0.72) if count >= 30 else Float32(0.42))
        fb.life = fb.duration
        self.fireballs[self.fireball_cursor] = fb
        self.fireball_cursor = (self.fireball_cursor + 1) % MAX_FIREBALLS
        var dist = (center - self.pos).length()
        if dist < Float32(160.0):
            self.cam_shake = min(Float32(2.6), self.cam_shake + (Float32(160.0) - dist) / Float32(85.0))

    def spawn_wave(mut self, wave_num: Int):
        self.wave = wave_num
        if wave_num <= 2:
            self.theater = THEATER_MACROSS_ISLAND
        elif wave_num <= 4:
            if self.theater == THEATER_MACROSS_ISLAND and not self.fast_packs:
                self.fast_packs = True
                self.missiles_ready = 60
            self.theater = THEATER_HIGH_ORBIT
        else:
            if not self.fast_packs:
                self.fast_packs = True
                self.missiles_ready = 60
            self.theater = THEATER_BOSS_FLEET

        var pod_count = min(MAX_ENEMIES - 2, 5 + wave_num * 3)
        var queadluun_count = 0 if wave_num <= 2 else min(4, wave_num - 1)
        var glaug_count = min(4, (wave_num + 1) // 2)
        var fwd = self.forward_vec()
        var right = self.right_vec()

        for i in range(MAX_ENEMIES):
            self.enemies[i] = ZentraediUnit()
        self.total_locks = 0
        self.boss_lock_marks = 0

        for i in range(pod_count):
            var kind = ENEMY_REGULT
            if i < queadluun_count:
                kind = ENEMY_QUEADLUUN
            elif i < queadluun_count + glaug_count:
                kind = ENEMY_GLAUG

            var angle = (Float32(i) / Float32(max(1, pod_count)) - Float32(0.5)) * Float32(1.85)
            var dist = Float32(95.0) + Float32((i * 19) % 65)
            var alt = Float32(16.0) + Float32((i * 13) % 36)
            var spawn_p = self.pos + fwd * (cos(angle) * dist) + right * (sin(angle) * dist)
            spawn_p.y = alt
            var u = ZentraediUnit()
            u.active = True
            u.kind = kind
            u.pos = spawn_p
            u.yaw = self.yaw + Float32(3.14159)
            u.max_hp = Float32(260.0) if kind == ENEMY_QUEADLUUN else (Float32(190.0) if kind == ENEMY_GLAUG else Float32(75.0))
            u.hp = u.max_hp
            u.fire_cd = Float32(0.55) + Float32(i % 5) * Float32(0.32)
            u.missile_cd = Float32(2.4) + Float32(i % 4) * Float32(0.85)
            u.stride_phase = Float32(i) * Float32(0.9)
            self.enemies[i] = u

        if self.theater == THEATER_BOSS_FLEET:
            self.boss_active = True
            var hp_scale = Float32(1.0) + Float32(max(0, wave_num - 5)) * Float32(0.22)
            self.boss_max_hp = Float32(1800.0) * hp_scale
            self.boss_hp = self.boss_max_hp
            self.boss_pos = self.pos + fwd * Float32(178.0) + right * Float32(26.0) + Vec3(0.0, 28.0, 0.0)
            self.boss_vel = Vec3()
            self.boss_yaw = self.yaw + Float32(3.14159)
            self.boss_fire_cd = Float32(1.8)
            self.boss_salvo_cd = Float32(4.2)
            self.boss_flash_t = Float32(0.0)
            self._reset_boss_subsystems(hp_scale)
            self._set_comms(
                "ALERT: ZENTRAEDI THUVERL-SALAN DESTROYER (4 SUBSYSTEMS) & QUADRONO ACES WARPING IN!",
                4.6,
            )
        elif self.theater == THEATER_HIGH_ORBIT:
            self.boss_active = False
            self.boss_hp = 0.0
            self.skimming_hull = False
            self._set_comms(
                "SPACE FOLD COMPLETE — WAVE "
                + String(wave_num)
                + " HIGH ORBIT DEFENSE (FAST PACKS + QUEADLUUN ACES)!",
                4.6,
            )
        else:
            self.boss_active = False
            self.boss_hp = 0.0
            self.skimming_hull = False
            self._set_comms(
                "MACROSS ISLAND RADAR: ZENTRAEDI WAVE "
                + String(wave_num)
                + " INBOUND ("
                + String(pod_count)
                + " HOSTILES)!",
                4.6,
            )

    def active_enemy_count(self) -> Int:
        var n = 0
        for i in range(MAX_ENEMIES):
            if self.enemies[i].active:
                n += 1
        return n

    def active_hostile_count(self) -> Int:
        return self.active_enemy_count() + (1 if self.boss_active else 0)

    def spawn_bolt(mut self, origin: Vec3, vel: Vec3, team: Int, life: Float32 = 1.6, kind: UInt8 = 0):
        for i in range(MAX_BOLTS):
            if not self.bolt_active[i]:
                self.bolt_active[i] = True
                self.bolt_team[i] = team
                self.bolt_kind[i] = kind if kind != 0 or team == TEAM_UN_SPACY else BOLT_ZENT_PLASMA
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
                self.mis_life.set(i, Float32(5.4))
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
        if self.boss_active:
            for s in range(len(self.boss_subsystems)):
                if self.boss_subsystems[s].active and self.boss_subsystems[s].lock_marks > 0:
                    for _ in range(self.boss_subsystems[s].lock_marks):
                        targets.append(SUBSYS_TARGET_BASE + s)
                    self.boss_subsystems[s].lock_marks = 0
            if self.boss_lock_marks > 0:
                for _ in range(self.boss_lock_marks):
                    targets.append(BOSS_TARGET_ID)
                self.boss_lock_marks = 0
        self.total_locks = 0

        # If quick-fired without pre-locking, distribute across all enemies, Destroyer subsystems & boss in forward hemisphere
        if len(targets) == 0:
            var fwd = self.forward_vec()
            if self.boss_active and (self.boss_pos - self.pos).normalized().dot(fwd) > Float32(0.10):
                for s in range(len(self.boss_subsystems)):
                    if self.boss_subsystems[s].active:
                        targets.append(SUBSYS_TARGET_BASE + s)
                targets.append(BOSS_TARGET_ID)
            for i in range(MAX_ENEMIES):
                if self.enemies[i].active:
                    var to_e = (self.enemies[i].pos - self.pos).normalized()
                    if to_e.dot(fwd) > Float32(0.15):
                        targets.append(i)
            if len(targets) == 0:
                if self.boss_active:
                    targets.append(BOSS_TARGET_ID)
                for i in range(MAX_ENEMIES):
                    if self.enemies[i].active:
                        targets.append(i)

        if len(targets) == 0:
            return

        var salvo_cap = max_count + (12 if self.fast_packs else 0)
        var count = min(self.missiles_ready, max(8, min(salvo_cap, len(targets) * 4)))
        var fwd = self.forward_vec()
        var right = self.right_vec()
        var up = right.cross(fwd).normalized()

        for m in range(count):
            var tid = targets[m % len(targets)]
            var sgn = Float32(-1.0) if (m & 1) == 0 else Float32(1.0)
            var ring_ang = Float32(m) * Float32(0.785398)
            var pod_offset = right * (sgn * Float32(1.65)) + up * (Float32(0.85) if self.fast_packs else Float32(0.55))
            # Wild outward radial blossom velocity (signature Macross missile pod burst!)
            var radial = right * (cos(ring_ang) * Float32(34.0)) + up * (sin(ring_ang) * Float32(28.0) + Float32(10.0))
            var launch_vel = self.vel + fwd * Float32(54.0) + radial
            self.spawn_micro_missile(self.pos + pod_offset, launch_vel, TEAM_UN_SPACY, tid)

        self.missiles_ready -= count
        self.sound_events |= EVT_SALVO
        self._set_comms("FOX THREE! ITANO CIRCUS SALVO (" + String(count) + " MICRO-MISSILES) AWAY!", 2.8)

    def _execute_daedalus_attack(mut self):
        """Unleashes the iconic SLV-111 Daedalus Pinpoint-Barrier ram & interior missile saturation burst!"""
        self.daedalus_attack_t = Float32(2.3)
        self.daedalus_cd = Float32(7.0)
        self.cam_shake = Float32(2.5)
        self.sound_events |= EVT_EXPLODE | EVT_SALVO

        var ram_origin = Vec3(
            self.pos.x * Float32(0.25) - Float32(92.0),
            Float32(78.0),
            self.pos.z + Float32(118.0),
        )
        var tgt = self.boss_pos if self.boss_active else (self.pos + self.forward_vec() * Float32(145.0))
        if self.boss_active:
            for s in range(len(self.boss_subsystems)):
                if self.boss_subsystems[s].active:
                    var spos = self.boss_subsystem_world_pos(s)
                    self._damage_boss_subsystem(s, Float32(190.0), spos)
                    break
            var bp = self.boss_pos
            self._damage_boss(Float32(460.0), bp)
            self.emit_explosion(bp, 64, 48.0)

        # Launch 16 internal Daedalus ramp micro-missiles!
        var burst_dir = (tgt - ram_origin).normalized()
        for k in range(16):
            var ang = Float32(k) * Float32(0.3927)
            var spread = Vec3(cos(ang) * Float32(18.0), sin(ang) * Float32(18.0) + Float32(6.0), 0.0)
            var tid = BOSS_TARGET_ID if self.boss_active else (k % max(1, MAX_ENEMIES))
            if not self.boss_active:
                tid = -1
                for ei in range(MAX_ENEMIES):
                    var cand = (ei + k) % MAX_ENEMIES
                    if self.enemies[cand].active:
                        tid = cand
                        break
            if tid >= 0:
                self.spawn_micro_missile(ram_origin, burst_dir * Float32(68.0) + spread, TEAM_UN_SPACY, tid)

        self._set_comms(
            "CLAUDIA GRANT: SDF-1 STORM ATTACKER — DAEDALUS PINPOINT BARRIER ATTACK!!",
            4.2,
        )

    def trigger_storm_daedalus(mut self) -> Bool:
        """Transforms SDF-1 between Cruiser & towering humanoid Storm Attacker, or triggers the Daedalus Attack!"""
        if not self.sdf1_storm_mode:
            self.sdf1_storm_mode = True
            self.sound_events |= EVT_TRANSFORM
            self._set_comms(
                "CAPTAIN GLOVAL: SDF-1 MODULAR TRANSFORMATION — STORM ATTACKER MODE ENGAGED!",
                3.8,
            )
            return True
        if self.active_hostile_count() > 0 and self.daedalus_cd <= Float32(0.0):
            self._execute_daedalus_attack()
            return True
        self.sdf1_storm_mode = False
        self.sound_events |= EVT_TRANSFORM
        self._set_comms("CAPTAIN GLOVAL: SDF-1 RETURNING TO FORTRESS CRUISER CONFIGURATION.", 3.2)
        return True

    def trigger_reflex_cannon(mut self) -> Bool:
        """Unleashes SDF-1 Macross's twin Main Boom Reflex Cannons across the combat sector."""
        if self.reflex_charge < Float32(100.0) or self.reflex_beam_t > Float32(0.0):
            return False

        self.reflex_charge = Float32(0.0)
        self.reflex_beam_t = Float32(2.4)
        self.sound_events |= EVT_REFLEX
        self.cam_shake = Float32(2.2)

        # Match SDF-1 Macross cruiser forward boom muzzles in high orbit / island support position
        var sdf_center = Vec3(
            self.pos.x * Float32(0.25) - Float32(52.0),
            Float32(66.0),
            self.pos.z + Float32(275.0),
        )
        self.reflex_origin_l = sdf_center + Vec3(-48.0, 2.5, -72.0)
        self.reflex_origin_r = sdf_center + Vec3(-33.0, 2.5, -82.0)

        # Converge on Thuverl-Salan Destroyer prow if present, else primary Zentraedi formation!
        var tgt = self.pos + self.forward_vec() * Float32(155.0)
        if self.boss_active:
            var toward_player = (self.pos - self.boss_pos).normalized()
            tgt = self.boss_pos + toward_player * Float32(34.0) + Vec3(0.0, 2.5, 0.0)
        else:
            var best_hp = Float32(-1.0)
            for i in range(MAX_ENEMIES):
                if self.enemies[i].active and self.enemies[i].hp > best_hp:
                    best_hp = self.enemies[i].hp
                    tgt = self.enemies[i].pos
        self.reflex_target_pos = tgt

        # Cataclysmic particle beam convergence damage!
        var beam_mid = (self.reflex_origin_l + self.reflex_origin_r) * Float32(0.5)
        var beam_axis = tgt - beam_mid
        var beam_len = max(Float32(1.0), beam_axis.length())
        var beam_dir = beam_axis * (Float32(1.0) / beam_len)

        if self.boss_active:
            var bp = self.boss_pos
            self._damage_boss(Float32(680.0), bp)
            self.emit_explosion(bp, 68, 52.0)

        for i in range(MAX_ENEMIES):
            if not self.enemies[i].active:
                continue
            var rel = self.enemies[i].pos - beam_mid
            var proj = max(Float32(0.0), min(beam_len + Float32(80.0), rel.dot(beam_dir)))
            var closest = beam_mid + beam_dir * proj
            var perp_d = (self.enemies[i].pos - closest).length()
            if perp_d < Float32(52.0) or (self.enemies[i].pos - tgt).length() < Float32(88.0):
                var epos = self.enemies[i].pos
                self._damage_enemy(i, Float32(340.0), epos)

        var mode_tag = "STORM ATTACKER " if self.sdf1_storm_morph.value > Float32(0.5) else ""
        self._set_comms(
            "CAPTAIN GLOVAL: SDF-1 " + mode_tag + "MAIN REFLEX CANNON — CONVERGING BEAM FIRE!!",
            4.2,
        )
        return True

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
        if self.daedalus_attack_t > Float32(0.0):
            self.daedalus_attack_t = max(Float32(0.0), self.daedalus_attack_t - dt)
        if self.daedalus_cd > Float32(0.0):
            self.daedalus_cd = max(Float32(0.0), self.daedalus_cd - dt)

        # Squadron Pilot Perks + Hull Trench Skimming Bonus
        var reflex_rate = Float32(6.8) if self.pilot_id == PILOT_SKULL_LEADER else Float32(5.2)
        self.skimming_hull = self.boss_active and (self.pos - self.boss_pos).length() < Float32(66.0)
        if self.skimming_hull:
            reflex_rate += Float32(7.5)
        if self.reflex_beam_t > Float32(0.0):
            self.reflex_beam_t = max(Float32(0.0), self.reflex_beam_t - dt)
        else:
            self.reflex_charge = min(Float32(100.0), self.reflex_charge + dt * reflex_rate)

        for fi in range(MAX_FIREBALLS):
            if self.fireballs[fi].active:
                self.fireballs[fi].life -= dt
                if self.fireballs[fi].life <= Float32(0.0):
                    self.fireballs[fi].active = False

        # 0. Campaign Theater, Squadron Pilot, FAST Packs, Storm Attacker / Daedalus & Reflex Cannon
        if ctrl.cycle_mission:
            self.set_mission((self.theater + 1) % 3)
        if ctrl.cycle_pilot:
            self.cycle_pilot()
        if ctrl.toggle_fast_packs:
            self.toggle_fast_pack_system()
        if ctrl.toggle_storm_attacker:
            _ = self.trigger_storm_daedalus()
        if ctrl.fire_reflex_cannon:
            _ = self.trigger_reflex_cannon()

        self.sdf1_storm_morph.update(Float32(1.0) if self.sdf1_storm_mode else Float32(0.0), Float32(0.28), dt)

        # 1. Mode Switch & Smooth VF-1 Transformation Springs
        if ctrl.request_mode >= 0:
            self.set_mode(ctrl.request_mode)
        var goal_fg = Float32(0.0) if self.mode == MODE_FIGHTER else Float32(1.0)
        var goal_gb = Float32(1.0) if self.mode == MODE_BATTLOID else Float32(0.0)
        self.morph_fg.update(goal_fg, Float32(0.14), dt)
        self.morph_gb.update(goal_gb, Float32(0.14), dt)

        # 2. Mode-Specific Veritech Flight / VTOL / Battloid Thruster Physics (+ Max Sterling Agility/Speed Perk!)
        var turn_mult = Float32(1.18) if self.pilot_id == PILOT_MAX_STERLING else Float32(1.0)
        var spd_mult = Float32(1.12) if self.pilot_id == PILOT_MAX_STERLING else Float32(1.0)
        var turn_rate = (
            Float32(1.55) if self.mode == MODE_FIGHTER else (Float32(2.15) if self.mode == MODE_GUARDIAN else Float32(2.65))
        ) * turn_mult
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
        var up = right.cross(fwd).normalized()
        var boosting = ctrl.boost_held and self.boost_fuel > Float32(2.0)
        var fuel_burn = Float32(16.0) if self.fast_packs else Float32(24.0)
        if boosting:
            self.boost_fuel = max(Float32(0.0), self.boost_fuel - dt * fuel_burn)
        else:
            self.boost_fuel = min(Float32(100.0), self.boost_fuel + dt * Float32(15.0))

        if self.mode == MODE_FIGHTER:
            var cruise = (Float32(72.0) if self.fast_packs else Float32(62.0)) * spd_mult
            var base_spd = cruise + ctrl.throttle_in * Float32(26.0)
            if boosting:
                base_spd = (Float32(126.0) if self.fast_packs else Float32(104.0)) * spd_mult
            var target_vel = fwd * max(Float32(34.0), base_spd)
            if self.barrel_roll_t > Float32(0.0):
                target_vel = target_vel + right * (self.barrel_roll_dir * Float32(38.0))
            self.vel = self.vel + (target_vel - self.vel) * min(Float32(1.0), dt * Float32(5.5))
        elif self.mode == MODE_GUARDIAN:
            var spd = ((Float32(56.0) if self.fast_packs else Float32(48.0)) if not boosting else (Float32(88.0) if self.fast_packs else Float32(72.0))) * spd_mult
            var hover_vel = (
                fwd * (ctrl.throttle_in * spd)
                + right * (ctrl.strafe_x * spd * Float32(0.95))
                + Vec3(0.0, ctrl.lift_y * Float32(34.0), 0.0)
            )
            self.vel = self.vel + (hover_vel - self.vel) * min(Float32(1.0), dt * Float32(6.5))
        else:
            # Battloid 360-degree omni-thruster combat + Pinpoint Barrier Shield on SHIFT
            var spd = (Float32(42.0) if self.fast_packs else Float32(36.0)) * spd_mult
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

        # Emit cyan VF-1 thermonuclear afterburner sparks (+ dorsal FAST Pack booster plumes!)
        self.emit_spark(
            self.pos - fwd * Float32(3.2) + right * Float32(-0.55),
            -fwd * Float32(24.0), Float32(0.18), 0,
        )
        self.emit_spark(
            self.pos - fwd * Float32(3.2) + right * Float32(0.55),
            -fwd * Float32(24.0), Float32(0.18), 0,
        )
        if self.fast_packs:
            self.emit_spark(
                self.pos - fwd * Float32(2.2) + up * Float32(0.65) + right * Float32(-0.48),
                -fwd * Float32(32.0), Float32(0.20), 0,
            )
            self.emit_spark(
                self.pos - fwd * Float32(2.2) + up * Float32(0.65) + right * Float32(0.48),
                -fwd * Float32(32.0), Float32(0.20), 0,
            )

        # 3. Weapons: GU-11 55mm Gunpod, RO-X2A Strike Beam Cannon & Itano Circus Multi-Lock Radar
        self.gun_cd = max(Float32(0.0), self.gun_cd - dt)
        self.gun_flash_t = max(Float32(0.0), self.gun_flash_t - dt)
        self.strike_cd = max(Float32(0.0), self.strike_cd - dt)
        self.strike_flash_t = max(Float32(0.0), self.strike_flash_t - dt)
        if self.gun_reload_t > Float32(0.0):
            self.gun_reload_t = max(Float32(0.0), self.gun_reload_t - dt)
            if self.gun_reload_t <= Float32(0.0):
                self.gun_ammo = 200

        var aim_dir = fwd
        var assist_cone = Float32(0.925) if self.pilot_id == PILOT_MAX_STERLING else Float32(0.955)
        var assist_id = self.nearest_reticle_enemy(assist_cone)
        if assist_id >= 0:
            var lead_pos = self.enemies[assist_id].pos + self.enemies[assist_id].vel * (
                (self.enemies[assist_id].pos - self.pos).length() / Float32(260.0)
            )
            aim_dir = (lead_pos - self.pos).normalized()
        elif self.boss_active:
            var best_sub = -1
            var best_sub_d = Float32(0.90)
            for s in range(len(self.boss_subsystems)):
                if self.boss_subsystems[s].active:
                    var sp = self.boss_subsystem_world_pos(s)
                    var d = (sp - self.pos).normalized().dot(fwd)
                    if d > best_sub_d:
                        best_sub_d = d
                        best_sub = s
            if best_sub >= 0:
                aim_dir = (self.boss_subsystem_world_pos(best_sub) - self.pos).normalized()
            elif (self.boss_pos - self.pos).normalized().dot(fwd) > Float32(0.92):
                aim_dir = (self.boss_pos - self.pos).normalized()

        var rof_scale = Float32(0.78) if self.pilot_id == PILOT_MIRIYA_ACE else Float32(1.0)
        if ctrl.fire_gunpod and self.gun_cd <= Float32(0.0) and self.gun_ammo > 0:
            var rof = (Float32(0.055) if self.mode == MODE_BATTLOID else Float32(0.078)) * rof_scale
            self.gun_cd = rof
            self.gun_flash_t = Float32(0.045)
            self.gun_ammo -= 1
            if self.gun_ammo == 0:
                self.gun_reload_t = Float32(1.1)
            self.spawn_bolt(self.pos + fwd * Float32(2.8), self.vel + aim_dir * Float32(260.0), TEAM_UN_SPACY, Float32(1.35), BOLT_UN_GUNPOD)
            self.sound_events |= EVT_GUNPOD

        # Dorsal Starboard RO-X2A Dual-Barrel Strike Beam Cannon when Super/Strike FAST Packs are equipped!
        if ctrl.fire_gunpod and self.fast_packs and self.strike_cd <= Float32(0.0):
            self.strike_cd = Float32(0.16) * rof_scale
            self.strike_flash_t = Float32(0.075)
            var s_origin = self.pos + right * Float32(0.48) + up * Float32(0.84) + fwd * Float32(2.1)
            self.spawn_bolt(s_origin - right * Float32(0.08), self.vel + aim_dir * Float32(315.0), TEAM_UN_SPACY, Float32(1.5), BOLT_UN_STRIKE_BEAM)
            self.spawn_bolt(s_origin + right * Float32(0.08), self.vel + aim_dir * Float32(315.0), TEAM_UN_SPACY, Float32(1.5), BOLT_UN_STRIKE_BEAM)

        # Missile regeneration & multi-lock painting
        var mis_cap = self.max_missiles()
        var regen_period = Float32(0.11) if self.fast_packs else Float32(0.18)
        if self.missiles_ready < mis_cap:
            self.missile_regen_t += dt
            if self.missile_regen_t >= regen_period:
                self.missile_regen_t = 0.0
                self.missiles_ready += 1

        var sweep_period = Float32(0.046) if self.pilot_id == PILOT_MIRIYA_ACE else Float32(0.070)
        var lock_limit = min(36 if self.fast_packs else 24, self.missiles_ready)
        if ctrl.lock_held and self.missiles_ready > 0:
            self.lock_sweep_t += dt
            if self.lock_sweep_t >= sweep_period and self.total_locks < lock_limit:
                self.lock_sweep_t = 0.0
                var painted = False
                for i in range(MAX_ENEMIES):
                    var idx = (self.tick_count + i) % MAX_ENEMIES
                    if self.enemies[idx].active and self.enemies[idx].lock_marks < 4:
                        var d = (self.enemies[idx].pos - self.pos).normalized().dot(fwd)
                        if d > Float32(0.68):
                            self.enemies[idx].lock_marks += 1
                            self.total_locks += 1
                            self.sound_events |= EVT_LOCK
                            painted = True
                            break
                if not painted and self.boss_active:
                    for s in range(len(self.boss_subsystems)):
                        var sidx = (self.tick_count + s) % len(self.boss_subsystems)
                        if self.boss_subsystems[sidx].active and self.boss_subsystems[sidx].lock_marks < 3:
                            var sp = self.boss_subsystem_world_pos(sidx)
                            if (sp - self.pos).normalized().dot(fwd) > Float32(0.62):
                                self.boss_subsystems[sidx].lock_marks += 1
                                self.total_locks += 1
                                self.sound_events |= EVT_LOCK
                                painted = True
                                break
                if not painted and self.boss_active and self.boss_lock_marks < 8:
                    var db = (self.boss_pos - self.pos).normalized().dot(fwd)
                    if db > Float32(0.60):
                        self.boss_lock_marks += 1
                        self.total_locks += 1
                        self.sound_events |= EVT_LOCK

        if ctrl.launch_salvo and self.total_locks > 0:
            self.launch_itano_salvo(min(36 if self.fast_packs else 30, self.total_locks * 2))
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
            self.wing_fire_cd = Float32(1.55)
            var fired_w = False
            for i in range(MAX_ENEMIES):
                if self.enemies[i].active and (self.enemies[i].pos - self.wing_pos).length() < Float32(195.0):
                    var wdir = (self.enemies[i].pos - self.wing_pos).normalized()
                    self.spawn_micro_missile(
                        self.wing_pos + Vec3(0.0, 1.5, 0.0),
                        wdir * Float32(45.0) + Vec3(0.0, 18.0, 0.0),
                        TEAM_UN_SPACY, i,
                    )
                    self.wing_gun_flash = Float32(0.12)
                    fired_w = True
                    break
            if not fired_w and self.boss_active:
                var bdir = (self.boss_pos - self.wing_pos).normalized()
                self.spawn_micro_missile(
                    self.wing_pos + Vec3(0.0, 1.5, 0.0),
                    bdir * Float32(48.0) + Vec3(0.0, 20.0, 0.0),
                    TEAM_UN_SPACY, BOSS_TARGET_ID,
                )
                self.wing_gun_flash = Float32(0.12)

        # 5. Update Zentraedi Tactical AI (Regults, Glaug Officers & Queadluun-Rau Aces)
        for i in range(MAX_ENEMIES):
            if not self.enemies[i].active:
                continue
            var e = self.enemies[i]
            var to_player = self.pos - e.pos
            var dist = max(Float32(1.0), to_player.length())
            var dir_p = to_player * (Float32(1.0) / dist)
            e.yaw = atan2(dir_p.x, dir_p.z)
            e.pitch = max(Float32(-0.5), min(Float32(0.5), dir_p.y))
            e.stride_phase += dt * (Float32(8.5) if e.kind == ENEMY_QUEADLUUN else Float32(6.5))
            e.firing_flash = max(Float32(0.0), e.firing_flash - dt)

            var tang = Vec3(dir_p.z, Float32(0.0), -dir_p.x) * (Float32(1.0) if (i & 1) == 0 else Float32(-1.0))
            var desired_spd = Float32(44.0) if e.kind == ENEMY_QUEADLUUN else (Float32(34.0) if e.kind == ENEMY_GLAUG else Float32(28.0))
            var approach = dir_p * (desired_spd if dist > Float32(62.0) else (Float32(-22.0) if dist < Float32(32.0) else Float32(0.0)))
            var weave_amp = Float32(16.0) if e.kind == ENEMY_QUEADLUUN else Float32(9.0)
            var weave_y = sin(self.sim_time * (Float32(2.9) if e.kind == ENEMY_QUEADLUUN else Float32(1.8)) + Float32(i)) * weave_amp
            var strafe_spd = Float32(32.0) if e.kind == ENEMY_QUEADLUUN else Float32(22.0)
            var target_v = approach + tang * strafe_spd + Vec3(0.0, weave_y, 0.0)
            e.vel = e.vel + (target_v - e.vel) * min(Float32(1.0), dt * (Float32(4.2) if e.kind == ENEMY_QUEADLUUN else Float32(2.8)))
            e.pos = e.pos + e.vel * dt
            if e.pos.y < Float32(3.5):
                e.pos.y = Float32(3.5)

            var fire_bolt_now = False
            e.fire_cd -= dt
            if e.fire_cd <= Float32(0.0) and dist < Float32(205.0):
                e.fire_cd = Float32(0.95) if e.kind == ENEMY_QUEADLUUN else (Float32(1.25) if e.kind == ENEMY_GLAUG else Float32(1.85))
                e.firing_flash = Float32(0.14)
                fire_bolt_now = True

            var fire_missile_now = False
            if e.kind == ENEMY_GLAUG or e.kind == ENEMY_QUEADLUUN:
                e.missile_cd -= dt
                if e.missile_cd <= Float32(0.0) and dist < Float32(235.0):
                    e.missile_cd = Float32(3.8) if e.kind == ENEMY_QUEADLUUN else Float32(5.2)
                    fire_missile_now = True

            self.enemies[i] = e
            if fire_bolt_now:
                if e.kind == ENEMY_QUEADLUUN:
                    self.spawn_bolt(e.pos + Vec3(-1.1, 1.2, 0.0) + dir_p * Float32(2.6), dir_p * Float32(168.0), TEAM_ZENTRAEDI, Float32(1.8), BOLT_ZENT_PLASMA)
                    self.spawn_bolt(e.pos + Vec3(1.1, 1.2, 0.0) + dir_p * Float32(2.6), dir_p * Float32(168.0), TEAM_ZENTRAEDI, Float32(1.8), BOLT_ZENT_PLASMA)
                else:
                    self.spawn_bolt(e.pos + dir_p * Float32(3.0), dir_p * Float32(145.0), TEAM_ZENTRAEDI, Float32(1.8), BOLT_ZENT_PLASMA)
            if fire_missile_now:
                self.spawn_micro_missile(
                    e.pos + Vec3(0.0, 2.4, 0.0),
                    dir_p * Float32(44.0) + Vec3(0.0, 18.0, 0.0),
                    TEAM_ZENTRAEDI, 0,
                )
                if e.kind == ENEMY_QUEADLUUN:
                    self.spawn_micro_missile(
                        e.pos + Vec3(1.2, 2.6, 0.0),
                        dir_p * Float32(44.0) + Vec3(8.0, 22.0, 0.0),
                        TEAM_ZENTRAEDI, 0,
                    )

        # 5b. Update Thuverl-Salan Heavy Destroyer Capital Warship Boss
        if self.boss_active:
            self.boss_flash_t = max(Float32(0.0), self.boss_flash_t - dt)
            var to_p = self.pos - self.boss_pos
            var b_dist = max(Float32(1.0), to_p.length())
            var b_dir = to_p * (Float32(1.0) / b_dist)
            self.boss_yaw = atan2(b_dir.x, b_dir.z)
            var orbit_tang = Vec3(b_dir.z, Float32(0.0), -b_dir.x)
            var keep_d = Float32(14.0) if b_dist > Float32(185.0) else (Float32(-12.0) if b_dist < Float32(115.0) else Float32(0.0))
            var b_goal_vel = b_dir * keep_d + orbit_tang * Float32(15.0) + Vec3(0.0, (Float32(52.0) - self.boss_pos.y) * Float32(0.35), 0.0)
            self.boss_vel = self.boss_vel + (b_goal_vel - self.boss_vel) * min(Float32(1.0), dt * Float32(1.4))
            self.boss_pos = self.boss_pos + self.boss_vel * dt

            self.boss_fire_cd -= dt
            if self.boss_fire_cd <= Float32(0.0):
                self.boss_fire_cd = Float32(2.15)
                var turrets_fired = 0
                if self.boss_subsystems[SUBSYS_PORT_TURRET].active:
                    var p_muz = self.boss_subsystem_world_pos(SUBSYS_PORT_TURRET) + b_dir * Float32(6.0)
                    self.spawn_bolt(p_muz, b_dir * Float32(168.0), TEAM_ZENTRAEDI, Float32(2.4), BOLT_BOSS_HEAVY)
                    turrets_fired += 1
                if self.boss_subsystems[SUBSYS_STBD_TURRET].active:
                    var s_muz = self.boss_subsystem_world_pos(SUBSYS_STBD_TURRET) + b_dir * Float32(6.0)
                    self.spawn_bolt(s_muz, b_dir * Float32(168.0), TEAM_ZENTRAEDI, Float32(2.4), BOLT_BOSS_HEAVY)
                    turrets_fired += 1
                if turrets_fired > 0:
                    self.boss_flash_t = Float32(0.28)
                    # Capital-ship bombardment chips SDF-1 hull until turrets or Destroyer are eliminated!
                    self.sdf1_hull = max(Float32(15.0), self.sdf1_hull - Float32(0.6) * Float32(turrets_fired))

            self.boss_salvo_cd -= dt
            if self.boss_salvo_cd <= Float32(0.0) and b_dist < Float32(310.0):
                self.boss_salvo_cd = Float32(5.4)
                if self.boss_subsystems[SUBSYS_TORPEDO_BAY].active:
                    var bay_pos = self.boss_subsystem_world_pos(SUBSYS_TORPEDO_BAY)
                    for k in range(4):
                        var sx = Float32(-9.0) + Float32(k) * Float32(6.0)
                        self.spawn_micro_missile(
                            bay_pos + Vec3(sx, 2.0, 0.0),
                            b_dir * Float32(42.0) + Vec3(sx * Float32(0.8), 22.0, 0.0),
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
                if tid >= SUBSYS_TARGET_BASE and tid < SUBSYS_TARGET_BASE + BOSS_SUBSYS_COUNT and self.boss_active:
                    var sidx = tid - SUBSYS_TARGET_BASE
                    if self.boss_subsystems[sidx].active:
                        goal_pos = self.boss_subsystem_world_pos(sidx) + self.boss_vel * Float32(0.18)
                    else:
                        goal_pos = self.boss_pos + self.boss_vel * Float32(0.20)
                    has_goal = True
                elif tid == BOSS_TARGET_ID and self.boss_active:
                    goal_pos = self.boss_pos + self.boss_vel * Float32(0.20)
                    has_goal = True
                elif tid >= 0 and tid < MAX_ENEMIES and self.enemies[tid].active:
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
                var spd = Float32(122.0) if self.mis_team[m] == TEAM_UN_SPACY else Float32(78.0)
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

        # 7. Step Gunpod, Strike Beam Cannon & Plasma Bolts
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

        if self.active_hostile_count() == 0:
            self.hull = min(self.max_hull, self.hull + Float32(30.0))
            self.sdf1_hull = min(Float32(100.0), self.sdf1_hull + Float32(15.0))
            self.shield = Float32(100.0)
            self.missiles_ready = self.max_missiles()
            self.spawn_wave(self.wave + 1)

    def _resolve_combat_collisions(mut self):
        var bp = BroadphaseInput()
        var filt = CollisionFilter()
        var skim_mult = Float32(1.35) if self.skimming_hull else Float32(1.0)

        # Row 0: Player VF-1
        bp.add(self.pos - Vec3(3.2, 3.2, 3.2), self.pos + Vec3(3.2, 3.2, 3.2), True, True, filt)
        # Rows 1..MAX_ENEMIES: Zentraedi Mecha
        for i in range(MAX_ENEMIES):
            var r = Float32(4.6) if self.enemies[i].kind != ENEMY_REGULT else Float32(3.6)
            var ep = self.enemies[i].pos
            bp.add(ep - Vec3(r, r, r), ep + Vec3(r, r, r), self.enemies[i].active, True, filt)

        # Row `boss_row`: Zentraedi Thuverl-Salan Heavy Destroyer
        var boss_row = 1 + MAX_ENEMIES
        bp.add(
            self.boss_pos - Vec3(24.0, 14.0, 46.0),
            self.boss_pos + Vec3(24.0, 14.0, 46.0),
            self.boss_active,
            True,
            filt,
        )

        # Rows `sub_base .. sub_base + BOSS_SUBSYS_COUNT`: Destructible Warship Subsystems
        var sub_base = boss_row + 1
        for s in range(BOSS_SUBSYS_COUNT):
            var sp = self.boss_subsystem_world_pos(s)
            var s_act = self.boss_active and s < len(self.boss_subsystems) and self.boss_subsystems[s].active
            bp.add(sp - Vec3(7.5, 7.5, 7.5), sp + Vec3(7.5, 7.5, 7.5), s_act, True, filt)

        var mis_base = sub_base + BOSS_SUBSYS_COUNT
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
                    var dmg = Float32(12.0) if self.bolt_kind[bid] == BOLT_BOSS_HEAVY else Float32(6.0)
                    self._damage_player(dmg, Vec3(self.bolt_x.get(bid), self.bolt_y.get(bid), self.bolt_z.get(bid)))

            # Zentraedi Unit (1..MAX_ENEMIES) vs UN Spacy Missile / Bolt
            elif a >= 1 and a <= MAX_ENEMIES:
                var eid = a - 1
                if not self.enemies[eid].active:
                    continue
                if b >= mis_base and b < bolt_base:
                    var mid = b - mis_base
                    if self.mis_active[mid] and self.mis_team[mid] == TEAM_UN_SPACY:
                        self.mis_active[mid] = False
                        self._damage_enemy(eid, Float32(34.0) * skim_mult, Vec3(self.mis_x.get(mid), self.mis_y.get(mid), self.mis_z.get(mid)))
                elif b >= bolt_base:
                    var bid = b - bolt_base
                    if self.bolt_active[bid] and self.bolt_team[bid] == TEAM_UN_SPACY:
                        self.bolt_active[bid] = False
                        var bdmg = (Float32(28.0) if self.bolt_kind[bid] == BOLT_UN_STRIKE_BEAM else Float32(19.0)) * skim_mult
                        self._damage_enemy(eid, bdmg, Vec3(self.bolt_x.get(bid), self.bolt_y.get(bid), self.bolt_z.get(bid)))

            # Destructible Destroyer Subsystem (`sub_base .. mis_base - 1`) vs UN Spacy Missile / Bolt
            # Checked before main hull so targeted subsystem hits register cleanly!
            elif a >= sub_base and a < mis_base and self.boss_active:
                var sid = a - sub_base
                if sid < len(self.boss_subsystems) and self.boss_subsystems[sid].active:
                    if b >= mis_base and b < bolt_base:
                        var mid = b - mis_base
                        if self.mis_active[mid] and self.mis_team[mid] == TEAM_UN_SPACY:
                            self.mis_active[mid] = False
                            self._damage_boss_subsystem(
                                sid,
                                Float32(42.0) * skim_mult,
                                Vec3(self.mis_x.get(mid), self.mis_y.get(mid), self.mis_z.get(mid)),
                            )
                    elif b >= bolt_base:
                        var bid = b - bolt_base
                        if self.bolt_active[bid] and self.bolt_team[bid] == TEAM_UN_SPACY:
                            self.bolt_active[bid] = False
                            var bdmg = (Float32(34.0) if self.bolt_kind[bid] == BOLT_UN_STRIKE_BEAM else Float32(24.0)) * skim_mult
                            self._damage_boss_subsystem(
                                sid,
                                bdmg,
                                Vec3(self.bolt_x.get(bid), self.bolt_y.get(bid), self.bolt_z.get(bid)),
                            )

            # Thuverl-Salan Destroyer Boss (`boss_row`) vs UN Spacy Missile / Bolt
            elif a == boss_row and self.boss_active:
                if b >= mis_base and b < bolt_base:
                    var mid = b - mis_base
                    if self.mis_active[mid] and self.mis_team[mid] == TEAM_UN_SPACY:
                        self.mis_active[mid] = False
                        self._damage_boss(Float32(34.0) * skim_mult, Vec3(self.mis_x.get(mid), self.mis_y.get(mid), self.mis_z.get(mid)))
                elif b >= bolt_base:
                    var bid = b - bolt_base
                    if self.bolt_active[bid] and self.bolt_team[bid] == TEAM_UN_SPACY:
                        self.bolt_active[bid] = False
                        var bdmg = (Float32(28.0) if self.bolt_kind[bid] == BOLT_UN_STRIKE_BEAM else Float32(19.0)) * skim_mult
                        self._damage_boss(bdmg, Vec3(self.bolt_x.get(bid), self.bolt_y.get(bid), self.bolt_z.get(bid)))

            # Gunpod / Strike Cannon shooting down incoming Zentraedi missiles in mid-air!
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
        self.hull = max(Float32(10.0), self.hull - mitigated)  # Simulator safety floor keeps session continuous
        self.emit_explosion(hit_pos, 16, 20.0)
        if self.hull <= Float32(25.0):
            self._set_comms("WARNING: VERITECH HULL CRITICAL — DEPLOY PINPOINT BARRIER [3 + SHIFT]!", 3.0)

    def _damage_enemy(mut self, eid: Int, amount: Float32, hit_pos: Vec3):
        self.enemies[eid].hp -= amount
        self.emit_explosion(hit_pos, 14, 18.0)
        if self.enemies[eid].hp <= Float32(0.0):
            var pos = self.enemies[eid].pos
            var kind = self.enemies[eid].kind
            self.enemies[eid].active = False
            self.kills += 1
            var pts = 750 if kind == ENEMY_QUEADLUUN else (500 if kind == ENEMY_GLAUG else 200)
            self.score += pts
            self.reflex_charge = min(Float32(100.0), self.reflex_charge + Float32(14.0))
            self.emit_explosion(pos, 60 if kind != ENEMY_REGULT else 36, 40.0)
            if kind == ENEMY_QUEADLUUN:
                self._set_comms("SPLASH ONE QUEADLUUN-RAU QUADRONO ACE! (+750 PTS)", 3.2)
            elif kind == ENEMY_GLAUG:
                self._set_comms("SPLASH ONE ZENTRAEDI GLAUG OFFICER POD! (+500 PTS)", 3.0)

    def _damage_boss_subsystem(mut self, sub_idx: Int, amount: Float32, hit_pos: Vec3):
        if not self.boss_active or sub_idx < 0 or sub_idx >= len(self.boss_subsystems):
            return
        if not self.boss_subsystems[sub_idx].active:
            self._damage_boss(amount, hit_pos)
            return
        self.boss_subsystems[sub_idx].hp -= amount
        self.emit_explosion(hit_pos, 20, 24.0)
        self._damage_boss(amount * Float32(0.35), hit_pos)
        if self.boss_active and self.boss_subsystems[sub_idx].hp <= Float32(0.0):
            self.boss_subsystems[sub_idx].active = False
            self.boss_subsystems[sub_idx].hp = Float32(0.0)
            self.boss_subsystems[sub_idx].lock_marks = 0
            self.score += 400
            self.reflex_charge = min(Float32(100.0), self.reflex_charge + Float32(20.0))
            self.emit_explosion(self.boss_subsystem_world_pos(sub_idx), 52, 42.0)
            if sub_idx == SUBSYS_BRIDGE:
                self._set_comms("COMMAND BRIDGE DESTROYED — 2X CRITICAL DAMAGE EXPOSED ON DESTROYER HULL! (+400 PTS)", 4.2)
            elif sub_idx == SUBSYS_TORPEDO_BAY:
                self._set_comms("DESTROYER TORPEDO BAY SILENCED — ENEMY MISSILE SALVOS HALTED! (+400 PTS)", 3.6)
            else:
                self._set_comms(self.boss_subsystem_label(sub_idx) + " PLASMA BATTERY DESTROYED! (+400 PTS)", 3.4)

    def _damage_boss(mut self, amount: Float32, hit_pos: Vec3):
        if not self.boss_active:
            return
        var crit_mult = Float32(2.0) if self.boss_bridge_destroyed() else Float32(1.0)
        self.boss_hp -= amount * crit_mult
        self.emit_explosion(hit_pos, 22, 26.0)
        if self.boss_hp <= Float32(0.0):
            var bpos = self.boss_pos
            self.boss_active = False
            self.boss_hp = Float32(0.0)
            for s in range(len(self.boss_subsystems)):
                self.boss_subsystems[s].active = False
            self.kills += 1
            self.score += 2500
            self.reflex_charge = min(Float32(100.0), self.reflex_charge + Float32(35.0))
            for k in range(5):
                var off = Vec3(Float32(k - 2) * Float32(8.5), Float32((k & 1) * 4 - 2), Float32(k - 2) * Float32(14.0))
                self.emit_explosion(bpos + off, 68, 52.0)
            self._set_comms("DIRECT HIT! ZENTRAEDI THUVERL-SALAN DESTROYER VAPORIZED! (+2500 PTS)", 4.5)

    # ------------------------------------------------------------------
    # LZ4 Binary State Checkpoints, Deterministic Hash & Flint MCP Telemetry
    # ------------------------------------------------------------------
    def serialize_state(self, mut w: BlobWriter):
        w.begin(fourcc("RBTC"), 2)
        w.integer(self.theater)
        w.integer(self.wave)
        w.integer(self.score)
        w.integer(self.kills)
        w.f32(self.sim_time)
        w.integer(self.tick_count)
        w.u32(self.rng_state)

        w.f32(self.sdf1_hull)
        w.boolean(self.sdf1_storm_mode)
        w.f32(self.sdf1_storm_morph.value)
        w.f32(self.sdf1_storm_morph.velocity)
        w.f32(self.daedalus_attack_t)
        w.f32(self.daedalus_cd)
        w.f32(self.reflex_charge)
        w.f32(self.reflex_beam_t)
        w.vec3(self.reflex_origin_l)
        w.vec3(self.reflex_origin_r)
        w.vec3(self.reflex_target_pos)

        w.boolean(self.boss_active)
        w.vec3(self.boss_pos)
        w.vec3(self.boss_vel)
        w.f32(self.boss_yaw)
        w.f32(self.boss_hp)
        w.f32(self.boss_max_hp)
        w.f32(self.boss_fire_cd)
        w.f32(self.boss_salvo_cd)
        w.f32(self.boss_flash_t)
        w.integer(self.boss_lock_marks)
        w.boolean(self.skimming_hull)
        w.integer(len(self.boss_subsystems))
        for s in range(len(self.boss_subsystems)):
            ref sub = self.boss_subsystems[s]
            w.boolean(sub.active)
            w.f32(sub.hp)
            w.f32(sub.max_hp)
            w.vec3(sub.local_offset)
            w.integer(sub.lock_marks)

        w.integer(self.pilot_id)
        w.integer(self.mode)
        w.boolean(self.fast_packs)
        w.f32(self.morph_fg.value)
        w.f32(self.morph_fg.velocity)
        w.f32(self.morph_gb.value)
        w.f32(self.morph_gb.velocity)
        w.vec3(self.pos)
        w.vec3(self.vel)
        w.f32(self.yaw)
        w.f32(self.pitch)
        w.f32(self.roll.value)
        w.f32(self.roll.velocity)
        w.f32(self.barrel_roll_t)
        w.f32(self.barrel_roll_dir)
        w.f32(self.hull)
        w.f32(self.max_hull)
        w.f32(self.shield)
        w.f32(self.boost_fuel)
        w.f32(self.gun_cd)
        w.f32(self.gun_flash_t)
        w.f32(self.strike_cd)
        w.f32(self.strike_flash_t)
        w.integer(self.gun_ammo)
        w.f32(self.gun_reload_t)
        w.integer(self.missiles_ready)
        w.f32(self.missile_regen_t)
        w.f32(self.lock_sweep_t)
        w.integer(self.total_locks)

        w.vec3(self.wing_pos)
        w.f32(self.wing_yaw)
        w.f32(self.wing_pitch)
        w.f32(self.wing_morph_fg.value)
        w.f32(self.wing_morph_fg.velocity)
        w.f32(self.wing_morph_gb.value)
        w.f32(self.wing_morph_gb.velocity)
        w.f32(self.wing_fire_cd)
        w.f32(self.wing_gun_flash)

        w.vec3(self.cam_eye_spring.value)
        w.vec3(self.cam_eye_spring.velocity)
        w.vec3(self.cam_target_spring.value)
        w.vec3(self.cam_target_spring.velocity)
        w.f32(self.cam_shake)

        for i in range(MAX_ENEMIES):
            ref e = self.enemies[i]
            w.boolean(e.active)
            w.integer(e.kind)
            w.vec3(e.pos)
            w.vec3(e.vel)
            w.f32(e.yaw)
            w.f32(e.pitch)
            w.f32(e.hp)
            w.f32(e.max_hp)
            w.f32(e.fire_cd)
            w.f32(e.missile_cd)
            w.f32(e.stride_phase)
            w.integer(e.lock_marks)
            w.f32(e.firing_flash)

        for m in range(MAX_MISSILES):
            w.boolean(self.mis_active[m])
            if self.mis_active[m]:
                w.integer(self.mis_team[m])
                w.integer(self.mis_target[m])
                w.f32(self.mis_x.get(m))
                w.f32(self.mis_y.get(m))
                w.f32(self.mis_z.get(m))
                w.f32(self.mis_vx.get(m))
                w.f32(self.mis_vy.get(m))
                w.f32(self.mis_vz.get(m))
                w.f32(self.mis_phase.get(m))
                w.f32(self.mis_life.get(m))

        for b in range(MAX_BOLTS):
            w.boolean(self.bolt_active[b])
            if self.bolt_active[b]:
                w.integer(self.bolt_team[b])
                w.u8(self.bolt_kind[b])
                w.f32(self.bolt_x.get(b))
                w.f32(self.bolt_y.get(b))
                w.f32(self.bolt_z.get(b))
                w.f32(self.bolt_vx.get(b))
                w.f32(self.bolt_vy.get(b))
                w.f32(self.bolt_vz.get(b))
                w.f32(self.bolt_life.get(b))

        w.end()

    def deserialize_state(mut self, mut r: BlobReader) raises:
        var ver = r.enter(fourcc("RBTC"), 2)
        self.theater = r.integer()
        self.wave = r.integer()
        self.score = r.integer()
        self.kills = r.integer()
        self.sim_time = r.f32()
        self.tick_count = r.integer()
        self.rng_state = r.u32()

        self.sdf1_hull = r.f32()
        if ver >= 2:
            self.sdf1_storm_mode = r.boolean()
            self.sdf1_storm_morph.value = r.f32()
            self.sdf1_storm_morph.velocity = r.f32()
            self.daedalus_attack_t = r.f32()
            self.daedalus_cd = r.f32()
        else:
            self.sdf1_storm_mode = False
            self.sdf1_storm_morph = SpringFloat(0.0)
            self.daedalus_attack_t = 0.0
            self.daedalus_cd = 0.0
        self.reflex_charge = r.f32()
        self.reflex_beam_t = r.f32()
        self.reflex_origin_l = r.vec3()
        self.reflex_origin_r = r.vec3()
        self.reflex_target_pos = r.vec3()

        self.boss_active = r.boolean()
        self.boss_pos = r.vec3()
        self.boss_vel = r.vec3()
        self.boss_yaw = r.f32()
        self.boss_hp = r.f32()
        self.boss_max_hp = r.f32()
        self.boss_fire_cd = r.f32()
        self.boss_salvo_cd = r.f32()
        self.boss_flash_t = r.f32()
        self.boss_lock_marks = r.integer()
        if ver >= 2:
            self.skimming_hull = r.boolean()
            var nsub = r.integer()
            self.boss_subsystems.clear()
            for _ in range(nsub):
                var sub = DestroyerSubsystem()
                sub.active = r.boolean()
                sub.hp = r.f32()
                sub.max_hp = r.f32()
                sub.local_offset = r.vec3()
                sub.lock_marks = r.integer()
                self.boss_subsystems.append(sub)
        else:
            self.skimming_hull = False
            self._reset_boss_subsystems(1.0)

        if ver >= 2:
            self.pilot_id = r.integer()
        else:
            self.pilot_id = PILOT_SKULL_LEADER
        self.mode = r.integer()
        self.fast_packs = r.boolean()
        self.morph_fg.value = r.f32()
        self.morph_fg.velocity = r.f32()
        self.morph_gb.value = r.f32()
        self.morph_gb.velocity = r.f32()
        self.pos = r.vec3()
        self.vel = r.vec3()
        self.yaw = r.f32()
        self.pitch = r.f32()
        self.roll.value = r.f32()
        self.roll.velocity = r.f32()
        self.barrel_roll_t = r.f32()
        self.barrel_roll_dir = r.f32()
        self.hull = r.f32()
        self.max_hull = r.f32()
        self.shield = r.f32()
        self.boost_fuel = r.f32()
        self.gun_cd = r.f32()
        self.gun_flash_t = r.f32()
        self.strike_cd = r.f32()
        self.strike_flash_t = r.f32()
        self.gun_ammo = r.integer()
        self.gun_reload_t = r.f32()
        self.missiles_ready = r.integer()
        self.missile_regen_t = r.f32()
        self.lock_sweep_t = r.f32()
        self.total_locks = r.integer()

        self.wing_pos = r.vec3()
        self.wing_yaw = r.f32()
        self.wing_pitch = r.f32()
        self.wing_morph_fg.value = r.f32()
        self.wing_morph_fg.velocity = r.f32()
        self.wing_morph_gb.value = r.f32()
        self.wing_morph_gb.velocity = r.f32()
        self.wing_fire_cd = r.f32()
        self.wing_gun_flash = r.f32()

        self.cam_eye_spring.value = r.vec3()
        self.cam_eye_spring.velocity = r.vec3()
        self.cam_target_spring.value = r.vec3()
        self.cam_target_spring.velocity = r.vec3()
        self.cam_shake = r.f32()

        for i in range(MAX_ENEMIES):
            var e = ZentraediUnit()
            e.active = r.boolean()
            e.kind = r.integer()
            e.pos = r.vec3()
            e.vel = r.vec3()
            e.yaw = r.f32()
            e.pitch = r.f32()
            e.hp = r.f32()
            e.max_hp = r.f32()
            e.fire_cd = r.f32()
            e.missile_cd = r.f32()
            e.stride_phase = r.f32()
            e.lock_marks = r.integer()
            e.firing_flash = r.f32()
            self.enemies[i] = e

        for m in range(MAX_MISSILES):
            var act = r.boolean()
            self.mis_active[m] = act
            self.mis_trail_len[m] = 0
            if act:
                self.mis_team[m] = r.integer()
                self.mis_target[m] = r.integer()
                var mx = r.f32()
                var my = r.f32()
                var mz = r.f32()
                self.mis_x.set(m, mx)
                self.mis_y.set(m, my)
                self.mis_z.set(m, mz)
                self.mis_px.set(m, mx)
                self.mis_py.set(m, my)
                self.mis_pz.set(m, mz)
                self.mis_vx.set(m, r.f32())
                self.mis_vy.set(m, r.f32())
                self.mis_vz.set(m, r.f32())
                self.mis_phase.set(m, r.f32())
                self.mis_life.set(m, r.f32())
                self.mis_trail[m * MIS_TRAIL_LEN] = Vec3(mx, my, mz)
                self.mis_trail_len[m] = 1

        for b in range(MAX_BOLTS):
            var act = r.boolean()
            self.bolt_active[b] = act
            if act:
                self.bolt_team[b] = r.integer()
                self.bolt_kind[b] = r.u8()
                self.bolt_x.set(b, r.f32())
                self.bolt_y.set(b, r.f32())
                self.bolt_z.set(b, r.f32())
                self.bolt_vx.set(b, r.f32())
                self.bolt_vy.set(b, r.f32())
                self.bolt_vz.set(b, r.f32())
                self.bolt_life.set(b, r.f32())

        r.leave()
        self._set_comms("CHECKPOINT RESTORED — " + self.mission_title() + " (WAVE " + String(self.wave) + ")!", 3.5)

    def state_hash(self) -> UInt64:
        var w = BlobWriter()
        self.serialize_state(w)
        return fnv64(w.data)

    def save_checkpoint(self, path: String) -> Bool:
        var w = BlobWriter()
        self.serialize_state(w)
        return save_checkpoint_lz4(path, w)

    def load_checkpoint(mut self, path: String) raises:
        var r = load_checkpoint_lz4(path)
        self.deserialize_state(r)

    def mcp_state_json(self) -> String:
        return (
            '{"theater":'
            + String(self.theater)
            + ',"mission":'
            + json_quote(self.mission_title())
            + ',"wave":'
            + String(self.wave)
            + ',"pilot":'
            + json_quote(self.pilot_callsign())
            + ',"mode":'
            + json_quote(self.mode_name())
            + ',"fast_packs":'
            + ("true" if self.fast_packs else "false")
            + ',"hull":'
            + String(Int(self.hull))
            + ',"sdf1_hull":'
            + String(Int(self.sdf1_hull))
            + ',"sdf1_storm_mode":'
            + ("true" if self.sdf1_storm_mode else "false")
            + ',"reflex_charge":'
            + String(Int(self.reflex_charge))
            + ',"missiles_ready":'
            + String(self.missiles_ready)
            + ',"max_missiles":'
            + String(self.max_missiles())
            + ',"hostiles":'
            + String(self.active_hostile_count())
            + ',"boss_active":'
            + ("true" if self.boss_active else "false")
            + ',"boss_hp":'
            + String(Int(self.boss_hp))
            + ',"boss_subsystems_intact":'
            + String(self.intact_subsystem_count())
            + ',"skimming_hull":'
            + ("true" if self.skimming_hull else "false")
            + ',"score":'
            + String(self.score)
            + ',"kills":'
            + String(self.kills)
            + "}"
        )

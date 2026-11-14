"""Procedural 48 kHz Robotech Sound Synthesizer powered by `flint.platform.audio`.

Synthesizes iconic VF-1 Veritech & Zentraedi combat sound effects in pure Mojo
with zero external WAV files:
  0: VF-1 3-stage hydraulic transformation servo clank ("CHU-KANG-WHIRR!")
  1: GU-11 55mm tri-barrel Gunpod burst
  2: SDF-1 holographic radar lock-on chirp
  3: Itano Circus micro-missile pod salvo launch whoosh
  4: Zentraedi Battlepod plasma detonation boom
  5: Thermonuclear turbine continuous ambient hum (for `NativeAudio.set_drone`)
"""
from std.math import sin, cos
from flint.platform.audio import NativeAudio, WavClip

comptime SAMPLE_RATE: Int = 48000

comptime SFX_TRANSFORM: Int = 0
comptime SFX_GUNPOD: Int = 1
comptime SFX_LOCK_BEEP: Int = 2
comptime SFX_MISSILE_SALVO: Int = 3
comptime SFX_EXPLOSION: Int = 4
comptime SFX_TURBINE_DRONE: Int = 5
comptime SFX_REFLEX_CANNON: Int = 6
comptime SFX_RADIO_COMMS: Int = 7


@always_inline
def _hash_noise(sample_idx: Int) -> Float32:
    var x = UInt32(sample_idx * 747796405 + 2891336453)
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737
    x = (x >> 22) ^ x
    return (Float32(x & 0xFFFF) / Float32(32768.0)) - Float32(1.0)


def synth_transform_clip() -> WavClip:
    """Iconic 4-step VF-1 Valkyrie hydraulic transformation servo sound (0.42s)."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.42))
    clip.samples.reserve(total)
    var phase = Float32(0.0)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var step_idx = min(3, Int(t / Float32(0.105)))
        var local_t = t - Float32(step_idx) * Float32(0.105)
        var base_freqs = [Float32(310.0), Float32(465.0), Float32(620.0), Float32(880.0)]
        var freq = base_freqs[step_idx] + local_t * Float32(900.0)
        phase += freq * Float32(6.2831853) / Float32(SAMPLE_RATE)
        var env = max(Float32(0.0), Float32(1.0) - local_t * Float32(7.5))
        var clank = _hash_noise(i) * max(Float32(0.0), Float32(1.0) - local_t * Float32(38.0)) * Float32(0.65)
        var servo = (sin(phase) * Float32(0.55) + sin(phase * Float32(2.5)) * Float32(0.35)) * env
        var out = max(Float32(-1.0), min(Float32(1.0), servo + clank))
        clip.samples.append(Int16(Int(out * Float32(22000.0))))
    return clip^


def synth_gunpod_clip() -> WavClip:
    """GU-11 55mm rotary gunpod punch (0.085s)."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.085))
    clip.samples.reserve(total)
    var phase = Float32(0.0)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var freq = Float32(95.0) + Float32(420.0) * max(Float32(0.0), Float32(1.0) - t * Float32(28.0))
        phase += freq * Float32(6.2831853) / Float32(SAMPLE_RATE)
        var env = max(Float32(0.0), Float32(1.0) - t / Float32(0.085))
        var wave = sin(phase) * Float32(0.70) + _hash_noise(i * 3) * env * Float32(0.45)
        var out = max(Float32(-1.0), min(Float32(1.0), wave * env))
        clip.samples.append(Int16(Int(out * Float32(23000.0))))
    return clip^


def synth_lock_beep_clip() -> WavClip:
    """SDF-1 tactical target lock radar chirp (0.07s)."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.07))
    clip.samples.reserve(total)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var freq = Float32(1320.0) if t < Float32(0.035) else Float32(1760.0)
        var env = max(Float32(0.0), Float32(1.0) - t / Float32(0.07))
        var wave = sin(t * freq * Float32(6.2831853)) * env * Float32(0.65)
        clip.samples.append(Int16(Int(wave * Float32(18000.0))))
    return clip^


def synth_missile_salvo_clip() -> WavClip:
    """Itano Circus micro-missile pod launch hiss & Igniter sweep (0.34s)."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.34))
    clip.samples.reserve(total)
    var lp = Float32(0.0)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var env = min(Float32(1.0), t * Float32(25.0)) * max(Float32(0.0), Float32(1.0) - t / Float32(0.34))
        var raw = _hash_noise(i + 911)
        lp += (raw - lp) * (Float32(0.12) + t * Float32(0.35))
        var whistle = sin(t * (Float32(520.0) + t * Float32(1400.0)) * Float32(6.2831853)) * Float32(0.28)
        var out = max(Float32(-1.0), min(Float32(1.0), (lp * Float32(0.85) + whistle) * env))
        clip.samples.append(Int16(Int(out * Float32(21000.0))))
    return clip^


def synth_explosion_clip() -> WavClip:
    """Zentraedi Battlepod plasma explosion boom (0.48s)."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.48))
    clip.samples.reserve(total)
    var low = Float32(0.0)
    var phase = Float32(0.0)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var env = max(Float32(0.0), Float32(1.0) - t / Float32(0.48))
        var sub_freq = Float32(42.0) + Float32(95.0) * max(Float32(0.0), Float32(1.0) - t * Float32(6.0))
        phase += sub_freq * Float32(6.2831853) / Float32(SAMPLE_RATE)
        low += (_hash_noise(i + 4096) - low) * Float32(0.08)
        var boom = (sin(phase) * Float32(0.65) + low * Float32(0.85)) * env * env
        var out = max(Float32(-1.0), min(Float32(1.0), boom))
        clip.samples.append(Int16(Int(out * Float32(25000.0))))
    return clip^


def synth_turbine_drone_clip() -> WavClip:
    """1.0s harmonic VF-1 thermonuclear turbine chord for raised-cosine continuous drone."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    clip.samples.reserve(SAMPLE_RATE)
    for i in range(SAMPLE_RATE):
        var ph = Float32(i) / Float32(SAMPLE_RATE) * Float32(6.2831853)
        var w = (
            sin(ph * Float32(98.0)) * Float32(0.50)
            + sin(ph * Float32(147.0)) * Float32(0.30)
            + sin(ph * Float32(294.0)) * Float32(0.15)
        )
        clip.samples.append(Int16(Int(w * Float32(14000.0))))
    return clip^


def synth_reflex_cannon_clip() -> WavClip:
    """SDF-1 Main Boom Reflex Cannon: 0.85s harmonic capacitor charge sweep + colossal mega-particle roar."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.85))
    clip.samples.reserve(total)
    var phase_sub = Float32(0.0)
    var phase_hi = Float32(0.0)
    var lp = Float32(0.0)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var out: Float32
        if t < Float32(0.22):
            # Rising Reflex Furnace capacitor whine
            var p = t / Float32(0.22)
            var cfreq = Float32(280.0) + p * p * Float32(1480.0)
            phase_hi += cfreq * Float32(6.2831853) / Float32(SAMPLE_RATE)
            out = (sin(phase_hi) * Float32(0.62) + sin(phase_hi * Float32(1.5)) * Float32(0.32)) * p
        else:
            # Colossal twin-boom heavy particle beam discharge
            var bt = t - Float32(0.22)
            var env = max(Float32(0.0), Float32(1.0) - bt / Float32(0.63))
            var sub_f = Float32(54.0) + Float32(36.0) * max(Float32(0.0), Float32(1.0) - bt * Float32(4.5))
            phase_sub += sub_f * Float32(6.2831853) / Float32(SAMPLE_RATE)
            phase_hi += (Float32(440.0) + sin(bt * Float32(28.0)) * Float32(45.0)) * Float32(6.2831853) / Float32(SAMPLE_RATE)
            lp += (_hash_noise(i + 8191) - lp) * Float32(0.16)
            var beam = sin(phase_sub) * Float32(0.58) + sin(phase_hi) * Float32(0.24) + lp * Float32(0.55)
            out = max(Float32(-1.0), min(Float32(1.0), beam * env))
        clip.samples.append(Int16(Int(out * Float32(24500.0))))
    return clip^


def synth_radio_comms_clip() -> WavClip:
    """UN Spacy VHF tactical radio squelch burst + formant-filtered bridge voice cadence (0.36s)."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = SAMPLE_RATE
    var total = Int(Float32(SAMPLE_RATE) * Float32(0.36))
    clip.samples.reserve(total)
    var lp = Float32(0.0)
    var ph_f0 = Float32(0.0)
    for i in range(total):
        var t = Float32(i) / Float32(SAMPLE_RATE)
        var out: Float32
        if t < Float32(0.032) or t > Float32(0.328):
            # VHF PTT squelch click + static hiss
            var sq = _hash_noise(i * 5 + 71) * Float32(0.55) + sin(t * Float32(2150.0) * Float32(6.2831853)) * Float32(0.35)
            out = max(Float32(-1.0), min(Float32(1.0), sq))
        else:
            # 3-syllable UN Spacy bridge comms formant cadence
            var vt = t - Float32(0.032)
            var syl = Int(vt / Float32(0.098))
            var local_s = vt - Float32(syl) * Float32(0.098)
            var env = min(Float32(1.0), local_s * Float32(55.0)) * max(Float32(0.15), Float32(1.0) - local_s * Float32(8.5))
            var f0 = Float32(182.0) if syl == 0 else (Float32(214.0) if syl == 1 else Float32(168.0))
            var f1 = Float32(720.0) if syl != 1 else Float32(940.0)
            var f2 = Float32(1840.0) if syl == 0 else Float32(1580.0)
            ph_f0 += f0 * Float32(6.2831853) / Float32(SAMPLE_RATE)
            var glot = sin(ph_f0) * Float32(0.45) + sin(ph_f0 * Float32(2.0)) * Float32(0.28)
            var formants = (
                sin(vt * f1 * Float32(6.2831853)) * Float32(0.32)
                + sin(vt * f2 * Float32(6.2831853)) * Float32(0.18)
            )
            lp += ((_hash_noise(i + 303) * Float32(0.16) + (glot + formants) * env) - lp) * Float32(0.34)
            out = max(Float32(-1.0), min(Float32(1.0), lp * Float32(1.15)))
        clip.samples.append(Int16(Int(out * Float32(18500.0))))
    return clip^


def synth_macross_bgm_clip() -> WavClip:
    """6.40s seamless stereo 150 BPM heroic 80s anime brass & driving synth-bass combat anthem loop."""
    var clip = WavClip()
    clip.valid = True
    clip.channels = 2
    clip.sample_rate = SAMPLE_RATE
    # 16 beats at 150 BPM (0.40s/beat, 0.20s eighth-note) = 6.40s exact seamless loop
    var loop_sec = Float32(6.40)
    var total_frames = Int(Float32(SAMPLE_RATE) * loop_sec)
    clip.samples.reserve(total_frames * 2)

    # Heroic D-minor / Bb-major / C-major Macross-style brass fanfare (32 eighth-note slots)
    var lead_hz = [
        Float32(293.66), Float32(293.66), Float32(440.00), Float32(440.00),
        Float32(523.25), Float32(440.00), Float32(392.00), Float32(349.23),
        Float32(392.00), Float32(392.00), Float32(440.00), Float32(523.25),
        Float32(587.33), Float32(587.33), Float32(523.25), Float32(440.00),
        Float32(466.16), Float32(466.16), Float32(587.33), Float32(587.33),
        Float32(659.25), Float32(587.33), Float32(523.25), Float32(440.00),
        Float32(523.25), Float32(523.25), Float32(587.33), Float32(659.25),
        Float32(587.33), Float32(587.33), Float32(440.00), Float32(392.00),
    ]
    var bass_hz = [
        Float32(73.42), Float32(146.83), Float32(73.42), Float32(146.83),
        Float32(73.42), Float32(146.83), Float32(65.41), Float32(130.81),
        Float32(58.27), Float32(116.54), Float32(58.27), Float32(116.54),
        Float32(65.41), Float32(130.81), Float32(65.41), Float32(130.81),
        Float32(58.27), Float32(116.54), Float32(58.27), Float32(116.54),
        Float32(65.41), Float32(130.81), Float32(65.41), Float32(130.81),
        Float32(73.42), Float32(146.83), Float32(73.42), Float32(146.83),
        Float32(65.41), Float32(130.81), Float32(55.00), Float32(110.00),
    ]

    var ph_lead_l = Float32(0.0)
    var ph_lead_r = Float32(0.0)
    var ph_bass = Float32(0.0)
    var ph_arp = Float32(0.0)
    var tau = Float32(6.2831853)
    var inv_sr = Float32(1.0) / Float32(SAMPLE_RATE)

    for f in range(total_frames):
        var t = Float32(f) * inv_sr
        var step8 = min(31, Int(t / Float32(0.20)))
        var step_t = t - Float32(step8) * Float32(0.20)
        var lf = lead_hz[step8]
        var bf = bass_hz[step8]

        # Brass envelope with bright attack + warm sustain
        var brass_env = min(Float32(1.0), step_t * Float32(45.0)) * max(Float32(0.32), Float32(1.0) - step_t * Float32(2.4))
        var vibrato = Float32(1.0) + sin(t * Float32(34.0)) * Float32(0.004)
        ph_lead_l += lf * vibrato * Float32(0.998) * tau * inv_sr
        ph_lead_r += lf * vibrato * Float32(1.002) * tau * inv_sr
        var brass_l = (
            sin(ph_lead_l) * Float32(0.46)
            + sin(ph_lead_l * Float32(2.0)) * Float32(0.26)
            + sin(ph_lead_l * Float32(3.0)) * Float32(0.12)
        ) * brass_env
        var brass_r = (
            sin(ph_lead_r) * Float32(0.46)
            + sin(ph_lead_r * Float32(2.0)) * Float32(0.26)
            + sin(ph_lead_r * Float32(3.0)) * Float32(0.12)
        ) * brass_env

        # Driving octave synth bass + punchy beat kick
        var bass_env = max(Float32(0.18), Float32(1.0) - step_t * Float32(4.2))
        ph_bass += bf * tau * inv_sr
        var bass = (sin(ph_bass) * Float32(0.58) + sin(ph_bass * Float32(2.0)) * Float32(0.28)) * bass_env

        var beat_t = t - Float32(Int(t / Float32(0.40))) * Float32(0.40)
        var kick = sin(beat_t * (Float32(115.0) - beat_t * Float32(220.0)) * tau) * max(Float32(0.0), Float32(1.0) - beat_t * Float32(11.0)) * Float32(0.34)

        # Shimmering 16th-note fifth arpeggio panned across stereo field
        var arp_mult = Float32(1.5) if ((f // (SAMPLE_RATE // 10)) & 1) == 1 else Float32(2.0)
        ph_arp += lf * arp_mult * tau * inv_sr
        var arp = sin(ph_arp) * Float32(0.12)
        var pan = Float32(0.5) + sin(t * Float32(3.14159)) * Float32(0.35)

        var mix_l = max(Float32(-1.0), min(Float32(1.0), (brass_l * Float32(0.52) + bass * Float32(0.44) + kick + arp * pan) * Float32(0.85)))
        var mix_r = max(Float32(-1.0), min(Float32(1.0), (brass_r * Float32(0.52) + bass * Float32(0.44) + kick + arp * (Float32(1.0) - pan)) * Float32(0.85)))
        clip.samples.append(Int16(Int(mix_l * Float32(21500.0))))
        clip.samples.append(Int16(Int(mix_r * Float32(21500.0))))
    return clip^


struct RobotechAudioBank(Movable):
    var audio: NativeAudio
    var base_id: Int
    var enabled: Bool
    var bgm_enabled: Bool

    def __init__(out self, enable_device: Bool = True):
        self.audio = NativeAudio()
        self.enabled = self.audio.initialize() if enable_device else False
        self.audio.clips.clear()
        self.base_id = 0
        self.bgm_enabled = True
        self.audio.clips.append(synth_transform_clip())
        self.audio.clips.append(synth_gunpod_clip())
        self.audio.clips.append(synth_lock_beep_clip())
        self.audio.clips.append(synth_missile_salvo_clip())
        self.audio.clips.append(synth_explosion_clip())
        self.audio.clips.append(synth_turbine_drone_clip())
        self.audio.clips.append(synth_reflex_cannon_clip())
        self.audio.clips.append(synth_radio_comms_clip())
        self.audio.set_music_clip(synth_macross_bgm_clip(), volume=0.22, pitch_follow=False)
        self.audio.drone_clip = SFX_TURBINE_DRONE
        self.audio.set_drone(0.35)

    def trigger(mut self, sfx_id: Int, pitch: Float32 = 1.0, volume: Float32 = 0.75, priority: Int = 2):
        self.audio.play_sound(self.base_id + sfx_id, pitch, volume, priority)

    def trigger_radio(mut self, volume: Float32 = 0.72):
        self.audio.play_speech(self.base_id + SFX_RADIO_COMMS, 1.0, volume, 3)

    def toggle_bgm(mut self) -> Bool:
        self.bgm_enabled = not self.bgm_enabled
        self.audio.set_bus_volumes(1.0, 1.0, Float32(1.0) if self.bgm_enabled else Float32(0.0))
        return self.bgm_enabled

    def update(mut self, throttle: Float32):
        var clamped = max(Float32(0.15), min(Float32(1.0), throttle))
        self.audio.set_drone(Float32(0.25) + clamped * Float32(0.65))
        if self.enabled:
            self.audio.update(clamped)

    def close(mut self):
        self.audio.close()

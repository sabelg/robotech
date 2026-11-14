# Robotech Visual Realism Upgrade Plan (For Desktop GPU Workstation)

This document captures the complete architecture and asset checklist to take **Robotech: SDF-1 Veritech 3D Combat Simulator** from its fast CPU `SimdCanvas` prototype (`~900` flat triangles) to **photorealistic GPU visuals** using **Flint Engine** (`../flint`).

---

## Phase 1: High-Detail 3D `.glb` Assets & Fonts (`assets/`)

Flint's native glTF 2.0 runtime (`flint.assets.glb_native`, `flint.assets.cooked_model`, `flint.anim.skinning`) already decodes binary `.glb` meshes, UVs, normals, PNG/JPEG textures, skeletons, and keyframe animations with zero C libraries.

Place these assets under `robotech/assets/`:
1. **`assets/models/vf1s_valkyrie.glb` (`30k–80k` tris)**:
   - Articulated hierarchy / skeletal rig for `Fighter <-> Guardian <-> Battloid` transformation, variable-sweep wings, intake fans, canopy glass, and GU-11 Gunpod.
   - 2K/4K textures: weathered UN Spacy white/skull trim Albedo, panel-line/rivet Normal map, Metallic/Roughness, and visor/thruster Emissive map.
2. **`assets/models/regult_pod.glb` & `glaug_officer.glb` (`20k–50k` tris)**:
   - Rounded ovoid armor, reverse-jointed leg pistons, glowing cyclops sensor, and plasma barrels.
3. **`assets/models/sdf1_macross.glb`**:
   - Detailed orbital SDF-1 Super Dimension Fortress cruiser with illuminated bridge windows and Reflex Furnace glow.
4. **`assets/sky/earth_orbit_4k.hdr` (or `.png`)**:
   - High-resolution Earth curvature, atmospheric limb glow, solar corona, and orbital starfield.
5. **`assets/fonts/cockpit_hud.ttf`**:
   - Technical HUD TrueType font baked at startup into an anti-aliased SDF atlas via `flint.text.ttf` + `flint.text.sdf_bake`.

---

## Phase 2: Switch `RobotechRenderer` to Flint's Existing GPU Pipeline

All of these modules **already exist** in `../flint/src/flint/render/` and `platform/`:
1. **GPU Rasterizer (`flint.render.gpu_raster` + `flint.platform.vulkan` / `amdgpu`)**:
   - Renders up to **`1,000,000` textured, perspective-correct, bilinear-filtered triangles** at 60–144+ FPS with a `2 GiB` VRAM texture heap (`OffsetAllocator`).
2. **Dynamic Combat Lighting (`flint.render.lighting`)**:
   - Configure `LightingProfile` with harsh orbital solar Key light, cool blue Earthshine Fill, crisp Rim backlight, and bind **48 dynamic point lights** (`MAX_REGISTERED_LIGHTS`) to active Itano Circus warheads, afterburners, muzzle flashes, and Zentraedi plasma explosions.
3. **AgX Filmic Tonemapping & ASC CDL Color Grading (`flint.render.grading`)**:
   - Enable `profile.filmic = 1.0` (Troy Sobotka's AgX curve) so bright plasma fireballs and afterburners roll off smoothly to white-hot cores without harsh RGB clipping.
4. **HDR Multi-Pass Bloom & Soft Particle Billboards (`flint.render.frame_passes`, `fx_atlas`, `particle_atlas`)**:
   - Replace `1px`/`2px` contrail lines and `8×8` point sprites with soft alpha-blended volumetric smoke ribbons, glowing thruster cones, and GPU bloom passes (`PASS_BLOOM`).

---

## Phase 3: Next-Gen Shader Passes on `flint.render.kernel`

Using Flint's write-once compute shader builder (`Kernel` -> emits **both** Vulkan SPIR-V 1.3 and Radeon RDNA GFX12 ISA):
1. **Cook-Torrance GGX Microfacet PBR + Tangent-Space Normal Mapping**:
   - Realistic metallic sheen on titanium nozzles, Fresnel sky reflections on the VF-1 amber canopy glass, and riveted panel micro-relief.
2. **Self-Shadowing (Cascaded / Ray-Marched Shadows)**:
   - Crisp solar shadows cast by the VF-1's wings and GU-11 Gunpod across its own legs and chestplate during mid-air transformation.
3. **Volumetric Afterburner Shock-Diamonds & Atmospheric Cloud Deck**:
   - Glowing Mach shock-diamonds on `SHIFT` afterburner boost and volumetric cloud layers for atmospheric Macross Island defense missions.
4. **Temporal Anti-Aliasing (TAA) & Per-Object Motion Blur**:
   - Sub-pixel edge stability and cinematic motion streaks on high-speed corkscrewing micro-missiles.

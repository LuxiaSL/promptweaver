# Render Performance Audit Outline

## Purpose

This document captures the current render-path performance findings for the
hyperobject viewport, the optimizations that appear worthwhile, and the order
in which they should land.

The goal is to improve frame time on `main` without changing display quality.
That means:

- No intentional reductions in geometric detail.
- No lower sampling density unless an equivalent visual result is demonstrated.
- No palette, shader-ramp, or composition regressions.
- No speculative refactors that widen the blast radius before the hot paths are
  proven.

This is an audit and implementation outline only. No renderer changes are
included here.

## Scope

Primary scope:

- `promptweaver/hyperobject/rasterizer.py`
- `promptweaver/hyperobject/transform.py`
- `promptweaver/hyperobject/scene.py`
- `promptweaver/hyperobject/geometry.py`
- `promptweaver/hyperobject/postfx.py`

Reference patterns consulted:

- `donut.c`: flat hot loop, inline math, pre-sized buffers, minimal allocation,
  direct z-buffer writes.
- `asciimare`: lookup-heavy rendering, precomputed shading decisions, bias
  toward avoiding work inside the frame loop.

The useful transfer is structural, not literal. Promptweaver is not a voxel
raycaster and should not be forced into one.

## Measurement Context

Local environment used during audit:

- Python `3.12.9`
- Terminal-sized render target for benchmarks: `100x30`
- Timing includes `scene.render(rast)` and `rast.grid.to_rich_text()`
- Timing excludes Textual's final terminal paint

Representative whole-frame timings observed:

| Case | Approx. avg ms/frame |
| --- | ---: |
| Empty scene | 0.58 |
| Filled mesh (`filled_icosa`) | 1.9 |
| Wireframe mesh (`wireframe_org`) | 1.9 |
| Dual mesh morph | 2.4 |
| Heightmap noise | 6.2 |
| Fragmenting solid | 6.1 |
| Direct surface (`surface_mobius`) | 11.4 |
| Heightmap terrain | 14.3 |
| Point cloud (`lorenz`) | 17.8 |
| Direct surface (`surface_torus`) | 25.7 |

Supporting micro-benchmarks:

- `CharGrid.clear()`: about `0.18ms`
- `CharGrid.to_rich_text()`: about `0.38ms`
- `postfx.apply_crt_warp()`: about `4.6ms`
- `postfx.apply_vignette()`: about `1.4ms`
- `postfx.apply_bloom()`: about `0.7ms`
- `postfx.apply_edge_glow()`: about `0.7ms`
- `postfx.apply_noise_grain()`: about `0.8ms`

## High-Level Conclusion

The current renderer is not "slow because Python" in a generic sense. It is
slow in a few specific places where the frame loop still pays Python object and
function-call overhead at a scale large enough to matter.

The audit suggests:

- A substantial share of current pain is recoverable in Python by flattening
  hot loops, caching immutable data, and avoiding per-sample/per-point object
  churn.
- A full rewrite of Promptweaver is not justified by the measured profile.
- If a native rewrite is ever warranted, it should target the renderer core
  after the Python structural wins have been exhausted.

Working estimate:

- Python-only refactors can likely recover roughly `40-60%` of the worst-frame
  costs in the current hot modes.
- A native core after that could still help, but likely as an incremental
  multiplier on the remaining hot loop rather than an order-of-magnitude win
  for the whole app.

## Findings By Area

### 1. Direct Surface Sampling Is The Largest Single Hot Path

Relevant code:

- `TorusSampler.samples()`
- `MobiusSampler.samples()`
- `AsciiRasterizer.draw_surface_direct()`

Observed behavior:

- Torus direct sampling is the slowest representative mode.
- The torus path emits about `14,700` samples per frame.
- The Mobius path emits about `1,890` samples per frame.
- The rasterizer hot loop is already partially inlined.
- The remaining avoidable cost is largely in generating the same sample geometry
  every frame.

Important detail:

- Torus and Mobius sample positions and normals are static in model space.
- Camera and model transforms change per frame, but the sample topology does
  not.
- That makes the sample stream cacheable with no change in display output.

Local prototype result:

- Torus: about `21.2ms` to `16.5ms` with cached samples.
- Mobius: about `13.7ms` to `1.7ms` with cached samples.

Interpretation:

- This is exactly the class of optimization implied by `donut.c`: keep the
  inner loop focused on transform, shade, test, write. Do not regenerate static
  source data inside it.

### 2. Point Cloud Rendering Pays Heavy Python Overhead

Relevant code:

- `AsciiRasterizer.draw_points()`
- `ProjectionContext.project_vertex()`
- `CharGrid.write()`

Observed behavior:

- Lorenz point clouds are expensive even though the underlying math is simple.
- Profiling shows a large number of calls into `project_vertex()` and `write()`
  from the per-point loop.
- The hot path is dominated less by raw arithmetic and more by function-call
  overhead, object access, and repeated bounds/z-test logic.

Local prototype result:

- An inline point-cloud path with equivalent output reduced the draw loop from
  about `12.9ms` to `3.4ms`.
- At whole-frame level, Lorenz dropped from about `15.5ms` to `7.9ms`.

Interpretation:

- This is a high-confidence Python win.
- A flatter loop can preserve the exact same rendering decisions while avoiding
  much of the interpreter overhead.

### 3. Heightmap Rebuilds Do Unnecessary Topology Work

Relevant code:

- `HeightMap.to_mesh()`
- `Scene._render_geometry_state()`

Observed behavior:

- Heightmaps are rendered through filled triangles.
- `HeightMap.to_mesh()` still computes edges even when the mesh is never used by
  a wireframe path.
- Terrain rebuild frames therefore do extra topology work that produces no
  visual effect.

Local prototype result:

- Terrain mesh-build cost dropped from about `11.9ms` to `8.9ms` when edge
  generation was skipped.

Interpretation:

- This is a clean "remove useless work" optimization.
- It does not alter the displayed image because the filled renderer never reads
  the edge list.

### 4. Fragment Rendering Rebuilds Too Much Every Frame

Relevant code:

- `Scene._mesh_for_render()`
- `Scene._build_fragment_mesh()`
- `Mesh.compute_normals()`

Observed behavior:

- Fragment rendering duplicates face-local geometry every frame and recomputes
  normals after translation.
- In the current effect, fragment motion is translational drift.
- Pure translation does not change face normals.

Interpretation:

- The current pipeline is paying reconstruction cost that is only partially
  necessary.
- The fragment mesh topology and per-face normals are strong candidates for
  caching.
- At minimum, normal recomputation should be scrutinized carefully because it is
  not automatically justified by the current motion model.

### 5. Filled Triangle Rendering Is Not The First Bottleneck

Relevant code:

- `AsciiRasterizer.draw_mesh_filled()`
- `AsciiRasterizer._fill_triangle_gouraud()`

Observed behavior:

- Standard filled meshes are comparatively cheap.
- The rasterizer already does the right broad things:
  one projection pass, incremental barycentric stepping, direct z-buffer access,
  and in-place cell writes.

Interpretation:

- This path still has micro-optimization headroom.
- It is not the right place to start if the goal is to move real frame time.

### 6. PostFX Is Secondary But Not Free

Relevant code:

- `postfx.apply_crt_warp()`
- `postfx.apply_vignette()`
- `postfx.apply_bloom()`
- `postfx.apply_edge_glow()`
- `postfx.apply_noise_grain()`

Observed behavior:

- `crt_warp` is expensive for a post-process pass because it allocates a new
  `Cell` array and repeatedly goes through `grid.get()` / `in_bounds()`.
- Other effects are smaller but still measurable.

Interpretation:

- PostFX should be optimized after the renderer hot paths.
- It is not the main problem today, but it can become visible once the core
  render loop is improved.

## Proposed Optimization Program

### Phase 1: High-Confidence Python Wins

#### A. Cache Direct-Surface Samples

Change:

- Convert Torus and Mobius samplers to precompute and retain model-space sample
  arrays or tuples.
- `samples()` should iterate cached data instead of regenerating positions and
  normals on each frame.

Why:

- Sample geometry is static.
- This removes repeated trig, object allocation, and iterator work.

Expected impact:

- Large for `minimal_object` and `essence`.
- Highest ROI among all currently identified options.

Quality risk:

- None if cached data is bit-for-bit the same sample stream.

#### B. Add A Fast Point-Cloud Path

Change:

- Rewrite `draw_points()` around an inlined projection/z-test/write loop in the
  same style already used by `draw_surface_direct()`.

Why:

- This removes a large amount of Python call overhead.
- The existing path is structurally simple enough to inline safely.

Expected impact:

- Large for `abstract_field` and any other large point-cloud mode.

Quality risk:

- Low, provided the exact projection, depth mapping, character selection, and
  style banding rules are preserved.

#### C. Remove Unused Edge Generation From Heightmap Mesh Builds

Change:

- Teach `HeightMap.to_mesh()` to skip edge computation by default or expose a
  flag that allows callers to opt in only when needed.

Why:

- Edges are not used by the filled heightmap path.

Expected impact:

- Moderate on rebuild frames for `textural_macro` and `environmental`.

Quality risk:

- None for filled rendering.

#### D. Cache Fragment Geometry And Reuse Normals Where Valid

Change:

- Separate fragment topology construction from per-frame offset application.
- Avoid recomputing normals when motion is translational only.

Why:

- Current per-frame work is heavier than the visual effect requires.

Expected impact:

- Moderate for `ruin_state`.

Quality risk:

- Low if normals are only reused when transform semantics justify it.

### Phase 2: Secondary Renderer Cleanup

#### E. Tighten Wireframe And Tesseract Paths

Change:

- Avoid building full point lists when Bresenham output can be streamed.
- Reduce repeated helper-call overhead in line drawing.

Why:

- These modes are not currently the worst offenders, but the path still has
  avoidable Python churn.

Expected impact:

- Small to moderate.

Quality risk:

- Low if rasterized endpoints and depth interpolation remain identical.

#### F. Improve `CharGrid` Access Patterns

Change:

- Favor direct flat-list access over helper methods inside hot loops.
- Review whether `clear()` can be made more cache-friendly without changing the
  cell model.

Why:

- Helper methods are convenient but measurable in inner loops.

Expected impact:

- Small per pass, but broadly useful.

Quality risk:

- Low.

### Phase 3: PostFX Cleanup

#### G. Make Expensive PostFX Allocation-Light

Change:

- Rewrite `crt_warp`, `vignette`, `bloom`, and `edge_glow` to operate more
  directly on flat arrays and avoid repeated `grid.get()` lookups.

Why:

- These are currently dominated by Python object access and temporary
  allocation.

Expected impact:

- Small to moderate depending on the active effect stack.

Quality risk:

- Low if the sampling behavior is preserved.

## What Should Not Be Done First

### Do Not Rewrite The Entire Project In Another Language First

Reason:

- Most measured pain is concentrated in a handful of render hot paths, not in
  the whole application.
- Python-only structural changes already show significant gains.
- A whole-project rewrite would add major complexity before the obvious wins are
  collected.

### Do Not Reduce Sample Counts Or Mesh Detail As A Shortcut

Reason:

- The stated constraint is to preserve display quality.
- The current optimization opportunities are strong enough that quality tradeoff
  is not the first lever to pull.

### Do Not Merge Broad Architectural Refactors With Performance Work

Reason:

- Perf work needs tight diff scope and repeatable measurement.
- Large unrelated changes make regressions harder to attribute.

## Native Rewrite Assessment

If Python refactors are exhausted and more headroom is still required, the
likely next step is not a full rewrite. It is a narrow native renderer core.

Best native candidates:

- Triangle fill hot loop
- Point-cloud projection/write path
- Direct-surface transform/shade/z-test loop
- Possibly selected postfx kernels

Poor native candidates:

- Broad scene orchestration
- Prompt interpretation
- Palette/style selection logic at module scale
- UI glue unless Textual itself becomes the limiting factor

Current position:

- Native code is a second-stage optimization strategy, not the first move.
- The audit suggests there is still too much easy Python headroom to justify a
  full-language migration today.

## Validation Requirements For Each Optimization

Every perf change should be validated against both speed and image stability.

Minimum validation:

- Benchmark before and after on the affected template(s)
- Compare rendered output for representative frames
- Verify no character-ramp changes unless intentional
- Verify no palette/style-band regressions
- Verify no depth ordering regressions

Suggested benchmark set:

- `minimal_object`
- `essence`
- `abstract_field`
- `environmental`
- `textural_macro`
- `ruin_state`
- A baseline filled mesh case for regression detection

## Suggested Implementation Order

1. Cache direct-surface samples.
2. Introduce the fast point-cloud path.
3. Stop computing unused heightmap edges.
4. Rework fragment mesh generation and normal reuse.
5. Tighten wireframe/tesseract helpers.
6. Optimize expensive postfx passes.
7. Re-measure.
8. Only then decide whether a native-core experiment is still justified.

## Expected Outcome If Phase 1 Lands Cleanly

If the highest-confidence Python optimizations land without regressions, the
expected outcome is:

- Noticeably better frame pacing in the slow templates.
- Material reduction in worst-case frame times.
- Better basis for deciding whether native code is actually needed.

Most importantly, it should answer the strategic question with evidence:

- If Phase 1 produces the expected gains, Promptweaver should stay Python-first
  and only consider a narrow native renderer later.
- If Phase 1 underperforms, then a native-core spike becomes easier to justify
  because the remaining cost will be better isolated.

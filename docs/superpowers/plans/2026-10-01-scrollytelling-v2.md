# ML Visual Lab V2 Scrollytelling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a premium scroll-driven neural-network lesson where one spam email visibly flows forward through a persistent network, produces loss, sends gradients backward, and updates parameters.

**Architecture:** Convert the current static step UI into an ES-module-based scrollytelling app. Keep neural-network math pure and testable in `model.js`, create the SVG once through `network-renderer.js`, drive deterministic scene transitions through `scroll-scenes.js`, and keep narrative/math copy in `scene-content.js`. The page remains a static GitHub Pages site with GSAP/ScrollTrigger and KaTeX loaded from CDN.

**Tech Stack:** HTML, CSS, vanilla JavaScript ES modules, SVG, GSAP + ScrollTrigger, KaTeX, Node built-in test runner for pure/static contract tests.

**Spec:** `docs/superpowers/specs/2026-10-01-scrollytelling-neural-network-design.md`

## Global Constraints

- Initial public story uses one canonical spam email only.
- No React/Vue/Svelte rewrite.
- Page must remain deployable as a static GitHub Pages site.
- Network SVG must be created once; scroll scenes update attributes/classes/values instead of rebuilding the full SVG.
- Forward signal uses cool blue, loss amber, backprop violet, update green, inactive structure muted steel gray.
- Desktop uses a sticky visualization with narrative sections; mobile uses a sticky compact visualization with no horizontal scrolling.
- Respect `prefers-reduced-motion`; all narrative content must remain understandable without particle travel or long transforms.
- Normal text must meet WCAG AA contrast.
- Color must not be the sole indicator of computational state.
- Avoid excessive blur/filter effects and keep particle count bounded.
- No arbitrary user-created architectures, multiple sample selector, 3D/WebGL, audio, or full free-play Lab Mode in initial V2.

## Review Focus

- **Reduced motion:** with `prefers-reduced-motion: reduce`, scene changes must resolve immediately without relying on moving particles.
- **Reverse scrolling:** moving from a later scene back to an earlier scene must restore the correct deterministic network state rather than accumulating animation artifacts.
- **Compact layout:** at mobile geometry, every neuron/loss element must remain within the SVG viewBox and no page-level horizontal scrolling may be introduced.
- **Missing GSAP/KaTeX CDN:** core lesson content and static network state must still render; enhancement failure must not leave a blank page.
- **Direct scene navigation:** keyboard activation of the scene nav must scroll to the correct narrative section and update the current-scene ARIA state.

---

### Task 1: Pure neural-network model and fixed teaching sample

**Files:**
- Create: `model.js`
- Create: `tests/model.test.mjs`
- Modify: `app.js` after tests pass to consume the model module

**Interfaces:**
- Consumes: no V2 module dependencies.
- Produces:
  - `TEACHING_SAMPLE: { email: string, x: number[], y: number }`
  - `FEATURES: { key: string, label: string }[]`
  - `INITIAL_PARAMS`
  - `forward(params, sample) -> ForwardSnapshot`
  - `backward(params, snapshot) -> GradientSnapshot`
  - `updatedParams(params, grads, learningRate) -> Params`
  - `trainingPreview(params, samples, learningRate, epochs) -> { params, history }`

- [ ] **Step 1: Write failing model tests**

Use Node's built-in `node:test` and `assert/strict`. Tests assert:

- teaching sample is exactly one public canonical spam email with four normalized inputs and target `1`
- `forward(INITIAL_PARAMS, TEACHING_SAMPLE)` returns finite `z1[3]`, `h[3]`, `z2`, `yHat` in `(0,1)`, and positive finite BCE loss
- `backward` returns `dW1` shape `3x4`, `dW2` length `3`, `db1` length `3`, finite `db2`
- `updatedParams` does not mutate its input and each updated parameter equals `old - η*g`
- one update on the teaching sample with `η=0.10` reduces that sample's BCE loss

- [ ] **Step 2: Run tests and verify failure**

Run: `node --test tests/model.test.mjs`  
Expected: FAIL because `../model.js` does not exist.

- [ ] **Step 3: Implement `model.js`**

Keep all math pure. Preserve the current 4 → 3 → 1 network, ReLU hidden layer, sigmoid output and BCE objective. Public UI exposes only the canonical spam email; an internal small training array may remain private to support the final training-loop scene.

- [ ] **Step 4: Run tests and verify pass**

Run: `node --test tests/model.test.mjs`  
Expected: all model tests PASS.

- [ ] **Step 5: Commit**

```bash
git add model.js tests/model.test.mjs app.js
git commit -m "refactor: extract neural network teaching model"
```

### Task 2: Scrollytelling page shell and premium visual system

**Files:**
- Modify: `index.html`
- Modify: `styles.css`
- Create: `tests/page-contract.test.mjs`

**Interfaces:**
- Consumes: scene IDs defined in this task and later reused by `scroll-scenes.js`.
- Produces:
  - narrative sections with IDs `scene-intro`, `scene-inputs`, `scene-weights`, `scene-relu`, `scene-output`, `scene-loss`, `scene-backprop`, `scene-update`, `scene-training`
  - `#network-stage`, `#networkSvg`, `#sceneNav`, `#mathPanel`, `#emailCard`, `#lossChart`
  - semantic layout classes for sticky visualization and scroll narrative

- [ ] **Step 1: Write failing static-contract tests**

Read `index.html` and `styles.css` with `node:fs`. Assert:

- all nine scene IDs exist exactly once and in the intended order
- `#sampleSelect` is absent from public UI
- `#networkSvg`, `#sceneNav`, `#mathPanel`, `#emailCard`, and `#lossChart` exist
- page loads `app.js` with `type="module"`
- GSAP ScrollTrigger and KaTeX resources are declared
- CSS includes `position: sticky` for the visualization stage
- CSS contains `@media (prefers-reduced-motion: reduce)`
- CSS compact breakpoint does not set a positive `min-width` on `#networkSvg`
- body/app shell prevent accidental page-level horizontal overflow

- [ ] **Step 2: Run tests and verify failure**

Run: `node --test tests/page-contract.test.mjs`  
Expected: FAIL because the current document is the old dashboard/controls layout.

- [ ] **Step 3: Implement the new page shell**

Replace the dashboard-first structure with:

- editorial intro
- sticky network stage
- scroll narrative sections
- compact scene navigation
- single spam-email card
- persistent math/metric rail in the visualization
- minimal end-of-story training summary

Remove public sample selector, dataset table and dominant training-control dock from the scroll-first experience.

- [ ] **Step 4: Implement the premium CSS system**

Use graphite surfaces, semantic computational colors, strong display/body/mono hierarchy, restrained radii, controlled spacing, and minimal shadows. Desktop should feel like one continuous visualization surface rather than nested cards.

- [ ] **Step 5: Run static-contract tests**

Run: `node --test tests/page-contract.test.mjs`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add index.html styles.css tests/page-contract.test.mjs
git commit -m "feat: add scroll-driven lesson shell"
```

### Task 3: Persistent SVG neural-network renderer

**Files:**
- Create: `network-renderer.js`
- Create: `tests/network-state.test.mjs`
- Modify: `app.js`

**Interfaces:**
- Consumes:
  - `ForwardSnapshot`, `GradientSnapshot`, `Params` from `model.js`
  - scene names from Task 2
- Produces:
  - `createNetworkRenderer(svgElement, options) -> NetworkRenderer`
  - `NetworkRenderer.setLayout("desktop" | "compact")`
  - `NetworkRenderer.setSnapshot(snapshot, grads, params)`
  - `NetworkRenderer.applyScene(sceneId, { immediate?: boolean })`
  - `NetworkRenderer.pulseForward(group)`
  - `NetworkRenderer.pulseBackward(group)`
  - `NetworkRenderer.flashUpdate()`
  - exported pure `getNetworkGeometry(layout)`
  - exported pure `getSceneVisualState(sceneId)`

- [ ] **Step 1: Write failing renderer-state tests**

Assert:

- desktop geometry has 4 input, 3 hidden, 1 output and 1 loss positions inside `1080x620`
- compact geometry has all positions inside its compact viewBox
- `scene-inputs` highlights inputs and keeps later layers subdued
- `scene-weights` marks input→hidden connections as forward-active
- `scene-relu` exposes hidden activation state
- `scene-loss` makes the loss state focal
- `scene-backprop` marks direction as backward/gradient
- `scene-update` marks parameter-update state
- unknown scene IDs fall back to a safe dormant state

- [ ] **Step 2: Run tests and verify failure**

Run: `node --test tests/network-state.test.mjs`  
Expected: FAIL because renderer module does not exist.

- [ ] **Step 3: Implement persistent SVG creation**

Create SVG groups once for:

- background guide/grid
- connections
- animated pulse layer
- four input neurons
- three hidden neurons
- output neuron
- loss module
- temporary annotation layer

Use stable `data-*` attributes so scenes can address components without replacing `innerHTML`.

- [ ] **Step 4: Implement neuron and connection visual states**

Neurons get persistent shell/core/value/label parts and state classes: dormant, receiving, active, zeroed, gradient, updated. Connections get inactive, forward, backward and updated states plus weight-magnitude width.

- [ ] **Step 5: Implement bounded pulse primitives**

Use a small reusable pool of SVG pulse elements; animate along existing paths. Reduced-motion or missing GSAP resolves to immediate state changes without particle travel.

- [ ] **Step 6: Run renderer-state and model tests**

Run: `node --test tests/model.test.mjs tests/network-state.test.mjs`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add network-renderer.js app.js tests/network-state.test.mjs
git commit -m "feat: build persistent neural network renderer"
```

### Task 4: Scene content, KaTeX math, and deterministic scroll choreography

**Files:**
- Create: `scene-content.js`
- Create: `scroll-scenes.js`
- Create: `tests/scenes.test.mjs`
- Modify: `app.js`
- Modify: `styles.css`

**Interfaces:**
- Consumes:
  - renderer interface from Task 3
  - model snapshots from Task 1
  - DOM scene IDs from Task 2
- Produces:
  - `SCENES` ordered descriptor array
  - `getSceneContent(sceneId, context)`
  - `renderMath(panel, expressions, context)`
  - `createScrollStory({ renderer, scenes, reducedMotion, onSceneChange })`
  - `scrollToScene(sceneId)`

- [ ] **Step 1: Write failing scene tests**

Assert:

- `SCENES` contains exactly the nine scene IDs in the same order as the HTML
- every scene has title, short intuition and renderer state
- ReLU content contains numeric pre-activation and post-ReLU substitution
- sigmoid scene contains logit and probability values from the current snapshot
- loss scene contains BCE notation and numeric loss
- backprop scene contains `∂L/∂z_out = ŷ - y` and at least one live gradient value
- update scene contains old value, gradient, `η=0.10`, delta and new value
- reduced-motion story adapter exposes the same scene transitions without scrub/particle requirements

- [ ] **Step 2: Run tests and verify failure**

Run: `node --test tests/scenes.test.mjs`  
Expected: FAIL because scene modules do not exist.

- [ ] **Step 3: Implement scene content and KaTeX wrapper**

Render semantic fallback text first. If `window.katex` is available, enhance math blocks with KaTeX; if it is unavailable, leave readable text equations in place.

- [ ] **Step 4: Implement ScrollTrigger choreography**

Create one trigger per narrative section. Enter/enterBack selects the scene deterministically. Use short scene timelines for staged computational events rather than binding every property continuously to scroll.

- [ ] **Step 5: Implement direct navigation**

Scene-nav buttons scroll to the matching section, preserve keyboard behavior, and set `aria-current="step"` on the active scene.

- [ ] **Step 6: Implement reverse-scroll safety**

Before applying any scene, renderer resets to that scene's canonical visual state, then runs only the transition for that scene. No scene may depend on mutation left behind by a later scene.

- [ ] **Step 7: Run all tests**

Run: `node --test tests/*.test.mjs`  
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scene-content.js scroll-scenes.js app.js styles.css tests/scenes.test.mjs
git commit -m "feat: choreograph neural network scroll story"
```

### Task 5: Training-loop finish, mobile behavior, resilience, and final verification

**Files:**
- Modify: `app.js`
- Modify: `network-renderer.js`
- Modify: `scroll-scenes.js`
- Modify: `styles.css`
- Modify: `README.md`
- Create: `tests/integration-contract.test.mjs`

**Interfaces:**
- Consumes all V2 modules.
- Produces the finished initial V2 experience and documented architecture.

- [ ] **Step 1: Write failing integration-contract tests**

Assert:

- `app.js` imports model, renderer, content and scroll-story modules
- application initializes a static first scene even when GSAP is unavailable
- compact breakpoint selects compact geometry
- scene navigation exposes `aria-current`
- loss chart element has an accessible label/title
- README identifies V2 as scroll-driven and documents the one-sample public story
- no legacy sample selector / dataset-table initialization remains

- [ ] **Step 2: Run tests and verify failure**

Run: `node --test tests/*.test.mjs`  
Expected: integration contract FAIL until V2 initialization is complete.

- [ ] **Step 3: Implement the final training-loop scene**

Generate a small fixed loss history from the model's internal training preview and draw it as a compact SVG sparkline/chart. Show before/after probability and loss without introducing a full dashboard.

- [ ] **Step 4: Implement robust startup**

Initialize model snapshot and persistent renderer first. Enhance with KaTeX and ScrollTrigger only when those globals exist. Missing CDN enhancement must leave a fully readable static first scene.

- [ ] **Step 5: Polish mobile and reduced-motion behavior**

At compact breakpoints:

- use compact network geometry
- hide dense per-edge numeric annotations
- keep neuron values and narrative math readable
- maintain sticky visualization without covering narrative text
- ensure no horizontal page scroll

For reduced motion, remove particle travel, long transforms and scroll scrubbing while preserving scene state changes.

- [ ] **Step 6: Update README**

Document:

- one-sample public story
- nine scroll scenes
- persistent SVG renderer
- GSAP/ScrollTrigger and KaTeX enhancement model
- local verification commands

- [ ] **Step 7: Run complete automated verification**

Run:

```bash
node --check app.js
node --check model.js
node --check network-renderer.js
node --check scene-content.js
node --check scroll-scenes.js
node --test tests/*.test.mjs
```

Expected: all syntax checks exit 0 and all tests PASS.

- [ ] **Step 8: Perform browser-level review**

Review at approximately 1440px, 1024px, 768px and 390px widths. Check:

- sticky stage behavior
- no horizontal scrolling
- scene transitions forward and backward
- neuron/connection values remain readable
- KaTeX enhancement
- direct nav
- reduced-motion mode
- missing-CDN fallback

- [ ] **Step 9: Commit**

```bash
git add app.js network-renderer.js scroll-scenes.js styles.css README.md tests/integration-contract.test.mjs
git commit -m "feat: finish ML Visual Lab scrollytelling V2"
```

## Final Verification

After all tasks:

```bash
node --check app.js
node --check model.js
node --check network-renderer.js
node --check scene-content.js
node --check scroll-scenes.js
node --test tests/*.test.mjs
```

Then compare `visual/scrollytelling-v2` against `design/gpt-taste-redesign`, confirm only intended V2 files changed, and run whole-branch review before opening the V2 PR.

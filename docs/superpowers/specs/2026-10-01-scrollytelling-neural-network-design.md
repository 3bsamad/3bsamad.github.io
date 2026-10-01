# ML Visual Lab V2 — Scroll-Driven Neural Network Story

**Date:** 2026-10-01  
**Branch:** `visual/scrollytelling-v2`  
**Baseline:** `design/gpt-taste-redesign`

## 1. Goal

Transform the current neural-network teaching page from a polished static/step-based interface into a premium scroll-driven educational experience.

The network becomes the main visual story. As the user scrolls, data moves forward through the network, loss is computed, gradients move backward, and parameters visibly update.

The experience should feel closer to a high-end interactive technical explainer than a dashboard.

## 2. Initial implementation scope

The first V2 deliberately uses **one canonical spam email** only.

Example:

> URGENT: Verify your bank account immediately using this link.

The sample is represented by the existing four normalized features:

- suspicious words
- links
- sender trust
- urgency

No sample selector and no multi-example dataset UI are required for the initial scrollytelling version.

This keeps the narrative coherent and allows motion, typography, math rendering, neuron states, and connection behavior to receive most of the design effort.

## 3. Core interaction model

### Desktop

Use a two-column scrollytelling composition:

- a large sticky visualization occupies the main visual area
- explanatory sections scroll alongside it
- each section activates a deterministic visualization scene
- the active scene transitions smoothly rather than rebuilding the visualization from scratch

The lesson sequence is:

1. Intro
2. Input features
3. Weighted connections
4. Hidden activations / ReLU
5. Output logit and sigmoid
6. Binary cross-entropy loss
7. Backpropagation
8. Gradient-descent update
9. Short training-loop summary

A compact progress/navigation rail remains available so users can jump directly to a scene.

### Mobile

Use a stacked scrollytelling layout:

- visualization remains sticky near the top of the viewport
- explanatory content scrolls below it
- numerical edge labels are reduced or hidden
- tapping a neuron or important connection can expose detail when appropriate
- the core story must remain understandable without horizontal scrolling

## 4. Visual direction

The design should feel like a premium scientific instrument or interactive ML debugger, not a generic AI SaaS page.

### Palette

Use a restrained graphite/near-black base with semantic computational colors:

- forward signal: cool blue
- neuron activation: bright neutral / ice blue
- loss: amber
- backpropagation: violet
- parameter update: green
- inactive network structure: muted steel gray

Avoid decorative purple-blue gradients unless they directly encode computational state.

### Typography

Use a strong sans + technical mono pairing.

Preferred direction:

- interface/display: Geist Sans or an equivalent high-quality neutral variable sans
- equations, parameter values and labels: Geist Mono / JetBrains Mono / equivalent

If self-hosting font files is not practical in the initial implementation, use a deliberate high-quality system stack and keep the type hierarchy independent of the exact font family.

Type hierarchy should emphasize:

- large editorial headline
- strong section titles
- readable explanatory body text
- monospaced equations and numerical values
- minimal use of tiny uppercase labels

## 5. Network visualization

The network should no longer read as only circles and lines.

### Neurons

Each neuron is a persistent SVG component with:

- outer shell
- inner/core region
- label
- current value
- semantic state classes
- subtle magnitude indication where useful

Supported states:

- dormant
- receiving
- active
- clamped to zero by ReLU
- receiving gradient
- updated

A neuron should visibly react when information reaches it.

### Connections

Connections are persistent SVG paths with:

- base inactive state
- semantic forward/backprop/update states
- optional width modulation based on weight magnitude
- active glow or emphasis only when relevant
- temporary inline annotations for the current concept

Connections should support moving pulses/particles:

- blue for forward activations
- violet for gradients
- green transition/flash for parameter update

The motion must show direction and causality rather than being decorative.

## 6. Scene choreography

### Scene 0 — Intro

- headline and short explanation establish the lesson
- network is visible but subdued
- canonical spam email appears
- prompt invites the user to scroll to follow the computation

### Scene 1 — Inputs become numbers

- important words/attributes in the email are visually associated with the four features
- feature values move or transition toward the input neurons
- input neurons activate one by one
- other network layers remain subdued

### Scene 2 — Weighted connections

- input-to-hidden connections draw or brighten
- data pulses move toward hidden neurons
- one representative hidden neuron gets extra emphasis
- its weighted contributions may appear temporarily along relevant edges
- unrelated annotations remain hidden

### Scene 3 — ReLU

- hidden pre-activation values become visible
- positive values pass through
- at least one negative value, if available in the fixed example/weights, visibly clamps to zero
- hidden neuron states clearly differentiate active versus zeroed neurons

### Scene 4 — Logit and sigmoid

- hidden activations travel toward the output neuron
- contribution values combine into the output logit
- the logit transitions into a probability
- a compact sigmoid curve may appear beside the output to illustrate the transformation

### Scene 5 — Loss

- target and prediction become visually paired
- binary cross-entropy equation appears with proper math rendering
- loss becomes the focal value
- network dims slightly to emphasize the objective

### Scene 6 — Backpropagation

- motion direction reverses
- violet pulses move from loss toward output, hidden layer and earlier connections
- annotations switch from weight/contribution emphasis to gradient emphasis
- parameter inspector shows `g = ∂L/∂w`
- gradients should feel like sensitivity/blame flowing backward

### Scene 7 — Gradient descent update

- selected parameters show:
  - old value
  - gradient
  - learning rate
  - update delta
  - new value
- relevant connections transition to their updated state
- green is used sparingly to indicate completed parameter changes

### Scene 8 — Training loop

- the full cycle compresses into a short repeatable loop
- a small loss history chart appears
- before/after prediction and loss are shown
- this scene explains that real training repeats this process over many examples

For V2 initial implementation, the loop can still use the existing tiny internal dataset for training calculations if convenient, but the user-facing story remains centered on one spam email.

## 7. Scroll behavior

Use CSS sticky positioning for layout and GSAP + ScrollTrigger for scroll choreography.

Each narrative section acts as a scene trigger.

Required behavior:

- enter scene: animate to the scene's target state
- reverse scroll: transition back to the previous state cleanly
- direct navigation: clicking a step scrolls to its section
- animations should not replay chaotically when moving across boundaries
- no sudden layout shifts

Scroll animation should be smooth but not over-scrubbed. Important computation events should have clear staged timing rather than every property being continuously tied to scroll position.

## 8. Rendering architecture

The current `renderNetwork()` pattern rebuilds the SVG as a string. V2 should move to a persistent renderer.

### Recommended modules

`app.js`
- fixed example data
- forward pass
- backpropagation
- parameter update math
- application initialization

`network-renderer.js`
- creates SVG nodes, paths, labels and particle layers once
- exposes functions for changing network visual state
- updates values without replacing the whole SVG

`scroll-scenes.js`
- defines scene triggers
- owns GSAP/ScrollTrigger timelines
- maps scroll sections to renderer states

`math-panel.js`
- renders equations and numerical substitutions
- wraps KaTeX integration

A smaller file split is acceptable if the implementation remains clear, but the renderer and scroll choreography should not become one monolithic function.

## 9. Math rendering

Use KaTeX for mathematical expressions where practical.

Important equations include:

- hidden weighted sum
- ReLU
- output logit
- sigmoid
- binary cross entropy
- `∂L/∂z_out = ŷ - y`
- gradient-descent update `w_new = w_old - η∂L/∂w`

The user should see both mathematical notation and the current numeric substitution.

## 10. Secondary interaction

The default experience is scroll-first.

Keep direct controls secondary:

- compact scene navigation
- replay current scene
- reduced-motion / manual navigation compatibility

The previous large block of lesson/training controls should not dominate the visual hierarchy.

A future Lab Mode may restore deeper manual experimentation, but it is outside the initial V2 scope.

## 11. Accessibility

- all narrative content remains readable without animation
- direct scene navigation is keyboard accessible
- current scene is exposed with suitable ARIA state where practical
- respect `prefers-reduced-motion`
- reduced-motion users receive state changes without particle travel or long transforms
- normal text must meet WCAG AA contrast
- color is not the sole indicator of forward/backprop/update state

## 12. Performance

- create the SVG structure once
- animate transforms, opacity, stroke and small text updates
- avoid rebuilding the full SVG on scroll
- avoid excessive blur/filter effects
- use a bounded number of particles
- no framework rewrite is required

Target technologies:

- HTML
- CSS
- vanilla JavaScript
- SVG
- GSAP + ScrollTrigger
- KaTeX

The page remains deployable on GitHub Pages.

## 13. Out of scope for initial V2

- arbitrary user-created neural-network architectures
- multiple user-selectable samples
- large datasets
- React/Vue/Svelte rewrite
- 3D/WebGL network rendering
- audio
- full free-play lab mode
- multiple ML lessons on the same page
- editable weights during the scroll story

These can be added later after the primary scrollytelling experience is excellent.

## 14. Success criteria

The V2 is successful when:

1. A first-time visitor understands forward pass, loss, backpropagation and parameter update by scrolling through one coherent example.
2. The network feels like a live computational system rather than a static diagram.
3. Forward and backward information flow are visually distinct and intuitive.
4. Typography and spacing feel polished enough for a showcase portfolio project.
5. Desktop scrolling feels cinematic but controlled.
6. Mobile remains readable and usable without horizontal scrolling.
7. The existing neural-network math remains correct.
8. Reduced-motion users can follow the same lesson.
9. The page still runs as a static GitHub Pages site.

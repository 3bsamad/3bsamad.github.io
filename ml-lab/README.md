# ML Visual Lab

A browser-only, scroll-driven neural-network explainer built as a visual teaching project.

The public V2 story deliberately follows **one spam email** through the complete learning cycle:

1. input features
2. weighted connections
3. ReLU
4. output logit + sigmoid
5. binary cross-entropy loss
6. backpropagation
7. gradient-descent update
8. repeated training

## What makes V2 different

The network is a **persistent SVG**, not a diagram rebuilt on every step. Neurons, connections, values and signal layers stay mounted while scroll scenes change their state.

Forward activations move through the network in blue. Gradients travel backward in violet. Parameter updates resolve in green. The visualization remains synchronized with real forward/backprop calculations from the small 4 → 3 → 1 teaching network.

The initial public narrative uses a single canonical spam email so the visual choreography can stay coherent from the first input value to the final parameter update.

## Stack

- HTML + CSS
- vanilla JavaScript ES modules
- SVG
- GSAP + ScrollTrigger for enhanced scroll choreography
- KaTeX for mathematical notation
- Node's built-in test runner for model and static contracts

Everything remains suitable for GitHub Pages.

## Architecture

- `model.js` — pure forward pass, backpropagation, gradient update and short training preview
- `network-renderer.js` — persistent SVG neurons, connections, signal pulses and semantic states
- `scene-content.js` — ordered lesson scenes and live math substitutions
- `scroll-scenes.js` — ScrollTrigger / reduced-motion scene activation
- `app.js` — application composition and view synchronization
- `styles.css` — responsive visual system and computational state styling

## Progressive enhancement

The core lesson and network initialize without animation libraries. If GSAP/ScrollTrigger is unavailable, scene activation falls back to IntersectionObserver. If KaTeX is unavailable, readable plain-text equations remain visible.

`prefers-reduced-motion` disables particle travel and long transitions while preserving the same lesson states.

## Local verification

```bash
node --check app.js
node --check model.js
node --check network-renderer.js
node --check scene-content.js
node --check scroll-scenes.js
node --test tests/*.test.mjs
```

Open `index.html` through a local static server to review the full scroll choreography.

import test from "node:test";
import assert from "node:assert/strict";
import { getNetworkGeometry, getSceneVisualState } from "../network-renderer.js";

test("desktop and compact geometries keep all major nodes inside the viewBox", () => {
  for (const mode of ["desktop", "compact"]) {
    const g = getNetworkGeometry(mode);
    const points = [...g.inputs, ...g.hidden, g.output, g.loss];
    assert.equal(g.inputs.length, 4);
    assert.equal(g.hidden.length, 3);
    for (const p of points) {
      assert.ok(p.x >= 0 && p.x <= g.width);
      assert.ok(p.y >= 0 && p.y <= g.height);
    }
  }
});

test("scene states are deterministic and semantic", () => {
  assert.equal(getSceneVisualState("scene-inputs").focus, "inputs");
  assert.equal(getSceneVisualState("scene-weights").direction, "forward");
  assert.equal(getSceneVisualState("scene-relu").focus, "hidden");
  assert.equal(getSceneVisualState("scene-loss").focus, "loss");
  assert.equal(getSceneVisualState("scene-backprop").direction, "backward");
  assert.equal(getSceneVisualState("scene-update").direction, "update");
  assert.equal(getSceneVisualState("unknown").focus, "dormant");
});

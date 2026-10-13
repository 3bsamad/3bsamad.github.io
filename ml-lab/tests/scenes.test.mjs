import test from "node:test";
import assert from "node:assert/strict";
import { SCENES, getSceneContent } from "../scene-content.js";
import { INITIAL_PARAMS, TEACHING_SAMPLE, forward, backward, updatedParams } from "../model.js";

const snapshot = forward(INITIAL_PARAMS, TEACHING_SAMPLE);
const grads = backward(INITIAL_PARAMS, snapshot);
const next = updatedParams(INITIAL_PARAMS, grads, 0.10);
const context = { snapshot, grads, params: INITIAL_PARAMS, nextParams: next, learningRate: 0.10 };

test("scene descriptors exactly match the narrative order", () => {
  assert.deepEqual(SCENES.map(s => s.id), [
    "scene-intro","scene-inputs","scene-weights","scene-relu",
    "scene-output","scene-loss","scene-backprop","scene-update","scene-training"
  ]);
  for (const scene of SCENES) {
    assert.ok(scene.title);
    assert.ok(scene.intuition);
    assert.ok(scene.rendererState);
  }
});

test("ReLU scene includes a real clamp to zero", () => {
  const c = getSceneContent("scene-relu", context);
  assert.match(c.plainMath.join(" "), /ReLU/);
  assert.ok(snapshot.z1.some(v => v < 0));
  assert.ok(snapshot.h.some(v => v === 0));
});

test("output, loss, backprop and update scenes contain live numeric math", () => {
  assert.match(getSceneContent("scene-output", context).plainMath.join(" "), /sigmoid|σ/i);
  assert.match(getSceneContent("scene-loss", context).plainMath.join(" "), /BCE|ln|loss/i);
  assert.match(getSceneContent("scene-backprop", context).plainMath.join(" "), /ŷ - y|gradient/i);
  const update = getSceneContent("scene-update", context).plainMath.join(" ");
  assert.match(update, /0\.10/);
  assert.match(update, /new|Δ|eta|η/i);
});

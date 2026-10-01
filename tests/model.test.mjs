import test from "node:test";
import assert from "node:assert/strict";

import {
  TEACHING_SAMPLE,
  INITIAL_PARAMS,
  forward,
  backward,
  updatedParams
} from "../model.js";

test("teaching sample is one four-feature spam example", () => {
  assert.equal(TEACHING_SAMPLE.y, 1);
  assert.equal(TEACHING_SAMPLE.x.length, 4);
  assert.match(TEACHING_SAMPLE.email, /urgent|bank|link/i);
  for (const value of TEACHING_SAMPLE.x) {
    assert.ok(value >= 0 && value <= 1);
  }
});

test("forward pass returns finite 4-3-1 BCE snapshot", () => {
  const result = forward(INITIAL_PARAMS, TEACHING_SAMPLE);
  assert.equal(result.z1.length, 3);
  assert.equal(result.h.length, 3);
  assert.ok(result.z1.every(Number.isFinite));
  assert.ok(result.h.every(Number.isFinite));
  assert.ok(Number.isFinite(result.z2));
  assert.ok(result.yHat > 0 && result.yHat < 1);
  assert.ok(Number.isFinite(result.loss) && result.loss > 0);
});

test("backward pass returns gradients with matching parameter shapes", () => {
  const snapshot = forward(INITIAL_PARAMS, TEACHING_SAMPLE);
  const grads = backward(INITIAL_PARAMS, snapshot);
  assert.equal(grads.dW1.length, 3);
  assert.deepEqual(grads.dW1.map(row => row.length), [4, 4, 4]);
  assert.equal(grads.dW2.length, 3);
  assert.equal(grads.db1.length, 3);
  assert.ok(Number.isFinite(grads.db2));
});

test("updatedParams is immutable and applies old minus eta times gradient", () => {
  const snapshot = forward(INITIAL_PARAMS, TEACHING_SAMPLE);
  const grads = backward(INITIAL_PARAMS, snapshot);
  const before = structuredClone(INITIAL_PARAMS);
  const next = updatedParams(INITIAL_PARAMS, grads, 0.10);

  assert.deepEqual(INITIAL_PARAMS, before);
  assert.equal(
    next.W2[0],
    INITIAL_PARAMS.W2[0] - 0.10 * grads.dW2[0]
  );
  assert.equal(
    next.W1[1][2],
    INITIAL_PARAMS.W1[1][2] - 0.10 * grads.dW1[1][2]
  );
});

test("one gradient step lowers BCE loss for the teaching sample", () => {
  const before = forward(INITIAL_PARAMS, TEACHING_SAMPLE);
  const grads = backward(INITIAL_PARAMS, before);
  const nextParams = updatedParams(INITIAL_PARAMS, grads, 0.10);
  const after = forward(nextParams, TEACHING_SAMPLE);

  assert.ok(after.loss < before.loss);
});

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const app = readFileSync(new URL("../app.js", import.meta.url), "utf8");
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");

test("app composes the V2 modules and only enables dimmed story states after JS starts", () => {
  assert.match(app, /document\.documentElement\.classList\.add\(["']js-ready["']\)/);
  for (const module of ["./model.js","./network-renderer.js","./scene-content.js","./scroll-scenes.js"]) {
    assert.match(app, new RegExp(module.replace(".", "\\.")));
  }
  assert.doesNotMatch(app, /sampleSelect|datasetTableBody/);
});

test("accessibility and final-scene affordances exist", () => {
  assert.match(html, /aria-current/);
  assert.match(html, /id=["']lossChart["'][^>]+aria-label=/s);
  assert.match(html, /aria-live=["']polite["']/);
});

test("README documents the scroll-driven single-sample V2", () => {
  assert.match(readme, /scroll/i);
  assert.match(readme, /one|single/i);
  assert.match(readme, /persistent SVG/i);
});


test("scroll scenes use a symmetric boundary when reversing", () => {
  const scroll = readFileSync(new URL("../scroll-scenes.js", import.meta.url), "utf8");
  assert.match(scroll, /onLeaveBack:\s*\(\)\s*=>\s*activate\(/);
  assert.doesNotMatch(scroll, /onEnterBack:\s*\(\)\s*=>\s*activate\(section\.id\)/);
});

test("renderer cancels in-flight motion before applying a canonical scene", () => {
  const renderer = readFileSync(new URL("../network-renderer.js", import.meta.url), "utf8");
  assert.match(renderer, /function cancelMotion\(/);
  assert.match(renderer, /function applyScene[\s\S]*?cancelMotion\(\)[\s\S]*?clearStates\(\)/);
});
